"""Session — stateful multi-turn wrapper for any DSPy module.

Key properties:
- Adapter-agnostic (works with Chat/JSON/XML/Template adapters)
- Linearization-first (each turn snapshots history for optimizer-ready examples)
- Program-safe (supports composed dspy.Module programs via predictor-level injection)
- Optimizer-safe (explicit history enables stateless pass-through)

v2.5 additions:
- Production-safe external state objects (`SessionState`) for contextvar-isolated concurrency
- Optional state binding context manager (`session.use_state(state)`) for shared-blueprint serving
- Projection helpers (`get_outer_history`, `get_node_memory`, ...)
"""

from __future__ import annotations

import contextvars
import copy
import inspect
import json
import logging
import threading
import types
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, get_args, get_origin, get_type_hints

import dspy
from dspy.adapters.types.history import History

logger = logging.getLogger(__name__)

# Context variable used by wrapped predictors to obtain the session history
_CURRENT_SESSION_HISTORY: contextvars.ContextVar[History | None] = contextvars.ContextVar(
    "dspy_session_history", default=None
)

# Extra projection context (TemplateAdapter power users)
_CURRENT_OUTER_HISTORY: contextvars.ContextVar[History | None] = contextvars.ContextVar(
    "dspy_session_outer_history", default=None
)
_CURRENT_NODE_MEMORY: contextvars.ContextVar[str] = contextvars.ContextVar(
    "dspy_session_node_memory", default=""
)

# Active per-session external state bindings (for concurrent, shared-blueprint execution)
_ACTIVE_SESSION_STATES: contextvars.ContextVar[dict[int, "SessionState"] | None] = contextvars.ContextVar(
    "dspy_session_active_states", default=None
)

# Active execution metadata (for advanced projection helpers)
_ACTIVE_SESSION_ROOT: contextvars.ContextVar["Session | None"] = contextvars.ContextVar(
    "dspy_session_active_root", default=None
)
_ACTIVE_SESSION_STACK: contextvars.ContextVar[tuple["Session", ...]] = contextvars.ContextVar(
    "dspy_session_active_stack", default=()
)


@dataclass
class Turn:
    """One session turn."""

    index: int
    inputs: dict[str, Any]
    outputs: dict[str, Any]
    history_snapshot: History
    score: float | None = None


@dataclass
class SessionState:
    """Serializable per-user state for a Session blueprint.

    This lets you keep one global `Session` (module weights + wrapping logic)
    while binding request/user-specific turns at runtime via `session.use_state(state)`.

    `node_states` stores per-node ledgers keyed by dotted session path
    (for example: ``"planner"`` or ``"researcher.writer"``).
    """

    turns: list[Turn] = field(default_factory=list)
    initial_history: History | None = None
    l2_memory: str = ""
    node_states: dict[str, "SessionState"] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.turns)

    def fork(self) -> "SessionState":
        return SessionState(
            turns=copy.deepcopy(self.turns),
            initial_history=copy.deepcopy(self.initial_history),
            l2_memory=self.l2_memory,
            node_states={k: v.fork() for k, v in self.node_states.items()},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "turns": [
                {
                    "index": t.index,
                    "inputs": _safe_serialize(t.inputs),
                    "outputs": _safe_serialize(t.outputs),
                    "history_snapshot": _safe_serialize_value(t.history_snapshot),
                    "score": t.score,
                }
                for t in self.turns
            ],
            "initial_history": _safe_serialize_value(self.initial_history) if self.initial_history is not None else None,
            "l2_memory": self.l2_memory,
            "node_states": {path: state.to_dict() for path, state in self.node_states.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionState":
        version = data.get("version", 1)
        if version != 1:
            raise ValueError(f"Unsupported SessionState version: {version}")

        initial_history_raw = data.get("initial_history")
        initial_history = None
        if isinstance(initial_history_raw, dict) and "messages" in initial_history_raw:
            initial_history = History(messages=initial_history_raw["messages"])

        turns: list[Turn] = []
        for t in data.get("turns", []):
            hs = t.get("history_snapshot")
            if isinstance(hs, dict) and "messages" in hs:
                snapshot = History(messages=hs["messages"])
            else:
                snapshot = History(messages=[])
            turns.append(
                Turn(
                    index=t.get("index", len(turns)),
                    inputs=t.get("inputs", {}),
                    outputs=t.get("outputs", {}),
                    history_snapshot=snapshot,
                    score=t.get("score"),
                )
            )

        node_states_raw = data.get("node_states") or {}
        node_states: dict[str, SessionState] = {}
        if isinstance(node_states_raw, dict):
            for path, raw in node_states_raw.items():
                if isinstance(raw, dict):
                    node_states[path] = cls.from_dict(raw)

        return cls(
            turns=turns,
            initial_history=initial_history,
            l2_memory=str(data.get("l2_memory", "")),
            node_states=node_states,
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, default=str))

    @classmethod
    def load_from(cls, path: str | Path) -> "SessionState":
        return cls.from_dict(json.loads(Path(path).read_text()))


class _StateBinding:
    """Sync/async context manager for binding external state to a Session."""

    def __init__(self, session: "Session", state: SessionState):
        self._session = session
        self._state = state
        self._token: contextvars.Token | None = None

    def __enter__(self) -> SessionState:
        current = _ACTIVE_SESSION_STATES.get()
        mapping = dict(current) if current is not None else {}
        mapping[id(self._session)] = self._state
        self._token = _ACTIVE_SESSION_STATES.set(mapping)
        return self._state

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._token is not None:
            _ACTIVE_SESSION_STATES.reset(self._token)
            self._token = None

    async def __aenter__(self) -> SessionState:
        return self.__enter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        self.__exit__(exc_type, exc_val, exc_tb)


class Session(dspy.Module):
    """Stateful multi-turn wrapper for DSPy modules/programs.

    Supports:
    - Direct predictor wrapping (`dspy.Predict`, `dspy.ChainOfThought`, ...)
    - Program wrapping (`dspy.Module` containing nested predictors)

    For composed programs, session history is injected into nested predictors using
    wrapped predictor forward/aforward methods + a contextvar.

    Advanced serving mode:
    - Keep one global Session object and bind per-request/per-user `SessionState`
      via `with session.use_state(state): ...`.
    """

    def __init__(
        self,
        module: dspy.Module,
        *,
        history_field: str = "history",
        max_turns: int | None = None,
        max_stored_turns: int | None = None,
        exclude_fields: set[str] | None = None,
        input_field_override: set[str] | None = None,
        history_input_fields: set[str] | None = None,
        initial_history: History | None = None,
        history_policy: Literal["override", "use_if_provided", "replace_session"] = "override",
        on_metric_error: Literal["zero", "raise"] = "zero",
        strict_history_annotation: bool = False,
        copy_mode: Literal["deep", "shallow", "none"] = "deep",
        lock: Literal["none", "thread", "async"] = "none",
        on_turn: Callable[["Session", Turn], None] | None = None,
        # memory-policy axes (used by with_memory / recursive graphs)
        isolation: Literal["isolated", "shared"] = "isolated",
        lifespan: Literal["persistent", "episodic", "stateless"] = "persistent",
        consolidator: dspy.Module | None = None,
        session_path: str = "root",
    ):
        super().__init__()

        self.on_turn = on_turn
        self.copy_mode = copy_mode
        self.module = self._clone_module(module)
        self.history_field = history_field
        self.max_turns = max_turns
        self.max_stored_turns = max_stored_turns

        if history_input_fields is not None and input_field_override is not None:
            raise ValueError("Provide either history_input_fields or input_field_override, not both.")
        self.history_input_fields = history_input_fields if history_input_fields is not None else input_field_override

        self.exclude_fields = set(exclude_fields or [])
        self.exclude_fields.add(history_field)

        # Legacy config field retained for backward compatibility and serialization.
        self._initial_history = initial_history
        self.history_policy = history_policy
        self.on_metric_error = on_metric_error
        self.strict_history_annotation = strict_history_annotation

        # Memory-policy axes
        if isolation not in ("isolated", "shared"):
            raise ValueError("isolation must be 'isolated' or 'shared'.")
        if lifespan not in ("persistent", "episodic", "stateless"):
            raise ValueError("lifespan must be 'persistent', 'episodic', or 'stateless'.")
        self.isolation: Literal["isolated", "shared"] = isolation
        self.lifespan: Literal["persistent", "episodic", "stateless"] = lifespan
        self.consolidator = consolidator
        self._session_path = session_path

        # Default local state (notebook mode).
        self._default_state = SessionState(initial_history=copy.deepcopy(initial_history))

        # lock config
        self._thread_lock: threading.Lock | None = None
        self._async_lock: Any = None
        self._lock_mode = lock
        if lock == "thread":
            self._thread_lock = threading.Lock()

        # Predictor wrapping metadata
        self._predictor_history_fields: dict[int, str] = {}

        # Prepare wrapped module: patch signatures + wrap predictor calls
        self._prepare_predictor_injection()

        # Whether top-level program forward accepts history kwarg
        self._module_accepts_history = self._detect_module_accepts_history()

        # Optional cache of recursive child sessions (with_memory)
        self._child_sessions: dict[str, Session] = {}

    # ------------------------------------------------------------------
    # External state binding API (production serving)
    # ------------------------------------------------------------------

    def new_state(self, *, initial_history: History | None = None, l2_memory: str = "") -> SessionState:
        """Create an empty external state container for this session blueprint."""
        if initial_history is None:
            initial_history = copy.deepcopy(self._default_state.initial_history)
        return SessionState(initial_history=initial_history, l2_memory=l2_memory)

    def use_state(self, state: SessionState) -> _StateBinding:
        """Bind a per-user/per-request state for the duration of a (sync/async) context."""
        if not isinstance(state, SessionState):
            raise TypeError(f"Expected SessionState, got {type(state).__name__}")
        return _StateBinding(self, state)

    def _active_state(self) -> SessionState:
        mapping = _ACTIVE_SESSION_STATES.get()
        if mapping is not None:
            # Direct binding (session.use_state(state) on this specific session)
            bound = mapping.get(id(self))
            if bound is not None:
                return bound

            # Indirect binding via active root: route child sessions to per-path node states.
            root = _ACTIVE_SESSION_ROOT.get()
            if root is not None:
                root_state = mapping.get(id(root))
                if root_state is not None:
                    if self is root:
                        return root_state
                    path = getattr(self, "_session_path", "")
                    if path and path != "root":
                        child_state = root_state.node_states.get(path)
                        if child_state is None:
                            child_state = SessionState(
                                initial_history=copy.deepcopy(self._default_state.initial_history),
                                l2_memory=self._default_state.l2_memory,
                            )
                            root_state.node_states[path] = child_state
                        return child_state

        return self._default_state

    def _iter_child_sessions(self) -> dict[str, "Session"]:
        """Discover nested Session objects under this session's wrapped module tree.

        Keys are dotted attribute paths relative to ``self.module``.
        """
        found: dict[str, Session] = {}
        visited_objs: set[int] = set()

        def walk(obj: Any, prefix: str) -> None:
            oid = id(obj)
            if oid in visited_objs:
                return
            visited_objs.add(oid)

            # Dive into wrapped modules and discover embedded Session children.
            attrs = getattr(obj, "__dict__", None)
            if not isinstance(attrs, dict):
                return

            for name, value in attrs.items():
                if name.startswith("_"):
                    continue

                path = f"{prefix}.{name}" if prefix else name

                if isinstance(value, Session):
                    found[path] = value
                    walk(value.module, path)
                elif isinstance(value, dspy.Module):
                    walk(value, path)

        walk(self.module, "")
        return found

    def _set_nested_attr(self, path: str, value: Any) -> None:
        """Set dotted attribute path on ``self.module``."""
        parts = path.split(".")
        obj: Any = self.module
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)

    def _configure_policy(
        self,
        *,
        isolation: Literal["isolated", "shared"] | None = None,
        lifespan: Literal["persistent", "episodic", "stateless"] | None = None,
        consolidator: dspy.Module | None | types.EllipsisType = ...,
    ) -> None:
        if isolation is not None:
            if isolation not in ("isolated", "shared"):
                raise ValueError("isolation must be 'isolated' or 'shared'.")
            self.isolation = isolation
        if lifespan is not None:
            if lifespan not in ("persistent", "episodic", "stateless"):
                raise ValueError("lifespan must be 'persistent', 'episodic', or 'stateless'.")
            self.lifespan = lifespan
        if consolidator is not ...:
            self.consolidator = consolidator

    def _prepare_recursive_children(
        self,
        *,
        recursive: bool | Literal["predictors"] = True,
        include: set[str] | None = None,
        exclude: set[str] | None = None,
        where: Callable[[str, Any], bool] | None = None,
        child_configs: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """Recursively wrap predictors as child sessions (user-space proxy polyfill)."""
        if not recursive:
            return

        include = set(include or [])
        exclude = set(exclude or [])
        child_configs = child_configs or {}

        def is_allowed(path: str, obj: Any) -> bool:
            if include:
                if path not in include and not any(path.startswith(prefix + ".") for prefix in include):
                    return False
            if exclude:
                if path in exclude or any(path.startswith(prefix + ".") for prefix in exclude):
                    return False
            if where is not None and not where(path, obj):
                return False
            return True

        # Adopt already-sessionified children first (avoid double wrapping).
        for path, child in self._iter_child_sessions().items():
            if not is_allowed(path, child):
                continue

            cfg = child_configs.get(path, {})
            if cfg.get("enabled", True) is False:
                # Explicit unwrap: keep this branch stateless by removing the Session proxy.
                self._set_nested_attr(path, child.module)
                continue

            child._session_path = path
            child._configure_policy(
                isolation=cfg.get("isolation"),
                lifespan=cfg.get("lifespan"),
                consolidator=cfg.get("consolidator", ...),
            )
            self._child_sessions[path] = child

        # Wrap raw predictors by path.
        for path, predictor in list(self.module.named_predictors()):
            # Skip root/self and internals under already sessionified children.
            if predictor is self.module:
                continue
            if ".module" in path or path.startswith("module"):
                continue
            if not is_allowed(path, predictor):
                continue

            # If already replaced/adopted as Session by this path, skip.
            if path in self._child_sessions:
                continue

            cfg = child_configs.get(path, {})
            if cfg.get("enabled", True) is False:
                continue

            child = Session(
                predictor,
                history_field=self.history_field,
                max_turns=self.max_turns,
                max_stored_turns=self.max_stored_turns,
                exclude_fields=set(self.exclude_fields),
                history_input_fields=(
                    set(self.history_input_fields) if self.history_input_fields is not None else None
                ),
                initial_history=copy.deepcopy(cfg.get("initial_history")),
                history_policy=self.history_policy,
                on_metric_error=self.on_metric_error,
                strict_history_annotation=self.strict_history_annotation,
                copy_mode="none",
                lock=self._lock_mode,
                on_turn=None,
                isolation=cfg.get("isolation", self.isolation),
                lifespan=cfg.get("lifespan", self.lifespan),
                consolidator=cfg.get("consolidator", self.consolidator),
                session_path=path,
            )
            self._set_nested_attr(path, child)
            self._child_sessions[path] = child

    # ------------------------------------------------------------------
    # Module cloning
    # ------------------------------------------------------------------

    def _clone_module(self, module: dspy.Module) -> dspy.Module:
        if self.copy_mode == "deep":
            return copy.deepcopy(module)
        if self.copy_mode == "shallow":
            return copy.copy(module)
        if self.copy_mode == "none":
            return module
        raise ValueError(f"Unknown copy_mode: {self.copy_mode}")

    # ------------------------------------------------------------------
    # Low-level attribute access helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_attr_quiet(obj: Any, name: str) -> Any | None:
        """Get attribute while bypassing dspy.Module.__getattribute__ warnings.

        DSPy warns when `forward` is accessed directly outside `__call__`.
        We still need read access for signature inspection/wrapping, so use
        `object.__getattribute__` to bypass that warning path.
        """
        try:
            return object.__getattribute__(obj, name)
        except AttributeError:
            return None

    # ------------------------------------------------------------------
    # Predictor preparation
    # ------------------------------------------------------------------

    def _prepare_predictor_injection(self) -> None:
        """Patch nested predictors to support history and auto-inject from contextvar."""
        for _, predictor in self.module.named_predictors():
            # Skip already-sessionified descendants to avoid double wrapping.
            if isinstance(predictor, Session):
                continue
            field_name = self._ensure_predictor_history_field(predictor)
            self._predictor_history_fields[id(predictor)] = field_name
            self._wrap_predictor(predictor, field_name)

    def _ensure_predictor_history_field(self, predictor: dspy.Predict) -> str:
        """Ensure predictor signature has a history field and return its field name."""
        sig = predictor.signature

        # Find existing history-like field
        existing_name = None
        for name, finfo in sig.input_fields.items():
            if _is_history_annotation(finfo.annotation, strict=self.strict_history_annotation):
                existing_name = name
                break

        if existing_name is not None:
            if existing_name != self.history_field:
                warnings.warn(
                    f"Predictor already has history-like field '{existing_name}'. "
                    f"Session requested '{self.history_field}'. Using '{existing_name}'.",
                    stacklevel=3,
                )
            # For root predictor sessions, align session.history_field with the
            # predictor's actual history field name.
            if predictor is self.module:
                self.history_field = existing_name
                self.exclude_fields.add(existing_name)
            return existing_name

        # Add requested history field
        new_sig = sig.append(
            self.history_field,
            dspy.InputField(desc="Conversation history", default=None),
            type_=History,
        )
        predictor.signature = new_sig
        return self.history_field

    def _wrap_predictor(self, predictor: dspy.Predict, field_name: str) -> None:
        """Wrap predictor.forward/aforward to inject history from contextvar when absent."""
        if getattr(predictor, "_dspy_session_wrapped", False):
            return

        orig_forward = self._get_attr_quiet(predictor, "forward")
        if orig_forward is None:
            raise AttributeError(f"Predictor {type(predictor).__name__} has no forward method.")

        def wrapped_forward(_self, **kwargs):
            if field_name not in kwargs:
                h = _CURRENT_SESSION_HISTORY.get()
                if h is not None:
                    kwargs[field_name] = h
            return orig_forward(**kwargs)

        predictor.forward = types.MethodType(wrapped_forward, predictor)

        orig_aforward = self._get_attr_quiet(predictor, "aforward")
        if orig_aforward is not None:

            async def wrapped_aforward(_self, **kwargs):
                if field_name not in kwargs:
                    h = _CURRENT_SESSION_HISTORY.get()
                    if h is not None:
                        kwargs[field_name] = h
                return await orig_aforward(**kwargs)

            predictor.aforward = types.MethodType(wrapped_aforward, predictor)

        predictor._dspy_session_wrapped = True

    def _detect_module_accepts_history(self) -> bool:
        """Whether top-level module.forward can accept history kwarg.

        Also detects history-like parameters by annotation and aligns the session
        history_field to that name when found.
        """
        forward_method = self._get_attr_quiet(self.module, "forward")
        if forward_method is None:
            return False

        try:
            sig = inspect.signature(forward_method)
        except (ValueError, TypeError):
            return False

        params = sig.parameters

        # direct name match
        if self.history_field in params:
            return True

        # Resolve postponed annotations (from __future__ import annotations)
        resolved_hints: dict[str, Any] = {}
        try:
            resolved_hints = get_type_hints(forward_method)
        except Exception:
            resolved_hints = {}

        # annotation-based match (e.g., parameter named `chat_history`)
        for name, p in params.items():
            ann = resolved_hints.get(name, p.annotation)
            if ann is inspect._empty:
                continue
            if _is_history_annotation(ann, strict=self.strict_history_annotation):
                if name != self.history_field:
                    warnings.warn(
                        f"Module forward() uses history-like parameter '{name}'. "
                        f"Session requested '{self.history_field}'. Using '{name}'.",
                        stacklevel=3,
                    )
                    self.history_field = name
                    self.exclude_fields.add(name)
                return True

        # **kwargs fallback
        return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())

    # ------------------------------------------------------------------
    # Core calling
    # ------------------------------------------------------------------

    def forward(self, **kwargs) -> Any:
        """Run one turn.

        Behavior when caller passes explicit history:
        - override (default): stateless pass-through, do NOT record turn
        - use_if_provided: run with provided history and record turn
        - replace_session: replace state seed with provided history, clear turns, then run statefully
        """
        if self._thread_lock is not None:
            with self._thread_lock:
                return self._forward_impl(**kwargs)
        return self._forward_impl(**kwargs)

    def _forward_impl(self, **kwargs) -> Any:
        state = self._active_state()
        explicit_history = kwargs.pop(self.history_field, None)
        is_root_call = _ACTIVE_SESSION_ROOT.get() is None

        if explicit_history is not None and not isinstance(explicit_history, History):
            raise TypeError(
                f"Expected {self.history_field} to be dspy.History, got {type(explicit_history).__name__}."
            )

        skip_finalize = False

        # explicit-history policy handling
        if explicit_history is not None and self.history_policy == "replace_session":
            state.initial_history = explicit_history
            state.turns.clear()
            run_history = self._build_history()
            record_turn = self.lifespan != "stateless"
        elif explicit_history is not None and self.history_policy == "use_if_provided":
            run_history = explicit_history
            record_turn = self.lifespan != "stateless"
        elif explicit_history is not None and self.history_policy == "override":
            # optimizer/stateless replay mode
            run_history = explicit_history
            record_turn = False
            skip_finalize = True
        else:
            # topology handling when no explicit history is provided
            if self.isolation == "shared" and not is_root_call:
                shared_history = _CURRENT_OUTER_HISTORY.get()
                if shared_history is None:
                    root = _ACTIVE_SESSION_ROOT.get()
                    shared_history = root._build_history() if root is not None else self._build_history()
                run_history = shared_history
                record_turn = False
            else:
                run_history = self._build_history()
                record_turn = self.lifespan != "stateless"

        try:
            result = self._invoke_inner(run_history, kwargs)

            if record_turn:
                self._record_turn(kwargs, result, run_history)

            return result
        finally:
            if is_root_call and not skip_finalize:
                self._finalize_macro_turn()

    async def aforward(self, **kwargs) -> Any:
        """Async run."""
        if self._lock_mode == "async":
            if self._async_lock is None:
                import asyncio

                self._async_lock = asyncio.Lock()
            async with self._async_lock:
                return await self._aforward_impl(**kwargs)
        return await self._aforward_impl(**kwargs)

    async def _aforward_impl(self, **kwargs) -> Any:
        state = self._active_state()
        explicit_history = kwargs.pop(self.history_field, None)
        is_root_call = _ACTIVE_SESSION_ROOT.get() is None

        if explicit_history is not None and not isinstance(explicit_history, History):
            raise TypeError(
                f"Expected {self.history_field} to be dspy.History, got {type(explicit_history).__name__}."
            )

        skip_finalize = False

        if explicit_history is not None and self.history_policy == "replace_session":
            state.initial_history = explicit_history
            state.turns.clear()
            run_history = self._build_history()
            record_turn = self.lifespan != "stateless"
        elif explicit_history is not None and self.history_policy == "use_if_provided":
            run_history = explicit_history
            record_turn = self.lifespan != "stateless"
        elif explicit_history is not None and self.history_policy == "override":
            run_history = explicit_history
            record_turn = False
            skip_finalize = True
        else:
            if self.isolation == "shared" and not is_root_call:
                shared_history = _CURRENT_OUTER_HISTORY.get()
                if shared_history is None:
                    root = _ACTIVE_SESSION_ROOT.get()
                    shared_history = root._build_history() if root is not None else self._build_history()
                run_history = shared_history
                record_turn = False
            else:
                run_history = self._build_history()
                record_turn = self.lifespan != "stateless"

        try:
            result = await self._ainvoke_inner(run_history, kwargs)

            if record_turn:
                self._record_turn(kwargs, result, run_history)

            return result
        finally:
            if is_root_call and not skip_finalize:
                self._finalize_macro_turn()

    async def acall(self, **kwargs) -> Any:
        return await self.aforward(**kwargs)

    def _invoke_inner(self, history: History, kwargs: dict[str, Any]) -> Any:
        """Invoke wrapped module with projection contextvars."""
        call_kwargs = dict(kwargs)
        if self._module_accepts_history:
            call_kwargs[self.history_field] = history

        outer = _CURRENT_OUTER_HISTORY.get()
        if outer is None:
            outer = history

        current_root = _ACTIVE_SESSION_ROOT.get()
        root = self if current_root is None else current_root
        stack = _ACTIVE_SESSION_STACK.get()

        token_root = _ACTIVE_SESSION_ROOT.set(root)
        token_stack = _ACTIVE_SESSION_STACK.set((*stack, self))
        token_history = _CURRENT_SESSION_HISTORY.set(history)
        token_outer = _CURRENT_OUTER_HISTORY.set(outer)
        token_l2 = _CURRENT_NODE_MEMORY.set(self._active_state().l2_memory)
        try:
            return self.module(**call_kwargs)
        finally:
            _CURRENT_NODE_MEMORY.reset(token_l2)
            _CURRENT_OUTER_HISTORY.reset(token_outer)
            _CURRENT_SESSION_HISTORY.reset(token_history)
            _ACTIVE_SESSION_STACK.reset(token_stack)
            _ACTIVE_SESSION_ROOT.reset(token_root)

    async def _ainvoke_inner(self, history: History, kwargs: dict[str, Any]) -> Any:
        call_kwargs = dict(kwargs)
        if self._module_accepts_history:
            call_kwargs[self.history_field] = history

        outer = _CURRENT_OUTER_HISTORY.get()
        if outer is None:
            outer = history

        current_root = _ACTIVE_SESSION_ROOT.get()
        root = self if current_root is None else current_root
        stack = _ACTIVE_SESSION_STACK.get()

        token_root = _ACTIVE_SESSION_ROOT.set(root)
        token_stack = _ACTIVE_SESSION_STACK.set((*stack, self))
        token_history = _CURRENT_SESSION_HISTORY.set(history)
        token_outer = _CURRENT_OUTER_HISTORY.set(outer)
        token_l2 = _CURRENT_NODE_MEMORY.set(self._active_state().l2_memory)
        try:
            if hasattr(self.module, "acall"):
                return await self.module.acall(**call_kwargs)
            if hasattr(self.module, "aforward"):
                return await self.module.aforward(**call_kwargs)
            raise TypeError(
                f"{type(self.module).__name__} does not support async. "
                "Use sync call or provide async-capable module."
            )
        finally:
            _CURRENT_NODE_MEMORY.reset(token_l2)
            _CURRENT_OUTER_HISTORY.reset(token_outer)
            _CURRENT_SESSION_HISTORY.reset(token_history)
            _ACTIVE_SESSION_STACK.reset(token_stack)
            _ACTIVE_SESSION_ROOT.reset(token_root)

    def _record_turn(self, kwargs: dict[str, Any], result: Any, history: History) -> None:
        """Record completed turn."""
        state = self._active_state()
        inputs_for_record = copy.deepcopy(self._filter_recordable_inputs(kwargs))
        outputs_for_record = copy.deepcopy(self._extract_outputs(result))
        turn = Turn(
            index=len(state.turns),
            inputs=inputs_for_record,
            outputs=outputs_for_record,
            history_snapshot=history,
        )
        state.turns.append(turn)

        if self.max_stored_turns is not None and len(state.turns) > self.max_stored_turns:
            state.turns[:] = state.turns[-self.max_stored_turns :]

        if self.on_turn is not None:
            try:
                self.on_turn(self, turn)
            except Exception as e:
                logger.warning("on_turn callback error: %s", e)

    def _finalize_macro_turn(self) -> None:
        """Apply lifespan policies and run optional consolidators at root turn boundary."""
        root = self
        children = root._iter_child_sessions()

        mapping = _ACTIVE_SESSION_STATES.get()
        has_bound_root = mapping is not None and id(root) in mapping
        root_state = mapping[id(root)] if has_bound_root else root._default_state

        def state_for(path: str, session: Session) -> SessionState:
            if session is root:
                return root_state

            # Notebook/local mode: children own independent default states.
            if not has_bound_root:
                return session._default_state

            # Bound mode: children route to root_state.node_states[path].
            child_state = root_state.node_states.get(path)
            if child_state is None:
                child_state = SessionState(
                    initial_history=copy.deepcopy(session._default_state.initial_history),
                    l2_memory=session._default_state.l2_memory,
                )
                root_state.node_states[path] = child_state
            return child_state

        sessions: list[tuple[str, Session]] = [("root", root)] + list(children.items())
        for path, session in sessions:
            state = state_for(path, session)

            if session.lifespan == "persistent":
                continue

            if session.lifespan == "episodic":
                if session.consolidator is not None and state.turns:
                    transcript = _serialize_turns(state.turns)
                    try:
                        pred = session.consolidator(
                            past_memory=state.l2_memory,
                            episode_transcript=transcript,
                        )
                        updated = _extract_updated_memory(pred)
                        if updated is not None:
                            state.l2_memory = str(updated)
                    except Exception as e:
                        logger.warning(
                            "consolidator error on session '%s' (%s): %s",
                            session._session_path,
                            type(session.module).__name__,
                            e,
                        )

                state.turns.clear()
                continue

            if session.lifespan == "stateless":
                state.turns.clear()

    # ------------------------------------------------------------------
    # Input/output filtering
    # ------------------------------------------------------------------

    def _filter_recordable_inputs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Filter runtime kwargs to avoid polluting examples with non-signature keys."""
        # If wrapped root is Predict, keep only signature inputs (excluding history field)
        if isinstance(self.module, dspy.Predict):
            allowed = [k for k in self.module.signature.input_fields.keys() if k != self.history_field]
            return {k: v for k, v in kwargs.items() if k in allowed}

        # For generic programs, if forward has explicit params and no **kwargs, keep only those
        forward_method = self._get_attr_quiet(self.module, "forward")
        if forward_method is not None:
            try:
                sig = inspect.signature(forward_method)
                params = sig.parameters
                has_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
                if not has_var_kwargs:
                    allowed = {
                        name
                        for name, p in params.items()
                        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
                    }
                    allowed.discard(self.history_field)
                    return {k: v for k, v in kwargs.items() if k in allowed}
            except (ValueError, TypeError):
                pass

        # Fallback: keep all kwargs except history
        return {k: v for k, v in kwargs.items() if k != self.history_field}

    def _extract_outputs(self, result: Any) -> dict[str, Any]:
        outputs: dict[str, Any] = {}
        if hasattr(result, "keys"):
            for key in result.keys():
                outputs[key] = result[key]
            return outputs
        if isinstance(result, dict):
            return dict(result)

        # Fallback for non-dict outputs
        if hasattr(result, "model_dump"):
            try:
                d = result.model_dump()
                if isinstance(d, dict):
                    return d
            except Exception:
                pass
        if hasattr(result, "dict"):
            try:
                d = result.dict()
                if isinstance(d, dict):
                    return d
            except Exception:
                pass

        # If root is Predict, try signature fields
        if isinstance(self.module, dspy.Predict):
            for name in self.module.signature.output_fields:
                if hasattr(result, name):
                    outputs[name] = getattr(result, name)

        # Last resort
        if not outputs:
            outputs["output"] = result
        return outputs

    # ------------------------------------------------------------------
    # Manual edits
    # ------------------------------------------------------------------

    def add_turn(self, inputs: dict[str, Any], outputs: dict[str, Any]) -> Turn:
        state = self._active_state()
        history = self._build_history()
        turn = Turn(
            index=len(state.turns),
            inputs=copy.deepcopy(inputs),
            outputs=copy.deepcopy(outputs),
            history_snapshot=history,
        )
        state.turns.append(turn)
        return turn

    def pop_turn(self) -> Turn | None:
        state = self._active_state()
        if state.turns:
            return state.turns.pop()
        return None

    def undo(self, steps: int = 1) -> list[Turn]:
        removed: list[Turn] = []
        for _ in range(max(0, steps)):
            t = self.pop_turn()
            if t is None:
                break
            removed.append(t)
        return removed

    def update_module(self, module: dspy.Module) -> None:
        self.module = self._clone_module(module)
        self._predictor_history_fields.clear()
        self._prepare_predictor_injection()
        self._module_accepts_history = self._detect_module_accepts_history()

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def _build_history(self) -> History:
        state = self._active_state()
        messages: list[dict[str, Any]] = []

        seed = state.initial_history if state.initial_history is not None else self._initial_history
        if seed is not None:
            messages.extend(seed.messages)

        for turn in state.turns:
            msg: dict[str, Any] = {}
            for k, v in turn.inputs.items():
                if k in self.exclude_fields:
                    continue
                if self.history_input_fields is None or k in self.history_input_fields:
                    msg[k] = v
            for k, v in turn.outputs.items():
                if k not in self.exclude_fields:
                    msg[k] = v
            messages.append(msg)

        if self.max_turns is not None:
            messages = messages[-self.max_turns :]

        return History(messages=messages)

    @property
    def session_history(self) -> History:
        return self._build_history()

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._active_state().turns.clear()

    def fork(self) -> "Session":
        new = copy.copy(self)
        new.module = self._clone_module(self.module)
        new._default_state = self._active_state().fork()
        new.exclude_fields = set(self.exclude_fields)
        new.history_input_fields = (
            set(self.history_input_fields) if self.history_input_fields is not None else None
        )
        # keep legacy mirror in sync with default state
        new._initial_history = copy.deepcopy(new._default_state.initial_history)
        if self._thread_lock is not None:
            new._thread_lock = threading.Lock()
        if self._async_lock is not None:
            new._async_lock = None
        new._predictor_history_fields = dict(self._predictor_history_fields)
        return new

    @property
    def turns(self) -> list[Turn]:
        return list(self._active_state().turns)

    def __len__(self) -> int:
        return len(self._active_state().turns)

    # ------------------------------------------------------------------
    # Scoring/linearization
    # ------------------------------------------------------------------

    def score(
        self,
        metric: Callable,
        gold: dspy.Example | None = None,
    ) -> list[float]:
        """Score each turn.

        If gold is None, each turn's own outputs are included in the example labels.
        """
        turns = self._active_state().turns
        scores: list[float] = []

        # Determine call strategy once (avoid inspect in loop)
        call_with_trace: bool
        try:
            metric(dspy.Example(), dspy.Prediction(), None)
            call_with_trace = True
        except TypeError:
            call_with_trace = False
        except Exception:
            # metric likely expects real fields; fallback to 3-arg and handle per-turn exceptions
            call_with_trace = True

        for turn in turns:
            pred = dspy.Prediction(**turn.outputs)
            ex = gold if gold is not None else dspy.Example(
                **turn.inputs,
                **{self.history_field: turn.history_snapshot},
                **turn.outputs,
            )

            try:
                if call_with_trace:
                    s = metric(ex, pred, None)
                else:
                    s = metric(ex, pred)
            except TypeError:
                # fallback opposite arity
                try:
                    if call_with_trace:
                        s = metric(ex, pred)
                    else:
                        s = metric(ex, pred, None)
                except Exception as e:
                    if self.on_metric_error == "raise":
                        raise
                    logger.warning("Metric error on turn %s: %s", turn.index, e)
                    s = 0.0
            except Exception as e:
                if self.on_metric_error == "raise":
                    raise
                logger.warning("Metric error on turn %s: %s", turn.index, e)
                s = 0.0

            turn.score = float(s) if s is not None else 0.0
            scores.append(turn.score)

        return scores

    def to_examples(
        self,
        *,
        metric: Callable | None = None,
        min_score: float | None = None,
        include_history: bool = True,
        strict_trajectory: bool = False,
        require_outputs: bool = True,
    ) -> list[dspy.Example]:
        turns = self._active_state().turns
        if metric is not None:
            self.score(metric)

        examples: list[dspy.Example] = []
        for turn in turns:
            if require_outputs and not turn.outputs:
                if strict_trajectory:
                    break
                continue

            if min_score is not None and (turn.score is None or turn.score < min_score):
                if strict_trajectory:
                    break
                continue

            input_dict = dict(turn.inputs)
            if include_history:
                input_dict[self.history_field] = turn.history_snapshot

            ex = dspy.Example(**input_dict, **turn.outputs).with_inputs(*list(input_dict.keys()))
            examples.append(ex)

        return examples

    def to_trainset(self, **kwargs) -> list[dspy.Example]:
        return self.to_examples(**kwargs)

    @staticmethod
    def merge_examples(*sessions: "Session", **kwargs) -> list[dspy.Example]:
        merged: list[dspy.Example] = []
        for s in sessions:
            merged.extend(s.to_examples(**kwargs))
        return merged

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.save_state(), indent=2, default=str))

    def save_state(self) -> dict[str, Any]:
        state = self._active_state()
        return {
            "version": 2,
            "history_field": self.history_field,
            "max_turns": self.max_turns,
            "max_stored_turns": self.max_stored_turns,
            "copy_mode": self.copy_mode,
            "exclude_fields": list(self.exclude_fields),
            "history_input_fields": (
                list(self.history_input_fields) if self.history_input_fields is not None else None
            ),
            "history_policy": self.history_policy,
            "turns": [
                {
                    "index": t.index,
                    "inputs": _safe_serialize(t.inputs),
                    "outputs": _safe_serialize(t.outputs),
                    "score": t.score,
                }
                for t in state.turns
            ],
            "initial_history": _safe_serialize_value(state.initial_history) if state.initial_history is not None else None,
            "l2_memory": state.l2_memory,
            "node_states": {path: node.to_dict() for path, node in state.node_states.items()},
        }

    @classmethod
    def load_from(cls, path: str | Path, module: dspy.Module, **kwargs) -> "Session":
        data = json.loads(Path(path).read_text())

        version = data.get("version", 1)
        if version not in (1, 2):
            raise ValueError(
                f"Unsupported session state version: {version}. "
                "Supported versions: 1, 2."
            )

        initial_history_raw = data.get("initial_history")
        initial_history = None
        if isinstance(initial_history_raw, dict) and "messages" in initial_history_raw:
            initial_history = History(messages=initial_history_raw["messages"])

        # version 1 compatibility: used input_field_override key
        history_input_fields_raw = data.get("history_input_fields", data.get("input_field_override"))

        config = {
            "history_field": data.get("history_field", "history"),
            "max_turns": data.get("max_turns"),
            "max_stored_turns": data.get("max_stored_turns"),
            "exclude_fields": set(data.get("exclude_fields", [])),
            "history_input_fields": (
                set(history_input_fields_raw) if history_input_fields_raw else None
            ),
            "history_policy": data.get("history_policy", "override"),
            "initial_history": initial_history,
            "copy_mode": data.get("copy_mode", "deep"),
        }
        config.update(kwargs)

        session = cls(module, **config)
        session._default_state.l2_memory = str(data.get("l2_memory", ""))

        for t in data.get("turns", []):
            snapshot = session._build_history()
            session._default_state.turns.append(
                Turn(
                    index=t["index"],
                    inputs=t["inputs"],
                    outputs=t["outputs"],
                    history_snapshot=snapshot,
                    score=t.get("score"),
                )
            )

        node_states_raw = data.get("node_states")
        if isinstance(node_states_raw, dict):
            for path, raw in node_states_raw.items():
                if isinstance(raw, dict):
                    session._default_state.node_states[path] = SessionState.from_dict(raw)

        return session

    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Session({type(self.module).__name__}, turns={len(self)}, "
            f"history_field='{self.history_field}')"
        )


# ---------------------------------------------------------------------------
# Factories & projection helper accessors
# ---------------------------------------------------------------------------


def sessionify(module: dspy.Module, **kwargs) -> Session:
    """Factory wrapper for Session."""
    return Session(module, **kwargs)


def with_memory(
    module: dspy.Module,
    *,
    recursive: bool | Literal["predictors"] = True,
    include: set[str] | None = None,
    exclude: set[str] | None = None,
    where: Callable[[str, Any], bool] | None = None,
    isolation: Literal["isolated", "shared"] = "isolated",
    lifespan: Literal["persistent", "episodic", "stateless"] = "persistent",
    consolidator: dspy.Module | None = None,
    child_configs: dict[str, dict[str, Any]] | None = None,
    **kwargs,
) -> Session:
    """Build a shared blueprint session with optional recursive proxy wrapping.

    Defaults are production-friendly:
    - ``copy_mode='none'`` (one shared module instance)
    - ``recursive=True`` (wrap inner predictors as session proxies)
    - ``isolation='isolated'``
    - ``lifespan='persistent'``

    `child_configs` allows per-node policy overrides, e.g.::

        with_memory(
            agent,
            child_configs={
                "query_generator": {"lifespan": "episodic", "consolidator": my_cons},
                "retriever": {"lifespan": "stateless"},
            },
        )
    """
    kwargs.setdefault("copy_mode", "none")
    session = Session(
        module,
        isolation=isolation,
        lifespan=lifespan,
        consolidator=consolidator,
        session_path="root",
        **kwargs,
    )
    session._prepare_recursive_children(
        recursive=recursive,
        include=include,
        exclude=exclude,
        where=where,
        child_configs=child_configs,
    )
    return session


def get_current_history() -> History | None:
    """Projection helper: current node history (L1 view)."""
    return _CURRENT_SESSION_HISTORY.get()


def get_outer_history() -> History | None:
    """Projection helper: outer/root history (if mounted)."""
    return _CURRENT_OUTER_HISTORY.get()


def get_node_memory() -> str:
    """Projection helper: current node semantic memory (L2 string)."""
    return _CURRENT_NODE_MEMORY.get()


def get_child_l1_ledger(path: str) -> str:
    """Projection helper: YAML-like ledger dump for a named child Session.

    ``path`` is the dotted attribute path relative to the root session's wrapped module.
    Example: ``"planner"`` or ``"researcher.writer"``.
    """
    root = _ACTIVE_SESSION_ROOT.get()
    if root is None:
        return ""

    if path in ("", "root", "."):
        target = root
    else:
        target = _resolve_child_session(root, path)
        if target is None:
            return ""

    return _serialize_turns(target._active_state().turns)


def get_execution_trace() -> str:
    """Projection helper: compact execution-tree summary for the active call."""
    root = _ACTIVE_SESSION_ROOT.get()
    if root is None:
        return ""

    sessions: dict[str, Session] = {"root": root}
    sessions.update(root._iter_child_sessions())

    lines: list[str] = []
    for name, session in sessions.items():
        st = session._active_state()
        lines.append(
            f"{name}: turns={len(st.turns)} l2_chars={len(st.l2_memory or '')}"
        )

    stack = _ACTIVE_SESSION_STACK.get()
    if stack:
        id_to_name = {id(sess): name for name, sess in sessions.items()}
        stack_names = [id_to_name.get(id(sess), type(sess.module).__name__) for sess in stack]
        lines.append("active_stack=" + " > ".join(stack_names))

    return "\n".join(lines)


def _resolve_child_session(root: "Session", path: str) -> "Session | None":
    children = root._iter_child_sessions()
    if path in children:
        return children[path]

    # Convenience aliases: allow leading "root." and suffix matching.
    if path.startswith("root."):
        stripped = path[len("root.") :]
        if stripped in children:
            return children[stripped]

    matches = [sess for p, sess in children.items() if p.endswith(path)]
    if len(matches) == 1:
        return matches[0]

    return None


def _serialize_turns(turns: list[Turn]) -> str:
    if not turns:
        return ""

    lines: list[str] = []
    for t in turns:
        lines.append(f"- turn: {t.index}")
        lines.append("  inputs: " + json.dumps(_safe_serialize(t.inputs), ensure_ascii=False))
        lines.append("  outputs: " + json.dumps(_safe_serialize(t.outputs), ensure_ascii=False))
    return "\n".join(lines)


def _extract_updated_memory(pred: Any) -> Any:
    """Best-effort extraction of a consolidator output field."""
    if pred is None:
        return None

    for name in ("updated_memory", "memory", "l2_memory", "output"):
        if hasattr(pred, name):
            try:
                return getattr(pred, name)
            except Exception:
                pass

    if isinstance(pred, dict):
        for name in ("updated_memory", "memory", "l2_memory", "output"):
            if name in pred:
                return pred[name]

    if hasattr(pred, "keys"):
        try:
            keys = list(pred.keys())
            if keys:
                return pred[keys[0]]
        except Exception:
            pass

    return str(pred)


def _is_history_annotation(annotation: Any, *, strict: bool = False) -> bool:
    """Robust history type detection (supports Optional/Union/Annotated)."""
    if annotation is None:
        return False

    # Unwrap Annotated[T, ...]
    origin = get_origin(annotation)
    if origin is not None and str(origin).endswith("Annotated"):
        args = get_args(annotation)
        if args:
            return _is_history_annotation(args[0], strict=strict)

    # Handle Union / Optional
    if origin in (types.UnionType, getattr(__import__("typing"), "Union", object)):
        return any(_is_history_annotation(a, strict=strict) for a in get_args(annotation) if a is not type(None))

    if annotation is History:
        return True

    if strict:
        return False

    if isinstance(annotation, type):
        try:
            return issubclass(annotation, History)
        except Exception:
            return False

    return False


def _safe_serialize(d: dict[str, Any]) -> dict[str, Any]:
    return {k: _safe_serialize_value(v) for k, v in d.items()}


def _safe_serialize_value(v: Any) -> Any:
    if isinstance(v, (str, int, float, bool, type(None))):
        return v
    if hasattr(v, "model_dump"):
        try:
            return v.model_dump()
        except Exception:
            return str(v)
    if hasattr(v, "toDict"):
        try:
            return v.toDict()
        except Exception:
            return str(v)
    if isinstance(v, dict):
        return {k: _safe_serialize_value(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_safe_serialize_value(x) for x in v]
    return str(v)
