"""Session — stateful multi-turn wrapper for any DSPy module.

Key properties:
- Adapter-agnostic (works with Chat/JSON/XML/Template adapters)
- Linearization-first (each turn snapshots history for optimizer-ready examples)
- Program-safe (supports composed dspy.Module programs via predictor-level injection)
- Optimizer-safe (explicit history enables stateless pass-through)
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, get_args, get_origin

import dspy
from dspy.adapters.types.history import History

logger = logging.getLogger(__name__)

# Context variable used by wrapped predictors to obtain the session history
_CURRENT_SESSION_HISTORY: contextvars.ContextVar[History | None] = contextvars.ContextVar(
    "dspy_session_history", default=None
)


@dataclass
class Turn:
    """One session turn."""

    index: int
    inputs: dict[str, Any]
    outputs: dict[str, Any]
    history_snapshot: History
    score: float | None = None


class Session(dspy.Module):
    """Stateful multi-turn wrapper for DSPy modules/programs.

    Supports:
    - Direct predictor wrapping (`dspy.Predict`, `dspy.ChainOfThought`, ...)
    - Program wrapping (`dspy.Module` containing nested predictors)

    For composed programs, session history is injected into nested predictors using
    wrapped predictor forward/aforward methods + a contextvar.
    """

    def __init__(
        self,
        module: dspy.Module,
        *,
        history_field: str = "history",
        max_turns: int | None = None,
        exclude_fields: set[str] | None = None,
        input_field_override: set[str] | None = None,
        history_input_fields: set[str] | None = None,
        initial_history: History | None = None,
        history_policy: Literal["override", "use_if_provided", "replace_session"] = "override",
        on_metric_error: Literal["zero", "raise"] = "zero",
        strict_history_annotation: bool = False,
        lock: Literal["none", "thread", "async"] = "none",
    ):
        super().__init__()

        self.module = copy.deepcopy(module)
        self.history_field = history_field
        self.max_turns = max_turns

        if history_input_fields is not None and input_field_override is not None:
            raise ValueError("Provide either history_input_fields or input_field_override, not both.")
        self.history_input_fields = history_input_fields if history_input_fields is not None else input_field_override

        self.exclude_fields = set(exclude_fields or [])
        self.exclude_fields.add(history_field)

        self._initial_history = initial_history
        self.history_policy = history_policy
        self.on_metric_error = on_metric_error
        self.strict_history_annotation = strict_history_annotation

        self._turns: list[Turn] = []

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

    # ------------------------------------------------------------------
    # Predictor preparation
    # ------------------------------------------------------------------

    def _prepare_predictor_injection(self) -> None:
        """Patch nested predictors to support history and auto-inject from contextvar."""
        for _, predictor in self.module.named_predictors():
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

        orig_forward = predictor.forward

        def wrapped_forward(_self, **kwargs):
            if field_name not in kwargs:
                h = _CURRENT_SESSION_HISTORY.get()
                if h is not None:
                    kwargs[field_name] = h
            return orig_forward(**kwargs)

        predictor.forward = types.MethodType(wrapped_forward, predictor)

        if hasattr(predictor, "aforward"):
            orig_aforward = predictor.aforward

            async def wrapped_aforward(_self, **kwargs):
                if field_name not in kwargs:
                    h = _CURRENT_SESSION_HISTORY.get()
                    if h is not None:
                        kwargs[field_name] = h
                return await orig_aforward(**kwargs)

            predictor.aforward = types.MethodType(wrapped_aforward, predictor)

        predictor._dspy_session_wrapped = True

    def _detect_module_accepts_history(self) -> bool:
        """Whether top-level module.forward can accept history kwarg."""
        forward_method = getattr(self.module, "forward", None)
        if forward_method is None:
            return False

        try:
            sig = inspect.signature(forward_method)
        except (ValueError, TypeError):
            return False

        params = sig.parameters
        if self.history_field in params:
            return True
        return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())

    # ------------------------------------------------------------------
    # Core calling
    # ------------------------------------------------------------------

    def forward(self, **kwargs) -> Any:
        """Run one turn.

        Behavior when caller passes explicit history:
        - override (default): stateless pass-through, do NOT record turn
        - use_if_provided: run with provided history and record turn
        - replace_session: replace session seed with provided history, clear turns, then run statefully
        """
        if self._thread_lock is not None:
            with self._thread_lock:
                return self._forward_impl(**kwargs)
        return self._forward_impl(**kwargs)

    def _forward_impl(self, **kwargs) -> Any:
        explicit_history = kwargs.pop(self.history_field, None)

        if explicit_history is not None and not isinstance(explicit_history, History):
            raise TypeError(
                f"Expected {self.history_field} to be dspy.History, got {type(explicit_history).__name__}."
            )

        # policy handling
        if explicit_history is not None and self.history_policy == "replace_session":
            self._initial_history = explicit_history
            self._turns.clear()
            run_history = self._build_history()
            record_turn = True
        elif explicit_history is not None and self.history_policy == "use_if_provided":
            run_history = explicit_history
            record_turn = True
        elif explicit_history is not None and self.history_policy == "override":
            # optimizer/stateless replay mode
            run_history = explicit_history
            record_turn = False
        else:
            run_history = self._build_history()
            record_turn = True

        result = self._invoke_inner(run_history, kwargs)

        if record_turn:
            self._record_turn(kwargs, result, run_history)

        return result

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
        explicit_history = kwargs.pop(self.history_field, None)

        if explicit_history is not None and not isinstance(explicit_history, History):
            raise TypeError(
                f"Expected {self.history_field} to be dspy.History, got {type(explicit_history).__name__}."
            )

        if explicit_history is not None and self.history_policy == "replace_session":
            self._initial_history = explicit_history
            self._turns.clear()
            run_history = self._build_history()
            record_turn = True
        elif explicit_history is not None and self.history_policy == "use_if_provided":
            run_history = explicit_history
            record_turn = True
        elif explicit_history is not None and self.history_policy == "override":
            run_history = explicit_history
            record_turn = False
        else:
            run_history = self._build_history()
            record_turn = True

        result = await self._ainvoke_inner(run_history, kwargs)

        if record_turn:
            self._record_turn(kwargs, result, run_history)

        return result

    async def acall(self, **kwargs) -> Any:
        return await self.aforward(**kwargs)

    def _invoke_inner(self, history: History, kwargs: dict[str, Any]) -> Any:
        """Invoke wrapped module with contextvar history (and optional top-level kwarg)."""
        call_kwargs = dict(kwargs)
        if self._module_accepts_history:
            call_kwargs[self.history_field] = history

        token = _CURRENT_SESSION_HISTORY.set(history)
        try:
            return self.module(**call_kwargs)
        finally:
            _CURRENT_SESSION_HISTORY.reset(token)

    async def _ainvoke_inner(self, history: History, kwargs: dict[str, Any]) -> Any:
        call_kwargs = dict(kwargs)
        if self._module_accepts_history:
            call_kwargs[self.history_field] = history

        token = _CURRENT_SESSION_HISTORY.set(history)
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
            _CURRENT_SESSION_HISTORY.reset(token)

    def _record_turn(self, kwargs: dict[str, Any], result: Any, history: History) -> None:
        """Record completed turn."""
        inputs_for_record = copy.deepcopy(self._filter_recordable_inputs(kwargs))
        outputs_for_record = self._extract_outputs(result)
        turn = Turn(
            index=len(self._turns),
            inputs=inputs_for_record,
            outputs=outputs_for_record,
            history_snapshot=history,
        )
        self._turns.append(turn)

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
        forward_method = getattr(self.module, "forward", None)
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
        history = self._build_history()
        turn = Turn(
            index=len(self._turns),
            inputs=copy.deepcopy(inputs),
            outputs=copy.deepcopy(outputs),
            history_snapshot=history,
        )
        self._turns.append(turn)
        return turn

    def pop_turn(self) -> Turn | None:
        if self._turns:
            return self._turns.pop()
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
        self.module = copy.deepcopy(module)
        self._predictor_history_fields.clear()
        self._prepare_predictor_injection()
        self._module_accepts_history = self._detect_module_accepts_history()

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def _build_history(self) -> History:
        messages: list[dict[str, Any]] = []

        if self._initial_history is not None:
            messages.extend(self._initial_history.messages)

        for turn in self._turns:
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
        self._turns.clear()

    def fork(self) -> Session:
        new = copy.copy(self)
        new.module = copy.deepcopy(self.module)
        new._turns = copy.deepcopy(self._turns)
        new.exclude_fields = set(self.exclude_fields)
        new.history_input_fields = (
            set(self.history_input_fields) if self.history_input_fields is not None else None
        )
        if self._thread_lock is not None:
            new._thread_lock = threading.Lock()
        if self._async_lock is not None:
            new._async_lock = None
        new._predictor_history_fields = dict(self._predictor_history_fields)
        return new

    @property
    def turns(self) -> list[Turn]:
        return list(self._turns)

    def __len__(self) -> int:
        return len(self._turns)

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

        for turn in self._turns:
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
        if metric is not None:
            self.score(metric)

        examples: list[dspy.Example] = []
        for turn in self._turns:
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
    def merge_examples(*sessions: Session, **kwargs) -> list[dspy.Example]:
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
        return {
            "version": 2,
            "history_field": self.history_field,
            "max_turns": self.max_turns,
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
                for t in self._turns
            ],
            "initial_history": _safe_serialize_value(self._initial_history) if self._initial_history is not None else None,
        }

    @classmethod
    def load_from(cls, path: str | Path, module: dspy.Module, **kwargs) -> Session:
        data = json.loads(Path(path).read_text())

        initial_history_raw = data.get("initial_history")
        initial_history = None
        if isinstance(initial_history_raw, dict) and "messages" in initial_history_raw:
            initial_history = History(messages=initial_history_raw["messages"])

        config = {
            "history_field": data.get("history_field", "history"),
            "max_turns": data.get("max_turns"),
            "exclude_fields": set(data.get("exclude_fields", [])),
            "history_input_fields": (
                set(data.get("history_input_fields", [])) if data.get("history_input_fields") else None
            ),
            "history_policy": data.get("history_policy", "override"),
            "initial_history": initial_history,
        }
        config.update(kwargs)

        session = cls(module, **config)

        for t in data.get("turns", []):
            snapshot = session._build_history()
            session._turns.append(
                Turn(
                    index=t["index"],
                    inputs=t["inputs"],
                    outputs=t["outputs"],
                    history_snapshot=snapshot,
                    score=t.get("score"),
                )
            )

        return session

    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Session({type(self.module).__name__}, turns={len(self._turns)}, "
            f"history_field='{self.history_field}')"
        )


def sessionify(module: dspy.Module, **kwargs) -> Session:
    """Factory wrapper for Session."""
    return Session(module, **kwargs)


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
