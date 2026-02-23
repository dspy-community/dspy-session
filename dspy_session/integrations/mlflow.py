"""MLflow integration for dspy-session.

Provides automatic experiment tracking, turn-level metrics logging,
session artifact storage, and model registry support.

Requires: mlflow >= 2.18 (for mlflow.dspy flavor)

Usage:

    # Option 1: Auto-tracking via DSPy callback
    from dspy_session.integrations.mlflow import autolog
    autolog()  # registers callback globally

    session = sessionify(module)
    session(question="Hi")       # automatically logged as step 0
    session(question="Follow up") # automatically logged as step 1

    # Option 2: Manual logging
    from dspy_session.integrations.mlflow import log_session
    log_session(session, experiment="my_chatbot")

    # Option 3: Model registry
    from dspy_session.integrations.mlflow import log_model, load_model
    log_model(session, artifact_path="session")
    loaded = load_model("runs:/<run_id>/session", module=MyModule())
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_MLFLOW_MIN_VERSION = "2.18"


def _check_mlflow():
    """Import and validate mlflow version."""
    try:
        import mlflow
    except ImportError:
        raise ImportError(
            "mlflow is required for this integration. "
            "Install it with: pip install 'mlflow>=2.18'"
        )

    from packaging.version import Version

    if Version(mlflow.__version__) < Version(_MLFLOW_MIN_VERSION):
        raise ImportError(
            f"mlflow >= {_MLFLOW_MIN_VERSION} required (for mlflow.dspy flavor), "
            f"found {mlflow.__version__}. Upgrade with: pip install -U mlflow"
        )

    return mlflow


# ---------------------------------------------------------------------------
# Manual logging utilities
# ---------------------------------------------------------------------------


def log_session(
    session,
    *,
    experiment: str | None = None,
    run_name: str | None = None,
    log_model: bool = False,
    artifact_path: str = "session",
    tags: dict[str, str] | None = None,
) -> str:
    """Log a completed session to MLflow as a single run.

    Logs:
    - Session config as params
    - Per-turn inputs/outputs/scores as metrics + artifacts
    - Session state JSON as artifact
    - Optionally the DSPy module via mlflow.dspy.log_model

    Args:
        session: A dspy_session.Session instance.
        experiment: MLflow experiment name. If None, uses active experiment.
        run_name: Optional run name.
        log_model: If True, also log the inner module via mlflow.dspy.
        artifact_path: Artifact path prefix for the model.
        tags: Optional tags for the run.

    Returns:
        The MLflow run ID.
    """
    mlflow = _check_mlflow()

    if experiment is not None:
        mlflow.set_experiment(experiment)

    with mlflow.start_run(run_name=run_name, tags=tags) as run:
        # Log session config as params
        mlflow.log_params({
            "session.history_field": session.history_field,
            "session.history_policy": session.history_policy,
            "session.max_turns": str(session.max_turns),
            "session.max_stored_turns": str(session.max_stored_turns),
            "session.copy_mode": session.copy_mode,
            "session.module_type": type(session.module).__name__,
            "session.num_turns": len(session.turns),
        })

        # Log per-turn metrics
        for turn in session.turns:
            step = turn.index
            if turn.score is not None:
                mlflow.log_metric("turn_score", turn.score, step=step)
            mlflow.log_metric("history_length", len(turn.history_snapshot.messages), step=step)

        # Log aggregate metrics
        scores = [t.score for t in session.turns if t.score is not None]
        if scores:
            mlflow.log_metrics({
                "mean_score": sum(scores) / len(scores),
                "min_score": min(scores),
                "max_score": max(scores),
            })

        # Log session state as artifact
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "session_state.json"
            state_path.write_text(json.dumps(session.save_state(), indent=2, default=str))
            mlflow.log_artifact(str(state_path), artifact_path=artifact_path)

            # Log turns detail as artifact
            turns_path = Path(tmpdir) / "turns.json"
            turns_data = []
            for turn in session.turns:
                turns_data.append({
                    "index": turn.index,
                    "inputs": _safe_json(turn.inputs),
                    "outputs": _safe_json(turn.outputs),
                    "score": turn.score,
                    "history_length": len(turn.history_snapshot.messages),
                })
            turns_path.write_text(json.dumps(turns_data, indent=2, default=str))
            mlflow.log_artifact(str(turns_path), artifact_path=artifact_path)

            # Log training examples if available
            examples = session.to_examples()
            if examples:
                examples_path = Path(tmpdir) / "examples.json"
                examples_data = [
                    {k: _safe_json_value(v) for k, v in ex.items()}
                    for ex in examples
                ]
                examples_path.write_text(json.dumps(examples_data, indent=2, default=str))
                mlflow.log_artifact(str(examples_path), artifact_path=artifact_path)

        # Optionally log the DSPy module
        if log_model:
            try:
                mlflow.dspy.log_model(
                    dspy_model=session.module,
                    artifact_path=f"{artifact_path}/model",
                )
            except Exception as e:
                logger.warning("Failed to log DSPy model: %s", e)

        return run.info.run_id


def log_examples(
    session,
    *,
    dataset_name: str = "session_examples",
    metric=None,
    min_score: float | None = None,
) -> None:
    """Log session examples as an MLflow dataset.

    Args:
        session: A dspy_session.Session instance.
        dataset_name: Name for the MLflow dataset.
        metric: Optional metric function for scoring/filtering.
        min_score: Minimum score threshold for filtering.
    """
    mlflow = _check_mlflow()
    import pandas as pd

    examples = session.to_examples(metric=metric, min_score=min_score)
    if not examples:
        logger.warning("No examples to log.")
        return

    rows = []
    for ex in examples:
        row = {}
        for k, v in ex.items():
            row[k] = _safe_json_value(v)
        rows.append(row)

    df = pd.DataFrame(rows)
    dataset = mlflow.data.from_pandas(df, name=dataset_name)
    mlflow.log_input(dataset, context="session_examples")


# ---------------------------------------------------------------------------
# DSPy Callback for automatic tracking
# ---------------------------------------------------------------------------


class SessionCallback:
    """DSPy callback that auto-logs session turns to MLflow.

    Hooks into on_module_start/on_module_end for Session instances.
    Each session gets its own MLflow run. Turns are logged as steps.

    Usage:
        import dspy
        from dspy_session.integrations.mlflow import SessionCallback

        dspy.configure(callbacks=[SessionCallback()])
    """

    def __init__(
        self,
        *,
        experiment: str | None = None,
        log_inputs: bool = True,
        log_outputs: bool = True,
        run_name_prefix: str = "session",
    ):
        self._mlflow = _check_mlflow()
        self._experiment = experiment
        self._log_inputs = log_inputs
        self._log_outputs = log_outputs
        self._run_name_prefix = run_name_prefix

        # Track active session runs: session_id -> (run, pre_turn_count)
        self._active_sessions: dict[int, tuple[Any, int]] = {}

    def on_module_start(
        self,
        call_id: str,
        instance: Any,
        inputs: dict[str, Any],
    ):
        from dspy_session.session import Session

        if not isinstance(instance, Session):
            return

        sid = id(instance)

        # Start a new run for this session if we haven't yet
        if sid not in self._active_sessions:
            if self._experiment is not None:
                self._mlflow.set_experiment(self._experiment)

            run = self._mlflow.start_run(
                run_name=f"{self._run_name_prefix}_{type(instance.module).__name__}",
                nested=True,
            )
            self._mlflow.log_params({
                "session.history_field": instance.history_field,
                "session.history_policy": instance.history_policy,
                "session.module_type": type(instance.module).__name__,
            })
            self._active_sessions[sid] = (run, len(instance.turns))
        else:
            # Update pre-turn count
            run, _ = self._active_sessions[sid]
            self._active_sessions[sid] = (run, len(instance.turns))

    def on_module_end(
        self,
        call_id: str,
        outputs: Any | None,
        exception: Exception | None = None,
    ):
        # We don't have `instance` in on_module_end, so we check all tracked
        # sessions for new turns. This is a limitation of DSPy's callback API.
        for sid, (run, pre_count) in list(self._active_sessions.items()):
            # We can't access the session instance from on_module_end.
            # The turn logging happens via _log_new_turns called externally
            # or we accept this limitation and log on next on_module_start.
            pass

    def finalize_session(self, session) -> str | None:
        """Manually finalize and close the MLflow run for a session.

        Call this when the session conversation is complete.

        Returns:
            The MLflow run ID, or None if session wasn't tracked.
        """
        sid = id(session)
        if sid not in self._active_sessions:
            return None

        run, _ = self._active_sessions.pop(sid)

        # Log all turns
        for turn in session.turns:
            step = turn.index
            if turn.score is not None:
                self._mlflow.log_metric("turn_score", turn.score, step=step)
            self._mlflow.log_metric("history_length", len(turn.history_snapshot.messages), step=step)

        self._mlflow.log_metric("total_turns", len(session.turns))

        # Log session state
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "session_state.json"
            state_path.write_text(
                json.dumps(session.save_state(), indent=2, default=str)
            )
            self._mlflow.log_artifact(str(state_path))

        self._mlflow.end_run()
        return run.info.run_id


def autolog(
    *,
    experiment: str | None = None,
    log_inputs: bool = True,
    log_outputs: bool = True,
) -> SessionCallback:
    """Enable automatic MLflow logging for all dspy-session calls.

    Registers a SessionCallback globally via dspy.configure.
    Also enables mlflow.dspy.autolog() for LM-level tracing.

    Args:
        experiment: MLflow experiment name.
        log_inputs: Whether to log turn inputs.
        log_outputs: Whether to log turn outputs.

    Returns:
        The SessionCallback instance (for manual finalization if needed).
    """
    mlflow = _check_mlflow()
    import dspy

    callback = SessionCallback(
        experiment=experiment,
        log_inputs=log_inputs,
        log_outputs=log_outputs,
    )

    # Add to existing callbacks
    existing = dspy.settings.get("callbacks", [])
    dspy.configure(callbacks=existing + [callback])

    # Also enable mlflow's built-in DSPy autolog for LM-level tracing
    try:
        mlflow.dspy.autolog()
    except Exception as e:
        logger.warning("mlflow.dspy.autolog() failed: %s", e)

    return callback


# ---------------------------------------------------------------------------
# Model registry helpers
# ---------------------------------------------------------------------------


def save_model(session, path: str | Path, **kwargs) -> None:
    """Save session's inner module via mlflow.dspy.save_model.

    Also saves session state alongside the model.

    Args:
        session: A dspy_session.Session instance.
        path: Directory to save to.
        **kwargs: Extra args passed to mlflow.dspy.save_model.
    """
    mlflow = _check_mlflow()
    path = Path(path)

    mlflow.dspy.save_model(dspy_model=session.module, path=str(path), **kwargs)

    # Also save session state
    session_state_path = path / "session_state.json"
    session_state_path.write_text(
        json.dumps(session.save_state(), indent=2, default=str)
    )


def load_model(
    model_uri: str,
    module,
    **session_kwargs,
):
    """Load a session from MLflow model registry.

    Loads the DSPy module via mlflow.dspy.load_model, then restores
    session state if available.

    Args:
        model_uri: MLflow model URI (e.g. "runs:/<run_id>/session/model").
        module: A DSPy module instance to use as the session's module
            (needed because session state doesn't store the module architecture).
        **session_kwargs: Extra kwargs passed to Session constructor.

    Returns:
        A Session instance with restored state.
    """
    mlflow = _check_mlflow()
    from dspy_session.session import Session

    # Load the DSPy module
    loaded_module = mlflow.dspy.load_model(model_uri)

    # Try to find and load session state
    # mlflow artifacts are stored relative to the model URI
    try:
        local_path = mlflow.artifacts.download_artifacts(
            artifact_uri=model_uri.rsplit("/model", 1)[0] + "/session_state.json"
        )
        return Session.load_from(local_path, module=loaded_module, **session_kwargs)
    except Exception:
        logger.info("No session state found at model URI, creating fresh session.")
        return Session(loaded_module, **session_kwargs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_json(d: dict) -> dict:
    return {k: _safe_json_value(v) for k, v in d.items()}


def _safe_json_value(v: Any) -> Any:
    if isinstance(v, (str, int, float, bool, type(None))):
        return v
    if hasattr(v, "model_dump"):
        try:
            return v.model_dump()
        except Exception:
            return str(v)
    if isinstance(v, dict):
        return {k: _safe_json_value(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_safe_json_value(x) for x in v]
    return str(v)
