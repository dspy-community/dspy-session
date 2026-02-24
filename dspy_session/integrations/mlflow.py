"""MLflow integration for dspy-session.

Provides experiment tracking, turn-level metrics logging,
session artifact storage, and model registry support.

Requires: mlflow >= 2.18 (for mlflow.dspy flavor)

Usage:

    # 1. Log a completed session as an MLflow run
    from dspy_session.integrations.mlflow import log_session

    session = sessionify(module)
    session(question="Hi")
    session(question="Follow up")
    run_id = log_session(session, experiment="my_chatbot")

    # 2. Log session examples as an MLflow dataset
    from dspy_session.integrations.mlflow import log_examples
    log_examples(session, dataset_name="chatbot_v1")

    # 3. Auto-log turns as they happen (requires on_turn hook in Session)
    from dspy_session.integrations.mlflow import mlflow_turn_logger
    session = sessionify(module, on_turn=mlflow_turn_logger())

    # 4. Model registry
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
# Core: log a completed session
# ---------------------------------------------------------------------------


def log_session(
    session,
    *,
    experiment: str | None = None,
    run_name: str | None = None,
    log_model_flag: bool = False,
    artifact_path: str = "session",
    tags: dict[str, str] | None = None,
) -> str:
    """Log a completed session to MLflow as a single run.

    Call this **after** the conversation is done — it snapshots the current
    session state (all accumulated turns, scores, examples).

    Logs:
    - Session config as params
    - Per-turn scores and history lengths as step metrics
    - Aggregate score metrics (mean/min/max)
    - Session state JSON as artifact
    - Turns detail JSON as artifact
    - Linearized examples JSON as artifact
    - Optionally the DSPy module via mlflow.dspy.log_model

    Args:
        session: A dspy_session.Session instance.
        experiment: MLflow experiment name. Uses active experiment if None.
        run_name: Optional run name.
        log_model_flag: If True, also log the inner module via mlflow.dspy.
        artifact_path: Artifact path prefix.
        tags: Optional tags for the run.

    Returns:
        The MLflow run ID.
    """
    mlflow = _check_mlflow()

    if experiment is not None:
        mlflow.set_experiment(experiment)

    with mlflow.start_run(run_name=run_name, tags=tags) as run:
        _log_session_to_active_run(mlflow, session, artifact_path, log_model_flag)

    return run.info.run_id


def _log_session_to_active_run(mlflow, session, artifact_path: str, log_model_flag: bool) -> None:
    """Log session data to the currently active MLflow run."""

    # -- Params --
    mlflow.log_params({
        "session.history_field": session.history_field,
        "session.history_policy": session.history_policy,
        "session.max_turns": str(session.max_turns),
        "session.max_stored_turns": str(session.max_stored_turns),
        "session.copy_mode": session.copy_mode,
        "session.module_type": type(session.module).__name__,
    })
    mlflow.log_metric("total_turns", len(session.turns))

    # -- Per-turn step metrics --
    for turn in session.turns:
        step = turn.index
        if turn.score is not None:
            mlflow.log_metric("turn_score", turn.score, step=step)
        mlflow.log_metric("history_length", len(turn.history_snapshot.messages), step=step)

    # -- Aggregate score metrics --
    scores = [t.score for t in session.turns if t.score is not None]
    if scores:
        mlflow.log_metrics({
            "score_mean": sum(scores) / len(scores),
            "score_min": min(scores),
            "score_max": max(scores),
        })

    # -- Artifacts --
    with tempfile.TemporaryDirectory() as tmpdir:
        # Session state
        state_path = Path(tmpdir) / "session_state.json"
        state_path.write_text(json.dumps(session.save_state(), indent=2, default=str))
        mlflow.log_artifact(str(state_path), artifact_path=artifact_path)

        # Turns detail
        turns_path = Path(tmpdir) / "turns.json"
        turns_data = [
            {
                "index": t.index,
                "inputs": _safe_json(t.inputs),
                "outputs": _safe_json(t.outputs),
                "score": t.score,
                "history_length": len(t.history_snapshot.messages),
            }
            for t in session.turns
        ]
        turns_path.write_text(json.dumps(turns_data, indent=2, default=str))
        mlflow.log_artifact(str(turns_path), artifact_path=artifact_path)

        # Linearized examples
        examples = session.to_examples()
        if examples:
            examples_path = Path(tmpdir) / "examples.json"
            examples_data = [
                {k: _safe_json_value(v) for k, v in ex.items()}
                for ex in examples
            ]
            examples_path.write_text(json.dumps(examples_data, indent=2, default=str))
            mlflow.log_artifact(str(examples_path), artifact_path=artifact_path)

    # -- Model --
    if log_model_flag:
        try:
            mlflow.dspy.log_model(
                dspy_model=session.module,
                artifact_path=f"{artifact_path}/model",
            )
        except Exception as e:
            logger.warning("Failed to log DSPy model via mlflow.dspy: %s", e)


# ---------------------------------------------------------------------------
# Log examples as MLflow dataset
# ---------------------------------------------------------------------------


def log_examples(
    session,
    *,
    dataset_name: str = "session_examples",
    metric=None,
    min_score: float | None = None,
) -> None:
    """Log session examples as an MLflow dataset to the active run.

    Must be called inside an active ``mlflow.start_run()`` context.

    Args:
        session: A dspy_session.Session instance.
        dataset_name: Name for the MLflow dataset.
        metric: Optional metric function for scoring/filtering.
        min_score: Minimum score threshold for filtering.
    """
    mlflow = _check_mlflow()

    examples = session.to_examples(metric=metric, min_score=min_score)
    if not examples:
        logger.warning("No examples to log.")
        return

    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required for log_examples. Install with: pip install pandas")

    rows = [
        {k: _safe_json_value(v) for k, v in ex.items()}
        for ex in examples
    ]
    df = pd.DataFrame(rows)
    dataset = mlflow.data.from_pandas(df, name=dataset_name)
    mlflow.log_input(dataset, context="session_examples")


# ---------------------------------------------------------------------------
# Turn-level auto-logging hook
# ---------------------------------------------------------------------------


def mlflow_turn_logger(
    *,
    experiment: str | None = None,
    run_name: str | None = None,
):
    """Create an on_turn callback that logs each turn to MLflow in real time.

    Returns a callable suitable for Session's ``on_turn`` hook.
    The first turn starts an MLflow run; subsequent turns log to it as steps.

    **Important**: Call ``tracker.end(session)`` when the conversation is done
    to finalize the run and save session state as an artifact.

    Usage::

        from dspy_session.integrations.mlflow import mlflow_turn_logger

        tracker = mlflow_turn_logger(experiment="my_chatbot")
        session = sessionify(dspy.Predict(QA), on_turn=tracker)

        session(question="Hi")        # starts MLflow run, logs step 0
        session(question="Follow up") # logs step 1

        tracker.end(session)           # saves artifacts, closes the run

    Returns:
        A TurnLogger instance (callable).
    """
    mlflow = _check_mlflow()

    class TurnLogger:
        def __init__(self):
            self._run = None
            self._experiment = experiment
            self._run_name = run_name

        def __call__(self, session, turn):
            """Called after each turn is recorded."""
            if self._run is None:
                if self._experiment is not None:
                    mlflow.set_experiment(self._experiment)
                self._run = mlflow.start_run(
                    run_name=self._run_name or f"session_{type(session.module).__name__}",
                )
                mlflow.log_params({
                    "session.history_field": session.history_field,
                    "session.history_policy": session.history_policy,
                    "session.module_type": type(session.module).__name__,
                })

            step = turn.index
            if turn.score is not None:
                mlflow.log_metric("turn_score", turn.score, step=step)
            mlflow.log_metric("history_length", len(turn.history_snapshot.messages), step=step)
            mlflow.log_metric("total_turns", len(session.turns), step=step)

        def end(self, session=None):
            """Finalize the run. Optionally log final session state.

            Args:
                session: If provided, logs session state JSON as an artifact.

            Returns:
                The MLflow run ID, or None if no run was started.
            """
            if self._run is None:
                return None

            if session is not None:
                with tempfile.TemporaryDirectory() as tmpdir:
                    state_path = Path(tmpdir) / "session_state.json"
                    state_path.write_text(
                        json.dumps(session.save_state(), indent=2, default=str)
                    )
                    mlflow.log_artifact(str(state_path))

            run_id = self._run.info.run_id
            mlflow.end_run()
            self._run = None
            return run_id

    return TurnLogger()


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------


def log_model(session, *, artifact_path: str = "session", **kwargs) -> None:
    """Log session's inner module to MLflow model registry.

    Must be called inside an active ``mlflow.start_run()`` context.
    Also saves session state alongside the model.

    Args:
        session: A dspy_session.Session instance.
        artifact_path: Artifact path for the model.
        **kwargs: Extra args passed to mlflow.dspy.log_model.
    """
    mlflow = _check_mlflow()

    mlflow.dspy.log_model(
        dspy_model=session.module,
        artifact_path=f"{artifact_path}/model",
        **kwargs,
    )

    # Also log session state as sibling artifact
    with tempfile.TemporaryDirectory() as tmpdir:
        state_path = Path(tmpdir) / "session_state.json"
        state_path.write_text(
            json.dumps(session.save_state(), indent=2, default=str)
        )
        mlflow.log_artifact(str(state_path), artifact_path=artifact_path)


def load_model(model_uri: str, module=None, **session_kwargs):
    """Load a session from MLflow.

    Loads the DSPy module via mlflow.dspy.load_model, then restores
    session state if a session_state.json artifact is found.

    Args:
        model_uri: MLflow model URI (e.g. "runs:/<run_id>/session/model").
        module: Optional fallback module if mlflow.dspy.load_model fails.
        **session_kwargs: Extra kwargs passed to Session constructor.

    Returns:
        A Session instance with restored state.
    """
    mlflow = _check_mlflow()
    from dspy_session.session import Session

    # Load the DSPy module
    try:
        loaded_module = mlflow.dspy.load_model(model_uri)
    except Exception as e:
        if module is None:
            raise ValueError(
                f"Could not load DSPy model from '{model_uri}' and no fallback module provided."
            ) from e
        logger.warning("mlflow.dspy.load_model failed (%s), using provided module.", e)
        loaded_module = module

    # Try to load session state from sibling artifact
    try:
        parent_uri = model_uri.rsplit("/model", 1)[0]
        local_path = mlflow.artifacts.download_artifacts(
            artifact_uri=f"{parent_uri}/session_state.json"
        )
        return Session.load_from(local_path, module=loaded_module, **session_kwargs)
    except Exception:
        logger.info("No session state found, creating fresh session.")
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
