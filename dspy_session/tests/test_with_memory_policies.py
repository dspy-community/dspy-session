"""Tests for with_memory() policy axes: topology, lifespan, consolidation."""

from __future__ import annotations

import dspy
from dspy.adapters.types.history import History

from dspy_session import Session, with_memory


class QASig(dspy.Signature):
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


def _worker_with_history_trace(trace: list[int]):
    worker = dspy.Predict(QASig)

    def fake_forward(**kwargs):
        h = kwargs.get("history")
        trace.append(len(h.messages) if isinstance(h, History) else -1)
        return dspy.Prediction(answer=f"ok:{kwargs.get('question', '')}")

    worker.forward = fake_forward
    worker.__call__ = fake_forward
    return worker


def _simple_consolidator():
    sig = dspy.Signature("past_memory, episode_transcript -> updated_memory")
    cons = dspy.Predict(sig)

    def fake_forward(**kwargs):
        prior = kwargs.get("past_memory", "")
        episode = kwargs.get("episode_transcript", "")
        turns = episode.count("- turn:")
        joined = f"{prior}|turns={turns}" if prior else f"turns={turns}"
        return dspy.Prediction(updated_memory=joined)

    cons.forward = fake_forward
    cons.__call__ = fake_forward
    return cons


def test_with_memory_recursive_wraps_predictors_and_routes_state_tree():
    trace: list[int] = []

    class Agent(dspy.Module):
        def __init__(self):
            super().__init__()
            self.worker = _worker_with_history_trace(trace)

        def forward(self, question):
            return self.worker(question=question)

    app = with_memory(Agent(), recursive=True)
    assert isinstance(app.module.worker, Session)

    alice = app.new_state()
    bob = app.new_state()

    with app.use_state(alice):
        app(question="A1")
        app(question="A2")

    with app.use_state(bob):
        app(question="B1")

    assert len(alice.turns) == 2
    assert len(bob.turns) == 1

    assert "worker" in alice.node_states
    assert "worker" in bob.node_states
    assert len(alice.node_states["worker"].turns) == 2
    assert len(bob.node_states["worker"].turns) == 1


def test_lifespan_stateless_child_records_no_turns():
    trace: list[int] = []

    class Agent(dspy.Module):
        def __init__(self):
            super().__init__()
            self.worker = _worker_with_history_trace(trace)

        def forward(self, question):
            return self.worker(question=question)

    app = with_memory(
        Agent(),
        recursive=True,
        child_configs={"worker": {"lifespan": "stateless"}},
    )

    state = app.new_state()
    with app.use_state(state):
        app(question="Q1")
        app(question="Q2")

    assert len(state.turns) == 2  # root still persistent by default
    assert "worker" in state.node_states
    assert len(state.node_states["worker"].turns) == 0


def test_lifespan_episodic_child_clears_after_macro_turn_and_consolidates():
    trace: list[int] = []

    class Agent(dspy.Module):
        def __init__(self):
            super().__init__()
            self.worker = _worker_with_history_trace(trace)

        def forward(self, question):
            first = self.worker(question=question).answer
            second = self.worker(question=f"{question} (retry)").answer
            return dspy.Prediction(answer=second, first=first)

    app = with_memory(
        Agent(),
        recursive=True,
        child_configs={
            "worker": {
                "lifespan": "episodic",
                "consolidator": _simple_consolidator(),
            }
        },
    )

    state = app.new_state()
    with app.use_state(state):
        app(question="Q1")

    # Within-turn accumulation happened (second call saw one prior worker turn)
    assert trace == [0, 1]

    worker_state = state.node_states["worker"]
    # Episodic ledger should be wiped after macro turn finalization
    assert len(worker_state.turns) == 0
    # Consolidator should have written L2 memory
    assert "turns=2" in worker_state.l2_memory


def test_isolation_shared_uses_root_history_and_keeps_child_unmutated():
    trace: list[int] = []

    class Agent(dspy.Module):
        def __init__(self):
            super().__init__()
            self.worker = _worker_with_history_trace(trace)

        def forward(self, question):
            reply = self.worker(question=question).answer
            return dspy.Prediction(answer=reply)

    app = with_memory(
        Agent(),
        recursive=True,
        child_configs={"worker": {"isolation": "shared"}},
    )

    state = app.new_state()
    with app.use_state(state):
        app(question="Q1")
        app(question="Q2")

    # Child saw root history depth (0 then 1)
    assert trace == [0, 1]

    # Shared isolation: child does not maintain its own turn ledger
    assert "worker" in state.node_states
    assert len(state.node_states["worker"].turns) == 0


def test_exclude_keeps_branch_unwrapped_and_stateless():
    trace: list[int] = []

    class Agent(dspy.Module):
        def __init__(self):
            super().__init__()
            self.worker = _worker_with_history_trace(trace)

        def forward(self, question):
            return self.worker(question=question)

    app = with_memory(Agent(), recursive=True, exclude={"worker"})
    assert not isinstance(app.module.worker, Session)

    state = app.new_state()
    with app.use_state(state):
        app(question="Q1")

    assert "worker" not in state.node_states


def test_pre_sessionified_child_is_adopted_and_can_be_overridden():
    trace: list[int] = []
    child = with_memory(_worker_with_history_trace(trace), recursive=False)

    class Agent(dspy.Module):
        def __init__(self):
            super().__init__()
            self.worker = child

        def forward(self, question):
            return self.worker(question=question)

    app = with_memory(
        Agent(),
        recursive=True,
        child_configs={"worker": {"lifespan": "stateless"}},
    )

    state = app.new_state()
    with app.use_state(state):
        app(question="Q1")

    assert "worker" in state.node_states
    assert len(state.node_states["worker"].turns) == 0
