"""Tests for external state binding (production serving mode)."""

from __future__ import annotations

import asyncio

import dspy
import pytest
from dspy.adapters.types.history import History

from dspy_session import (
    SessionState,
    get_child_l1_ledger,
    get_execution_trace,
    get_outer_history,
    sessionify,
    with_memory,
)


class QASig(dspy.Signature):
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


def _fake_predict():
    p = dspy.Predict(QASig)

    def fake_forward(**kwargs):
        return dspy.Prediction(answer=f"ok:{kwargs.get('question', '')}")

    async def fake_aforward(**kwargs):
        return dspy.Prediction(answer=f"ok:{kwargs.get('question', '')}")

    p.forward = fake_forward
    p.__call__ = fake_forward
    p.aforward = fake_aforward
    return p


def test_external_state_binding_routes_turns_to_bound_state():
    agent = with_memory(_fake_predict())

    alice = agent.new_state()
    bob = agent.new_state()

    with agent.use_state(alice):
        agent(question="A1")
        agent(question="A2")

    with agent.use_state(bob):
        agent(question="B1")

    assert len(alice.turns) == 2
    assert len(bob.turns) == 1
    assert [t.inputs["question"] for t in alice.turns] == ["A1", "A2"]
    assert [t.inputs["question"] for t in bob.turns] == ["B1"]

    # Default notebook state remains untouched when always using external state.
    assert len(agent.turns) == 0


@pytest.mark.anyio
async def test_external_state_binding_isolated_across_concurrent_tasks():
    agent = with_memory(_fake_predict())

    alice = agent.new_state()
    bob = agent.new_state()

    async def run(state: SessionState, prefix: str):
        async with agent.use_state(state):
            await agent.acall(question=f"{prefix}1")
            # Force a scheduling boundary to exercise context isolation.
            await asyncio.sleep(0)
            await agent.acall(question=f"{prefix}2")

    await asyncio.gather(
        run(alice, "A"),
        run(bob, "B"),
    )

    assert [t.inputs["question"] for t in alice.turns] == ["A1", "A2"]
    assert [t.inputs["question"] for t in bob.turns] == ["B1", "B2"]


def test_session_state_roundtrip_dict():
    from dspy_session import Turn

    state = SessionState(
        turns=[
            Turn(
                index=0,
                inputs={"question": "Q1"},
                outputs={"answer": "A1"},
                history_snapshot=History(messages=[]),
            )
        ],
        initial_history=History(messages=[{"question": "seed", "answer": "ctx"}]),
        l2_memory="known fact",
        node_states={
            "worker": SessionState(
                turns=[
                    Turn(
                        index=0,
                        inputs={"question": "W1"},
                        outputs={"answer": "WA1"},
                        history_snapshot=History(messages=[]),
                    )
                ],
                l2_memory="worker fact",
            )
        },
    )
    d = state.to_dict()
    restored = SessionState.from_dict(d)

    assert len(restored.turns) == 1
    assert restored.turns[0].inputs["question"] == "Q1"
    assert restored.initial_history is not None
    assert restored.initial_history.messages[0]["question"] == "seed"
    assert restored.l2_memory == "known fact"
    assert "worker" in restored.node_states
    assert restored.node_states["worker"].turns[0].inputs["question"] == "W1"
    assert restored.node_states["worker"].l2_memory == "worker fact"


def test_outer_history_projection_helper_is_set_during_call():
    p = dspy.Predict(QASig)

    def fake_forward(**kwargs):
        # Inside model invocation context, outer history should be mounted.
        outer = get_outer_history()
        assert outer is not None
        assert isinstance(outer, History)
        return dspy.Prediction(answer="ok")

    p.forward = fake_forward
    p.__call__ = fake_forward

    s = sessionify(p)
    s(question="Q1")


def test_child_ledger_and_execution_trace_helpers():
    class Parent(dspy.Module):
        def __init__(self):
            super().__init__()
            self.worker = sessionify(_fake_predict(), copy_mode="none")

        def forward(self, question):
            before = get_child_l1_ledger("worker")
            out = self.worker(question=question).answer
            after = get_child_l1_ledger("worker")
            trace = get_execution_trace()
            return dspy.Prediction(answer=out, before=before, after=after, trace=trace)

    root = sessionify(Parent(), copy_mode="none")
    out = root(question="Q1")

    assert out.before == ""
    assert "Q1" in out.after
    assert "worker" in out.trace
    assert "root" in out.trace


def test_execution_trace_empty_outside_active_call():
    assert get_execution_trace() == ""
    assert get_child_l1_ledger("anything") == ""
