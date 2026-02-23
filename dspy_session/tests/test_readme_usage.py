"""Smoke tests for README usage patterns (no LM calls)."""

from __future__ import annotations

import dspy
import pytest

from dspy_session import Session, sessionify


class QA(dspy.Signature):
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


def fake_predict(sig=QA):
    p = dspy.Predict(sig)

    def fake_forward(**kwargs):
        return dspy.Prediction(answer=f"ok:{kwargs.get('question', '')}")

    async def fake_aforward(**kwargs):
        return dspy.Prediction(answer=f"ok:{kwargs.get('question', '')}")

    p.forward = fake_forward
    p.__call__ = fake_forward
    p.aforward = fake_aforward
    return p


def test_readme_quickstart_pattern():
    session = sessionify(fake_predict())
    session(question="Q1")
    session(question="Q2")

    assert len(session.turns) == 2
    assert len(session.session_history.messages) == 2


def test_readme_program_wrapping_pattern():
    class Agent(dspy.Module):
        def __init__(self):
            super().__init__()
            self.gen = fake_predict()

        def forward(self, question):
            # no history kwarg
            return self.gen(question=question)

    session = sessionify(Agent())
    session(question="Q1")
    session(question="Q2")
    assert len(session.turns) == 2


def test_readme_history_policy_override_stateless():
    session = sessionify(fake_predict(), history_policy="override")
    session(question="Q1")

    explicit = dspy.History(messages=[{"question": "X", "answer": "Y"}])
    session(question="Q2", history=explicit)

    # explicit-history call is stateless pass-through
    assert len(session.turns) == 1


def test_readme_history_policy_use_if_provided_records():
    session = sessionify(fake_predict(), history_policy="use_if_provided")
    explicit = dspy.History(messages=[{"question": "X", "answer": "Y"}])

    session(question="Q1", history=explicit)
    assert len(session.turns) == 1


def test_readme_history_policy_replace_session():
    session = sessionify(fake_predict(), history_policy="replace_session")
    session(question="Q1")
    assert len(session.turns) == 1

    explicit = dspy.History(messages=[{"question": "seed", "answer": "seed"}])
    session(question="Q2", history=explicit)
    # replace_session clears old turns, then records new turn
    assert len(session.turns) == 1


def test_readme_to_examples_strict_trajectory():
    p = fake_predict()

    # Custom behavior for good/bad labels
    def fake_forward(**kwargs):
        q = kwargs.get("question", "")
        if q == "bad":
            return dspy.Prediction(answer="bad")
        return dspy.Prediction(answer="good")

    p.forward = fake_forward
    p.__call__ = fake_forward

    session = sessionify(p)
    session(question="good1")
    session(question="bad")
    session(question="good2")

    def metric(example, pred, trace=None):
        return 1.0 if pred.answer == "good" else 0.0

    non_strict = session.to_examples(metric=metric, min_score=0.5, strict_trajectory=False)
    strict = session.to_examples(metric=metric, min_score=0.5, strict_trajectory=True)

    assert len(non_strict) == 2
    assert len(strict) == 1


def test_readme_serialization_pattern(tmp_path):
    session = sessionify(fake_predict())
    session(question="Q1")

    path = tmp_path / "session.json"
    session.save(path)

    restored = Session.load_from(path, fake_predict())
    assert len(restored.turns) == 1


def test_readme_update_module_pattern():
    session = sessionify(fake_predict())
    session(question="Q1")

    session.update_module(fake_predict())
    session(question="Q2")
    assert len(session.turns) == 2


@pytest.mark.anyio
async def test_readme_async_pattern():
    session = sessionify(fake_predict(), lock="async")
    out = await session.acall(question="Q1")
    assert out.answer.startswith("ok:")
