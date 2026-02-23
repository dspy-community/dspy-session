"""Tests for dspy-session — no LM calls, only Session logic."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import dspy
from dspy.adapters.types.history import History

import pytest
from dspy_session import Session, sessionify
from dspy_session.session import Turn, _safe_serialize, _safe_serialize_value


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class QASig(dspy.Signature):
    """Answer questions."""
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


class QAWithHistory(dspy.Signature):
    """Answer questions with history."""
    question: str = dspy.InputField()
    history: dspy.History = dspy.InputField()
    answer: str = dspy.OutputField()


class MultiOutputSig(dspy.Signature):
    """Classify and explain."""
    text: str = dspy.InputField()
    category: str = dspy.OutputField()
    reasoning: str = dspy.OutputField()


class RAGSig(dspy.Signature):
    """Answer with context."""
    question: str = dspy.InputField()
    context: str = dspy.InputField()
    answer: str = dspy.OutputField()


def _fake_predict_factory(sig_class, responses: list[dict]):
    """Create a Predict-like mock that returns canned responses in order."""
    predict = dspy.Predict(sig_class)
    call_idx = [0]

    def fake_forward(**kwargs):
        idx = call_idx[0] % len(responses)
        call_idx[0] += 1
        return dspy.Prediction(**responses[idx])

    predict.forward = fake_forward
    predict.__call__ = fake_forward
    return predict


# ---------------------------------------------------------------------------
# Signature patching
# ---------------------------------------------------------------------------


class TestSignaturePatching:
    def test_adds_history_field_to_signature_without_one(self):
        predict = dspy.Predict(QASig)
        session = Session(predict)

        sig = session.module.signature
        assert "history" in sig.input_fields
        assert sig.input_fields["history"].annotation is History

    def test_preserves_existing_history_field(self):
        predict = dspy.Predict(QAWithHistory)
        session = Session(predict)

        sig = session.module.signature
        assert "history" in sig.input_fields
        assert sig.input_fields["history"].annotation is History

    def test_does_not_mutate_original_module(self):
        predict = dspy.Predict(QASig)
        session = Session(predict)

        assert "history" not in predict.signature.input_fields
        assert "history" in session.module.signature.input_fields

    def test_custom_history_field_name(self):
        predict = dspy.Predict(QASig)
        session = Session(predict, history_field="chat_history")

        sig = session.module.signature
        assert "chat_history" in sig.input_fields
        assert sig.input_fields["chat_history"].annotation is History

    def test_detects_existing_history_field_with_different_name(self):
        """If the signature has a History-typed field named differently, use that."""
        sig = dspy.Signature("question -> answer")
        sig = sig.append("conv_history", dspy.InputField(desc="history"), type_=History)
        predict = dspy.Predict(sig)
        session = Session(predict)

        assert session.history_field == "conv_history"

    def test_warns_when_existing_history_name_differs_from_requested(self):
        """If user asks for 'chat_log' but signature has 'history', warn and use existing."""
        predict = dspy.Predict(QAWithHistory)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            session = Session(predict, history_field="chat_log")
            # Should warn about the mismatch
            history_warnings = [x for x in w if "history" in str(x.message).lower()]
            assert len(history_warnings) >= 1
        # Should use the existing name
        assert session.history_field == "history"


# ---------------------------------------------------------------------------
# Turn recording
# ---------------------------------------------------------------------------


class TestTurnRecording:
    def test_records_turns(self):
        responses = [{"answer": "Hello!"}, {"answer": "I'm fine"}]
        predict = _fake_predict_factory(QASig, responses)
        session = Session(predict)

        session(question="Hi")
        session(question="How are you?")

        assert len(session) == 2
        assert session.turns[0].inputs == {"question": "Hi"}
        assert session.turns[0].outputs == {"answer": "Hello!"}
        assert session.turns[1].inputs == {"question": "How are you?"}
        assert session.turns[1].outputs == {"answer": "I'm fine"}

    def test_history_snapshot_grows(self):
        responses = [{"answer": "A"}, {"answer": "B"}, {"answer": "C"}]
        predict = _fake_predict_factory(QASig, responses)
        session = Session(predict)

        session(question="Q1")
        session(question="Q2")
        session(question="Q3")

        # Turn 0: empty history
        assert session.turns[0].history_snapshot.messages == []

        # Turn 1: sees turn 0
        assert len(session.turns[1].history_snapshot.messages) == 1
        assert session.turns[1].history_snapshot.messages[0] == {
            "question": "Q1", "answer": "A"
        }

        # Turn 2: sees turns 0 and 1
        assert len(session.turns[2].history_snapshot.messages) == 2

    def test_multiple_output_fields(self):
        responses = [{"category": "tech", "reasoning": "It's about code"}]
        predict = _fake_predict_factory(MultiOutputSig, responses)
        session = Session(predict)

        session(text="Python is great")

        assert session.turns[0].outputs == {
            "category": "tech",
            "reasoning": "It's about code",
        }

    def test_exclude_fields(self):
        responses = [{"answer": "Paris"}]
        predict = _fake_predict_factory(RAGSig, responses)
        session = Session(predict, exclude_fields={"context"})

        session(question="Capital of France?", context="France is a country in Europe...")

        history = session.session_history
        assert len(history.messages) == 1
        assert "context" not in history.messages[0]
        assert "question" in history.messages[0]
        assert "answer" in history.messages[0]

    def test_forward_alias(self):
        """session.forward() should work the same as session()."""
        responses = [{"answer": "A"}]
        predict = _fake_predict_factory(QASig, responses)
        session = Session(predict)

        result = session.forward(question="Q1")
        assert len(session) == 1
        assert session.turns[0].inputs["question"] == "Q1"


# ---------------------------------------------------------------------------
# input_field_override
# ---------------------------------------------------------------------------


class TestInputFieldOverride:
    def test_only_specified_fields_in_history(self):
        """input_field_override limits which input fields appear in history."""
        responses = [{"answer": "A"}, {"answer": "B"}]
        predict = _fake_predict_factory(RAGSig, responses)
        session = Session(predict, input_field_override={"question"})

        session(question="Q1", context="big context 1")
        session(question="Q2", context="big context 2")

        # History should only have 'question' from inputs (not 'context')
        msg = session.turns[1].history_snapshot.messages[0]
        assert "question" in msg
        assert "context" not in msg
        # But outputs are always included
        assert "answer" in msg

    def test_input_field_override_none_includes_all(self):
        """When input_field_override is None, all input fields are included."""
        responses = [{"answer": "A"}, {"answer": "B"}]
        predict = _fake_predict_factory(RAGSig, responses)
        session = Session(predict, input_field_override=None)

        session(question="Q1", context="ctx1")
        session(question="Q2", context="ctx2")

        msg = session.turns[1].history_snapshot.messages[0]
        assert "question" in msg
        assert "context" in msg

    def test_input_field_override_with_exclude_fields(self):
        """input_field_override and exclude_fields interact correctly."""
        responses = [{"answer": "A"}, {"answer": "B"}]
        predict = _fake_predict_factory(RAGSig, responses)
        session = Session(
            predict,
            input_field_override={"question", "context"},
            exclude_fields={"context"},
        )

        session(question="Q1", context="ctx1")
        session(question="Q2", context="ctx2")

        # context is in input_field_override but also in exclude_fields
        # exclude_fields should win
        msg = session.turns[1].history_snapshot.messages[0]
        assert "question" in msg
        assert "context" not in msg


# ---------------------------------------------------------------------------
# Max turns (sliding window)
# ---------------------------------------------------------------------------


class TestMaxTurns:
    def test_sliding_window(self):
        responses = [{"answer": str(i)} for i in range(5)]
        predict = _fake_predict_factory(QASig, responses)
        session = Session(predict, max_turns=2)

        for i in range(5):
            session(question=f"Q{i}")

        # All 5 turns are recorded
        assert len(session) == 5

        # But current history only has last 2
        history = session.session_history
        assert len(history.messages) == 2
        assert history.messages[0]["question"] == "Q3"
        assert history.messages[1]["question"] == "Q4"

    def test_early_turn_snapshot_preserved_after_many_turns(self):
        """Turn 0's snapshot should remain empty regardless of how many turns follow."""
        responses = [{"answer": str(i)} for i in range(10)]
        predict = _fake_predict_factory(QASig, responses)
        session = Session(predict, max_turns=3)

        for i in range(10):
            session(question=f"Q{i}")

        # Turn 0 was captured when history was empty — should still be empty
        assert session.turns[0].history_snapshot.messages == []
        # Turn 1 should still only see turn 0
        assert len(session.turns[1].history_snapshot.messages) == 1


# ---------------------------------------------------------------------------
# Initial history
# ---------------------------------------------------------------------------


class TestInitialHistory:
    def test_initial_history_appears_in_first_turn(self):
        prior = History(messages=[{"question": "Hi", "answer": "Hello!"}])
        responses = [{"answer": "I'm an AI"}]
        predict = _fake_predict_factory(QASig, responses)
        session = Session(predict, initial_history=prior)

        session(question="What are you?")

        # First turn should see the initial history
        snapshot = session.turns[0].history_snapshot
        assert len(snapshot.messages) == 1
        assert snapshot.messages[0] == {"question": "Hi", "answer": "Hello!"}

    def test_initial_history_grows_with_turns(self):
        prior = History(messages=[{"question": "Hi", "answer": "Hello!"}])
        responses = [{"answer": "A"}, {"answer": "B"}]
        predict = _fake_predict_factory(QASig, responses)
        session = Session(predict, initial_history=prior)

        session(question="Q1")
        session(question="Q2")

        # Turn 1 should see initial + turn 0
        snapshot = session.turns[1].history_snapshot
        assert len(snapshot.messages) == 2
        assert snapshot.messages[0] == {"question": "Hi", "answer": "Hello!"}
        assert snapshot.messages[1]["question"] == "Q1"

    def test_reset_preserves_initial_history(self):
        prior = History(messages=[{"question": "Hi", "answer": "Hello!"}])
        responses = [{"answer": "A"}, {"answer": "B"}]
        predict = _fake_predict_factory(QASig, responses)
        session = Session(predict, initial_history=prior)

        session(question="Q1")
        session.reset()

        # After reset, history should still have initial_history
        assert len(session.session_history.messages) == 1
        assert session.session_history.messages[0] == {"question": "Hi", "answer": "Hello!"}


# ---------------------------------------------------------------------------
# add_turn / pop_turn
# ---------------------------------------------------------------------------


class TestManualTurns:
    def test_add_turn(self):
        predict = _fake_predict_factory(QASig, [{"answer": "A"}])
        session = Session(predict)

        turn = session.add_turn(
            inputs={"question": "manually added"},
            outputs={"answer": "hand-written"},
        )

        assert len(session) == 1
        assert turn.inputs["question"] == "manually added"
        assert turn.outputs["answer"] == "hand-written"
        assert turn.history_snapshot.messages == []

    def test_add_turn_then_call(self):
        """Manual turn should appear in history for subsequent calls."""
        responses = [{"answer": "B"}]
        predict = _fake_predict_factory(QASig, responses)
        session = Session(predict)

        session.add_turn(
            inputs={"question": "Q1"},
            outputs={"answer": "A"},
        )
        session(question="Q2")

        # Turn 1 (the call) should see the manual turn in history
        snapshot = session.turns[1].history_snapshot
        assert len(snapshot.messages) == 1
        assert snapshot.messages[0] == {"question": "Q1", "answer": "A"}

    def test_pop_turn(self):
        responses = [{"answer": "A"}, {"answer": "B"}]
        predict = _fake_predict_factory(QASig, responses)
        session = Session(predict)

        session(question="Q1")
        session(question="Q2")
        assert len(session) == 2

        popped = session.pop_turn()
        assert popped is not None
        assert popped.inputs["question"] == "Q2"
        assert len(session) == 1

    def test_pop_turn_empty(self):
        predict = dspy.Predict(QASig)
        session = Session(predict)
        assert session.pop_turn() is None


# ---------------------------------------------------------------------------
# Reset & Fork
# ---------------------------------------------------------------------------


class TestResetFork:
    def test_reset_clears_turns(self):
        responses = [{"answer": "A"}]
        predict = _fake_predict_factory(QASig, responses)
        session = Session(predict)

        session(question="Hi")
        assert len(session) == 1

        session.reset()
        assert len(session) == 0
        assert session.session_history.messages == []

    def test_fork_creates_independent_copy(self):
        responses = [{"answer": "A"}, {"answer": "B"}, {"answer": "C"}]
        predict = _fake_predict_factory(QASig, responses)
        session = Session(predict)

        session(question="Q1")
        forked = session.fork()

        # Add more to original
        session(question="Q2")

        # Forked should still have only 1 turn
        assert len(session) == 2
        assert len(forked) == 1

    def test_fork_preserves_history(self):
        responses = [{"answer": "A"}, {"answer": "B"}]
        predict = _fake_predict_factory(QASig, responses)
        session = Session(predict)

        session(question="Q1")
        session(question="Q2")
        forked = session.fork()

        assert len(forked.turns) == 2
        assert forked.turns[0].inputs["question"] == "Q1"
        assert forked.turns[1].inputs["question"] == "Q2"

    def test_fork_deep_copies_module(self):
        """Fork should deep-copy the module so changes don't leak."""
        predict = dspy.Predict(QASig)
        session = Session(predict)
        forked = session.fork()

        # The modules should be different objects
        assert session.module is not forked.module


# ---------------------------------------------------------------------------
# Linearization — to_examples()
# ---------------------------------------------------------------------------


class TestLinearization:
    def test_to_examples_basic(self):
        responses = [{"answer": "A"}, {"answer": "B"}]
        predict = _fake_predict_factory(QASig, responses)
        session = Session(predict)

        session(question="Q1")
        session(question="Q2")

        examples = session.to_examples()
        assert len(examples) == 2

        # First example: empty history
        ex0 = examples[0]
        assert ex0.question == "Q1"
        assert ex0.answer == "A"
        assert ex0.history.messages == []
        assert set(ex0.inputs().keys()) == {"question", "history"}

        # Second example: history includes turn 0
        ex1 = examples[1]
        assert ex1.question == "Q2"
        assert ex1.answer == "B"
        assert len(ex1.history.messages) == 1
        assert ex1.history.messages[0] == {"question": "Q1", "answer": "A"}

    def test_to_examples_without_history(self):
        responses = [{"answer": "A"}]
        predict = _fake_predict_factory(QASig, responses)
        session = Session(predict)

        session(question="Q1")

        examples = session.to_examples(include_history=False)
        assert len(examples) == 1
        assert "history" not in examples[0].inputs().keys()

    def test_to_examples_with_metric_filtering(self):
        responses = [{"answer": "good"}, {"answer": "bad"}, {"answer": "good"}]
        predict = _fake_predict_factory(QASig, responses)
        session = Session(predict)

        session(question="Q1")
        session(question="Q2")
        session(question="Q3")

        def metric(example, pred, trace=None):
            return 1.0 if pred.answer == "good" else 0.0

        examples = session.to_examples(metric=metric, min_score=0.5)
        assert len(examples) == 2
        assert examples[0].question == "Q1"
        assert examples[1].question == "Q3"

    def test_to_examples_with_inputs_set(self):
        responses = [{"answer": "A"}]
        predict = _fake_predict_factory(QASig, responses)
        session = Session(predict)

        session(question="Q1")

        examples = session.to_examples()
        ex = examples[0]

        inputs = ex.inputs()
        labels = ex.labels()
        assert "question" in inputs.keys()
        assert "history" in inputs.keys()
        assert "answer" in labels.keys()
        assert "answer" not in inputs.keys()

    def test_to_trainset_alias(self):
        responses = [{"answer": "A"}]
        predict = _fake_predict_factory(QASig, responses)
        session = Session(predict)
        session(question="Q1")

        trainset = session.to_trainset()
        examples = session.to_examples()
        assert len(trainset) == len(examples)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


class TestScoring:
    def test_score_all_turns(self):
        responses = [{"answer": "yes"}, {"answer": "no"}, {"answer": "yes"}]
        predict = _fake_predict_factory(QASig, responses)
        session = Session(predict)

        for i in range(3):
            session(question=f"Q{i}")

        def metric(example, pred, trace=None):
            return 1.0 if pred.answer == "yes" else 0.0

        scores = session.score(metric)
        assert scores == [1.0, 0.0, 1.0]

        assert session.turns[0].score == 1.0
        assert session.turns[1].score == 0.0
        assert session.turns[2].score == 1.0

    def test_score_self_evaluation_includes_outputs(self):
        """In self-evaluation mode, the example should contain outputs too."""
        responses = [{"answer": "Paris"}]
        predict = _fake_predict_factory(QASig, responses)
        session = Session(predict)
        session(question="Capital of France?")

        def metric_that_checks_labels(example, pred, trace=None):
            # Verify the example has the output field available
            return 1.0 if hasattr(example, "answer") else 0.0

        scores = session.score(metric_that_checks_labels)
        assert scores == [1.0]

    def test_score_logs_metric_errors(self, caplog):
        """Metric errors should be logged, not silently swallowed."""
        responses = [{"answer": "A"}]
        predict = _fake_predict_factory(QASig, responses)
        session = Session(predict)
        session(question="Q1")

        def bad_metric(example, pred, trace=None):
            raise ValueError("intentional error")

        import logging
        with caplog.at_level(logging.WARNING):
            scores = session.score(bad_metric)

        assert scores == [0.0]
        assert any("intentional error" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Merge examples from multiple sessions
# ---------------------------------------------------------------------------


class TestMerge:
    def test_merge_sessions(self):
        responses1 = [{"answer": "A"}]
        responses2 = [{"answer": "B"}, {"answer": "C"}]

        p1 = _fake_predict_factory(QASig, responses1)
        p2 = _fake_predict_factory(QASig, responses2)

        s1 = Session(p1)
        s2 = Session(p2)

        s1(question="Q1")
        s2(question="Q2")
        s2(question="Q3")

        examples = Session.merge_examples(s1, s2)
        assert len(examples) == 3


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_save_and_load_from(self, tmp_path):
        responses = [{"answer": "A"}, {"answer": "B"}]
        predict = _fake_predict_factory(QASig, responses)
        session = Session(predict)

        session(question="Q1")
        session(question="Q2")

        path = tmp_path / "session.json"
        session.save(path)

        # Verify JSON is valid
        data = json.loads(path.read_text())
        assert data["version"] == 2
        assert len(data["turns"]) == 2

        # Load into a new session via classmethod
        predict2 = _fake_predict_factory(QASig, [{"answer": "X"}])
        session2 = Session.load_from(path, predict2)

        assert len(session2) == 2
        assert session2.turns[0].inputs["question"] == "Q1"
        assert session2.turns[1].inputs["question"] == "Q2"

    def test_load_from_preserves_history_snapshots(self, tmp_path):
        responses = [{"answer": "A"}, {"answer": "B"}]
        predict = _fake_predict_factory(QASig, responses)
        session = Session(predict)

        session(question="Q1")
        session(question="Q2")

        path = tmp_path / "session.json"
        session.save(path)

        predict2 = _fake_predict_factory(QASig, [{"answer": "X"}])
        session2 = Session.load_from(path, predict2)

        # Turn 0's snapshot should be empty
        assert session2.turns[0].history_snapshot.messages == []
        # Turn 1's snapshot should have turn 0
        assert len(session2.turns[1].history_snapshot.messages) == 1
        assert session2.turns[1].history_snapshot.messages[0]["question"] == "Q1"

    def test_load_from_respects_exclude_fields(self, tmp_path):
        """Loaded history snapshots should respect exclude_fields from the saved config."""
        responses = [{"answer": "A"}, {"answer": "B"}]
        predict = _fake_predict_factory(RAGSig, responses)
        session = Session(predict, exclude_fields={"context"})

        session(question="Q1", context="big ctx")
        session(question="Q2", context="big ctx 2")

        path = tmp_path / "session.json"
        session.save(path)

        predict2 = _fake_predict_factory(RAGSig, [{"answer": "X"}])
        session2 = Session.load_from(path, predict2)

        # History snapshot should not contain 'context'
        msg = session2.turns[1].history_snapshot.messages[0]
        assert "context" not in msg
        assert "question" in msg

    def test_save_state_dict(self):
        responses = [{"answer": "A"}]
        predict = _fake_predict_factory(QASig, responses)
        session = Session(predict)
        session(question="Q1")

        state = session.save_state()
        assert state["version"] == 2
        assert len(state["turns"]) == 1
        # Should be JSON-serializable
        json.dumps(state)


# ---------------------------------------------------------------------------
# Program-level wrapping
# ---------------------------------------------------------------------------


class TestProgramWrapping:
    def test_program_with_history_kwarg(self):
        """Program whose forward() accepts history."""

        class MyProgram(dspy.Module):
            def __init__(self):
                super().__init__()
                self._last_history = None

            def forward(self, question, history=None):
                self._last_history = history
                return dspy.Prediction(answer=f"answer to {question}")

        prog = MyProgram()
        session = Session(prog)

        session(question="Q1")
        session(question="Q2")

        assert len(session) == 2
        # The program should have received history on the second call
        assert session.module._last_history is not None

    def test_program_with_kwargs(self):
        """Program whose forward() accepts **kwargs."""

        class FlexProgram(dspy.Module):
            def forward(self, question, **kwargs):
                return dspy.Prediction(answer="ok")

        prog = FlexProgram()
        session = Session(prog)
        session(question="Q1")
        assert len(session) == 1

    def test_program_without_history_still_works_via_predictor_injection(self):
        """Program forward() can omit history; nested predictors still receive it."""

        class StrictProgram(dspy.Module):
            def __init__(self):
                super().__init__()
                self.gen = dspy.Predict(QASig)

                def fake_forward(**kwargs):
                    # confirm injected history arrives even though program forward has no history kwarg
                    assert "history" in kwargs
                    return dspy.Prediction(answer=f"ok:{kwargs['question']}")

                self.gen.forward = fake_forward
                self.gen.__call__ = fake_forward

            def forward(self, question):
                # forward does NOT accept history
                return self.gen(question=question)

        prog = StrictProgram()
        session = Session(prog)

        session(question="Q1")
        session(question="Q2")

        assert len(session) == 2

    def test_program_without_history_custom_field_name(self):
        """Custom history_field works even when forward doesn't accept it."""

        class StrictProgram(dspy.Module):
            def __init__(self):
                super().__init__()
                self.gen = dspy.Predict(QASig)

                def fake_forward(**kwargs):
                    assert "chat_log" in kwargs
                    return dspy.Prediction(answer=f"ok:{kwargs['question']}")

                self.gen.forward = fake_forward
                self.gen.__call__ = fake_forward

            def forward(self, question):
                return self.gen(question=question)

        session = Session(StrictProgram(), history_field="chat_log")
        session(question="Q1")
        assert len(session) == 1


# ---------------------------------------------------------------------------
# sessionify() factory
# ---------------------------------------------------------------------------


class TestSessionify:
    def test_returns_session(self):
        predict = dspy.Predict(QASig)
        session = sessionify(predict)
        assert isinstance(session, Session)

    def test_passes_kwargs(self):
        predict = dspy.Predict(QASig)
        session = sessionify(predict, max_turns=5, exclude_fields={"context"})
        assert session.max_turns == 5
        assert "context" in session.exclude_fields

    def test_repr(self):
        predict = dspy.Predict(QASig)
        session = sessionify(predict)
        r = repr(session)
        assert "Session" in r
        assert "Predict" in r
        assert "turns=0" in r


# ---------------------------------------------------------------------------
# Safe serialization
# ---------------------------------------------------------------------------


class TestSafeSerialization:
    def test_preserves_list_structure(self):
        """Lists should be recursively serialized, not stringified."""
        d = {"items": [1, "two", {"nested": True}]}
        result = _safe_serialize(d)
        assert result["items"] == [1, "two", {"nested": True}]

    def test_preserves_nested_dicts(self):
        d = {"outer": {"inner": "value", "num": 42}}
        result = _safe_serialize(d)
        assert result["outer"] == {"inner": "value", "num": 42}

    def test_stringifies_unknown_types(self):
        class Custom:
            def __str__(self):
                return "custom_repr"

        d = {"obj": Custom()}
        result = _safe_serialize(d)
        assert result["obj"] == "custom_repr"

    def test_handles_pydantic_models(self):
        history = History(messages=[{"q": "hi"}])
        d = {"h": history}
        result = _safe_serialize(d)
        assert result["h"] == {"messages": [{"q": "hi"}]}


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_session_history(self):
        predict = dspy.Predict(QASig)
        session = Session(predict)
        assert session.session_history.messages == []

    def test_empty_session_to_examples(self):
        predict = dspy.Predict(QASig)
        session = Session(predict)
        assert session.to_examples() == []

    def test_reset_then_continue(self):
        responses = [{"answer": "A"}, {"answer": "B"}, {"answer": "C"}]
        predict = _fake_predict_factory(QASig, responses)
        session = Session(predict)

        session(question="Q1")
        session.reset()
        session(question="Q2")

        assert len(session) == 1
        assert session.turns[0].history_snapshot.messages == []
        assert session.turns[0].inputs["question"] == "Q2"

    def test_exclude_fields_not_in_history_but_in_turn(self):
        """Excluded fields are not in history messages but are in turn.inputs."""
        responses = [{"answer": "A"}]
        predict = _fake_predict_factory(RAGSig, responses)
        session = Session(predict, exclude_fields={"context"})

        session(question="Q1", context="big context")

        assert session.turns[0].inputs["context"] == "big context"
        assert "context" not in session.session_history.messages[0]

    def test_turn_not_recorded_on_error(self):
        """If the module raises, no turn should be recorded."""
        predict = dspy.Predict(QASig)

        def failing_forward(**kwargs):
            raise RuntimeError("LM failed")

        predict.forward = failing_forward
        predict.__call__ = failing_forward
        session = Session(predict)

        with pytest.raises(RuntimeError, match="LM failed"):
            session(question="Q1")

        assert len(session) == 0


# ---------------------------------------------------------------------------
# dspy.Module inheritance & optimizer compatibility
# ---------------------------------------------------------------------------


class TestModuleInheritance:
    def test_session_is_dspy_module(self):
        predict = dspy.Predict(QASig)
        session = Session(predict)
        assert isinstance(session, dspy.Module)

    def test_inner_predictor_discoverable(self):
        """named_predictors() should find the inner Predict module."""
        predict = dspy.Predict(QASig)
        session = Session(predict)

        predictor_names = [name for name, _ in session.named_predictors()]
        assert any("module" in name for name in predictor_names)

    def test_nested_in_agent_discoverable(self):
        """When session is nested in a larger agent, optimizer can find it."""

        class Agent(dspy.Module):
            def __init__(self):
                super().__init__()
                self.chat = sessionify(dspy.Predict(QASig))

            def forward(self, question):
                return self.chat(question=question)

        agent = Agent()
        predictor_names = [name for name, _ in agent.named_predictors()]
        assert len(predictor_names) >= 1, f"Expected predictors, got: {predictor_names}"

    def test_optimizer_bypass_with_explicit_history(self):
        """When history is passed explicitly, Session is a stateless pass-through."""
        responses = [{"answer": "A"}, {"answer": "B"}]
        predict = _fake_predict_factory(QASig, responses)
        session = Session(predict)

        # Normal call — records a turn
        session(question="Q1")
        assert len(session) == 1

        # Explicit history — optimizer bypass, no turn recorded
        explicit_history = History(messages=[{"question": "prior", "answer": "ctx"}])
        session(question="Q2", history=explicit_history)
        assert len(session) == 1  # still 1, not 2

    def test_optimizer_bypass_forwards_to_module(self):
        """Bypass mode still returns a valid result from the inner module."""
        responses = [{"answer": "bypass_result"}]
        predict = _fake_predict_factory(QASig, responses)
        session = Session(predict)

        result = session(
            question="Q1",
            history=History(messages=[]),
        )
        # Should get the result even in bypass mode
        assert result.answer == "bypass_result"


# ---------------------------------------------------------------------------
# Strict trajectory
# ---------------------------------------------------------------------------


class TestStrictTrajectory:
    def test_strict_trajectory_drops_after_first_failure(self):
        """strict_trajectory=True drops a bad turn and all subsequent ones."""
        responses = [
            {"answer": "good1"},
            {"answer": "good2"},
            {"answer": "bad"},
            {"answer": "good3"},  # has contaminated history
        ]
        predict = _fake_predict_factory(QASig, responses)
        session = Session(predict)

        for i in range(4):
            session(question=f"Q{i}")

        def metric(example, pred, trace=None):
            return 1.0 if "good" in pred.answer else 0.0

        examples = session.to_examples(
            metric=metric, min_score=0.5, strict_trajectory=True,
        )
        # Only turns 0 and 1 (before the bad turn)
        assert len(examples) == 2
        assert examples[0].question == "Q0"
        assert examples[1].question == "Q1"

    def test_non_strict_keeps_good_turns_after_bad(self):
        """Default (non-strict) keeps good turns even after a bad one."""
        responses = [
            {"answer": "good1"},
            {"answer": "bad"},
            {"answer": "good2"},
        ]
        predict = _fake_predict_factory(QASig, responses)
        session = Session(predict)

        for i in range(3):
            session(question=f"Q{i}")

        def metric(example, pred, trace=None):
            return 1.0 if "good" in pred.answer else 0.0

        examples = session.to_examples(
            metric=metric, min_score=0.5, strict_trajectory=False,
        )
        assert len(examples) == 2  # good1 and good2


# ---------------------------------------------------------------------------
# Module hot-swapping
# ---------------------------------------------------------------------------


class TestUpdateModule:
    def test_update_module_preserves_turns(self):
        responses = [{"answer": "A"}]
        predict = _fake_predict_factory(QASig, responses)
        session = Session(predict)
        session(question="Q1")

        new_predict = dspy.Predict(QASig)
        session.update_module(new_predict)

        assert len(session) == 1
        assert session.turns[0].inputs["question"] == "Q1"

    def test_update_module_patches_signature(self):
        responses = [{"answer": "A"}]
        predict = _fake_predict_factory(QASig, responses)
        session = Session(predict)

        new_predict = dspy.Predict(QASig)
        session.update_module(new_predict)

        # New module should have history field
        assert "history" in session.module.signature.input_fields


# ---------------------------------------------------------------------------
# Mutable input safety
# ---------------------------------------------------------------------------


class TestInputImmutability:
    def test_mutating_input_after_call_doesnt_corrupt_history(self):
        """Inputs are deep-copied, so post-call mutation doesn't affect recorded turns."""
        responses = [{"answer": "A"}, {"answer": "B"}]
        predict = _fake_predict_factory(QASig, responses)
        session = Session(predict)

        mutable_list = ["item1", "item2"]
        # Use a signature that accepts a list-like input
        session.add_turn(
            inputs={"question": "Q1", "extra": mutable_list},
            outputs={"answer": "A"},
        )

        # Mutate the original list AFTER recording
        mutable_list.append("item3")

        # The recorded turn should NOT be affected
        assert session.turns[0].inputs["extra"] == ["item1", "item2"]
