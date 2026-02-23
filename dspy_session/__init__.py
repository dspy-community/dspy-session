"""
dspy-session — Multi-turn session wrapper for DSPy programs.

Turn any DSPy module into a stateful conversation. Each call accumulates
history automatically. Sessions linearize into optimizer-ready training
examples where every turn is an independent (history, input) → output example.

Works with any DSPy adapter: ChatAdapter, JSONAdapter, XMLAdapter, TemplateAdapter, etc.

Usage:
    from dspy_session import sessionify

    session = sessionify(dspy.Predict(MySig))
    out1 = session(question="Hi")
    out2 = session(question="Follow up")
    session.reset()

    # Linearize for optimization
    examples = session.to_examples()
"""

from dspy_session.session import Session, Turn, sessionify

__all__ = ["Session", "Turn", "sessionify"]
