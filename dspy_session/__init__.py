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

from dspy_session.session import (
    Session,
    SessionState,
    Turn,
    get_child_l1_ledger,
    get_current_history,
    get_execution_trace,
    get_node_memory,
    get_outer_history,
    sessionify,
    with_memory,
)

__all__ = [
    "Session",
    "SessionState",
    "Turn",
    "sessionify",
    "with_memory",
    "get_current_history",
    "get_outer_history",
    "get_node_memory",
    "get_child_l1_ledger",
    "get_execution_trace",
]
