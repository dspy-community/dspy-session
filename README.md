# dspy-session

`dspy-session` adds stateful, multi-turn behavior to DSPy modules while staying adapter-agnostic.

It works with:
- `dspy.Predict` / `dspy.ChainOfThought`
- composed `dspy.Module` programs with nested predictors
- any adapter (`ChatAdapter`, `JSONAdapter`, `XMLAdapter`, `TemplateAdapter`, ...)

## Install

```bash
pip install dspy-session
```

---

## Why this exists

DSPy already has `dspy.History`, but it is manual:

```python
history = dspy.History(messages=[...])
out = predictor(question="...", history=history)
```

`dspy-session` automates this and adds:
- per-turn state accumulation
- turn snapshots (`history_snapshot` at call time)
- linearization into optimizer-ready `dspy.Example`s
- per-turn scoring + filtering

---

## Quickstart (single predictor)

```python
import dspy
from dspy_session import sessionify

class QA(dspy.Signature):
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

session = sessionify(dspy.Predict(QA))

session(question="What is DSPy?")
session(question="How is it different from plain prompting?")

print(len(session.turns))
print(session.session_history)
```

What happens:
1. Session deep-copies the module
2. It ensures predictors have a history input field
3. Each call builds a `History` from previous turns
4. That history is injected into predictor calls
5. Turn is recorded with an exact `history_snapshot`

---

## Program wrapping (no `history` kwarg required)

You do **not** need to modify your program’s `forward` signature.

```python
import dspy
from dspy_session import sessionify

class QA(dspy.Signature):
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

class Agent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.gen = dspy.Predict(QA)

    def forward(self, question):
        # No history argument here.
        return self.gen(question=question)

session = sessionify(Agent())
```

Internally, `dspy-session` wraps nested predictors and injects history via context when not explicitly provided.

---

## Linearization for optimizers

Each turn becomes an independent training example:

```python
examples = session.to_examples()
# each example has inputs including history snapshot + output labels
```

### Filtering with a metric

```python
def quality_metric(example, pred, trace=None):
    return 1.0 if "good" in pred.answer.lower() else 0.0

good_examples = session.to_examples(metric=quality_metric, min_score=0.5)
```

### Strict trajectory mode

If a bad turn should invalidate all later turns in that session:

```python
strict_examples = session.to_examples(
    metric=quality_metric,
    min_score=0.5,
    strict_trajectory=True,
)
```

---

## Optimizer workflow

If your examples include history (`include_history=True`, default), optimize a module/signature that accepts history.

Typical workflow:

```python
# collect turns
session = sessionify(dspy.Predict(QA))
...
trainset = session.to_examples()

# optimize the session-aware module
optimized = dspy.BootstrapFewShot().compile(session, trainset=trainset)
```

If you want to optimize a non-history base program, use:

```python
trainset = session.to_examples(include_history=False)
```

---

## History policies for explicit history input

When caller passes `history=...` explicitly:

- `history_policy="override"` (default): stateless pass-through, no turn recorded
- `history_policy="use_if_provided"`: use provided history for this call and record turn
- `history_policy="replace_session"`: replace session seed history, clear turns, then continue

```python
session = sessionify(my_module, history_policy="use_if_provided")
```

---

## Controlling what enters history

### Sliding window

```python
session = sessionify(my_module, max_turns=10)
```

### Excluding fields (e.g. giant RAG context)

```python
session = sessionify(my_module, exclude_fields={"context"})
```

### Include only selected input fields in history

```python
session = sessionify(my_module, history_input_fields={"question"})
```

---

## Seed history / resume conversations

```python
seed = dspy.History(messages=[{"question": "Hi", "answer": "Hello!"}])
session = sessionify(my_module, initial_history=seed)
```

---

## Manual turn editing

```python
session.add_turn(
    inputs={"question": "edited question"},
    outputs={"answer": "edited answer"},
)

session.pop_turn()     # remove last turn
session.undo(steps=2) # remove last 2 turns
```

---

## Serialization

Save/load session state (module is not serialized):

```python
session.save("session.json")

restored = dspy_session.Session.load_from("session.json", my_module)
```

---

## Hot-swap optimized module

Keep user conversation state, replace model/program weights:

```python
session.update_module(new_optimized_module)
```

---

## Async + locks

```python
session = sessionify(my_module, lock="async")
out = await session.acall(question="...")
```

`lock` options:
- `"none"` (default)
- `"thread"`
- `"async"`

---

## API reference

```python
sessionify(module, **kwargs) -> Session

Session(
    module,
    history_field="history",
    max_turns=None,
    exclude_fields=None,
    history_input_fields=None,      # alias: input_field_override
    initial_history=None,
    history_policy="override",     # override | use_if_provided | replace_session
    on_metric_error="zero",        # zero | raise
    strict_history_annotation=False,
    lock="none",                   # none | thread | async
)

# calls
session(...)
session.forward(...)
await session.acall(...)
await session.aforward(...)

# state
session.turns
session.session_history
len(session)
session.reset()
session.fork()

# manual editing
session.add_turn(inputs=..., outputs=...)
session.pop_turn()
session.undo(steps=1)

# scoring / examples
session.score(metric, gold=None)
session.to_examples(...)
session.to_trainset(...)
Session.merge_examples(*sessions, ...)

# persistence
session.save(path)
session.save_state()
Session.load_from(path, module)

# module lifecycle
session.update_module(new_module)
```

---

## Notes

- `Session` is a `dspy.Module`, so nested use inside larger DSPy programs remains optimizer-discoverable via `named_predictors()`.
- For optimizer replay calls that pass explicit history, Session can behave statelessly (`history_policy="override"`).
- For very long sessions, use `max_turns` and/or strict filtering strategies.

## License

MIT
