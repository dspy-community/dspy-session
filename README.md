# dspy-session

`dspy-session` adds stateful, multi-turn behavior to DSPy modules while staying adapter-agnostic.

It works with:
- `dspy.Predict` / `dspy.ChainOfThought`
- `dspy.ProgramOfThought`
- `dspy.ReAct`
- `dspy.CodeAct`
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

## Examples

### 1. Multi-turn Q&A — the basics

Wrap any predictor with `sessionify`. Each call automatically accumulates
conversation history so the model sees prior turns.

```python
import dspy
from dspy_session import sessionify

dspy.configure(lm=dspy.LM("groq/moonshotai/kimi-k2-instruct-0905"))

class chatsig(dspy.Signature):
    """You are a helpful assistant"""
    user: str = dspy.InputField()
    assistant: str = dspy.OutputField()

chat = sessionify(dspy.Predict(chatsig))
chat
```

    Session(Predict, turns=0, history_field='history')


So If I introduce myself.

```python
chat(user="Hi! My name is Max")
```

```output:exec-1771881361855-kpp6w
Prediction(
    assistant='Hi Max! Nice to meet you. How can I help you today?'
)
```


```python
# Turn 2 — the model remembers your name from turn 1
out2 = session(question="What is my name?")
print(out2.answer)
# Your name is Max.
```

```python
# Inspect accumulated state
print(f"Turns so far: {len(session.turns)}")
# Turns so far: 2
```

```python
print(session.session_history)
# messages=[
#     {'question': 'Hi! My name is Max and I'm learning DSPy.', 'answer': 'Hello Max! ...'},
#     {'question': 'What is my name?', 'answer': 'Your name is Max.'}
# ]
```



**What happens under the hood:**
1. `sessionify` deep-copies the module and adds a `history` input field to its signature
2. Each call builds a `dspy.History` from all previous turns
3. That history is injected into the predictor so the LM sees it
4. The completed turn is recorded with an exact `history_snapshot`

---

### 2. Wrapping a composed program (no signature changes needed)

You don't need to modify your existing `dspy.Module`. Session wraps the
nested predictors and injects history via context variables — your `forward()`
stays clean.

```python
import dspy
from dspy_session import sessionify

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

class TravelAdvice(dspy.Signature):
    """Give helpful travel advice."""
    question: str = dspy.InputField()
    advice: str = dspy.OutputField()

class TravelAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.advisor = dspy.ChainOfThought(TravelAdvice)

    def forward(self, question):
        # No history parameter here — session handles it automatically
        return self.advisor(question=question)

session = sessionify(TravelAgent())

out1 = session(question="I'm planning a 2-week trip to Japan in April.")
print(out1.advice)
# April is perfect for cherry blossoms! I'd suggest splitting time between
# Tokyo, Kyoto, and Osaka...

out2 = session(question="What should I pack for the weather?")
print(out2.advice)
# For Japan in April, expect mild temperatures (10-20°C). Pack layers,
# a light rain jacket, and comfortable walking shoes...

out3 = session(question="Any food recommendations for the cities you mentioned?")
print(out3.advice)
# In Tokyo, try Tsukiji Outer Market for sushi. In Kyoto, don't miss
# kaiseki dining. In Osaka, head to Dotonbori for takoyaki and okonomiyaki...
```

Notice that by turn 3, the model remembers both the destination (Japan) and the
specific cities it recommended — all without you passing any history manually.

---

### 3. RAG assistant — keeping history lean with `exclude_fields`

When your signature includes large inputs like retrieved documents, you don't
want those bloating the conversation history. Use `exclude_fields` to keep
history focused on the actual dialogue.

```python
import dspy
from dspy_session import sessionify

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

class RAGAnswer(dspy.Signature):
    """Answer the question using the provided context."""
    question: str = dspy.InputField()
    context: str = dspy.InputField(desc="Retrieved documents")
    answer: str = dspy.OutputField()

# Exclude 'context' from history — it's different each turn and can be huge
session = sessionify(
    dspy.Predict(RAGAnswer),
    exclude_fields={"context"},
    max_turns=10,  # sliding window: only the last 10 turns enter the prompt
)

# Turn 1 — retriever finds Python docs
session(
    question="What are Python decorators?",
    context="[PEP 318] A decorator is a callable that takes a function and returns ..."
)

# Turn 2 — retriever finds different docs, but history has turn 1's Q&A
session(
    question="Can you show me a real-world example of what you just explained?",
    context="[RealPython] @login_required is a common decorator in Flask that ..."
)

# The history contains questions and answers, but NOT the bulky context
for msg in session.session_history.messages:
    print(msg.keys())
    # dict_keys(['question', 'answer'])  — no 'context'

# The full inputs are still saved in turns for debugging
print(session.turns[0].inputs.keys())
# dict_keys(['question', 'context'])  — context is preserved in the turn record
```

You can also use `history_input_fields` to allow-list only specific fields instead:

```python
# Equivalent: only 'question' enters history (everything else is excluded)
session = sessionify(
    dspy.Predict(RAGAnswer),
    history_input_fields={"question"},
)
```

---

### 4. Generating optimizer training data from conversations

Every turn is automatically linearized into an independent `dspy.Example` with
its history snapshot — ready for DSPy optimizers like `BootstrapFewShot`.

```python
import dspy
from dspy_session import Session, sessionify

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

class MathTutor(dspy.Signature):
    """You are a patient math tutor. Explain step by step."""
    question: str = dspy.InputField()
    explanation: str = dspy.OutputField()

# Simulate a tutoring conversation
session = sessionify(dspy.Predict(MathTutor))
session(question="What is a derivative?")
session(question="Can you give me an example with f(x) = x²?")
session(question="What about the chain rule?")
session(question="Apply the chain rule to f(x) = sin(x²)")

# Each turn becomes a standalone training example with its history snapshot
examples = session.to_examples()

print(f"Generated {len(examples)} training examples")
# Generated 4 training examples

# Example 0: no prior history (it was the first turn)
ex0 = examples[0]
print(f"  Input: {ex0.question}")
print(f"  History length: {len(ex0.history.messages)}")
print(f"  Label: {ex0.explanation[:60]}...")
# Input: What is a derivative?
# History length: 0
# Label: A derivative measures the rate of change of a function...

# Example 2: has 2 prior turns as history
ex2 = examples[2]
print(f"  Input: {ex2.question}")
print(f"  History length: {len(ex2.history.messages)}")
# Input: What about the chain rule?
# History length: 2

# These examples are ready for optimization
optimized = dspy.BootstrapFewShot().compile(session, trainset=examples)
```

---

### 5. Quality filtering with metrics

Score each turn and filter to only keep high-quality examples. This is critical
when collecting training data from real user conversations where some turns may
be low quality.

```python
import dspy
from dspy_session import Session, sessionify

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

class CodeHelp(dspy.Signature):
    """Help the user with their coding question. Include a code example."""
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

session = sessionify(dspy.Predict(CodeHelp))
session(question="How do I read a CSV file in Python?")
session(question="What if the file is really large?")
session(question="How do I filter rows where age > 30?")

# Define a quality metric — here we check if the answer contains code
def has_code_example(example, pred, trace=None):
    """Score 1.0 if the answer contains a code block, 0.0 otherwise."""
    return 1.0 if "```" in pred.answer or "import " in pred.answer else 0.0

# Score all turns and keep only those with code examples
good_examples = session.to_examples(metric=has_code_example, min_score=0.5)
print(f"Kept {len(good_examples)} of {len(session.turns)} turns")

# strict_trajectory mode: if a bad turn appears, discard it AND all later turns
# (because later turns may rely on a flawed answer)
strict_examples = session.to_examples(
    metric=has_code_example,
    min_score=0.5,
    strict_trajectory=True,
)
print(f"Strict kept: {len(strict_examples)} turns")
```

#### Merging examples from multiple sessions

```python
session_alice = sessionify(dspy.Predict(CodeHelp))
session_alice(question="How do I sort a list?")
session_alice(question="What about sorting by a custom key?")

session_bob = sessionify(dspy.Predict(CodeHelp))
session_bob(question="How do I connect to a Postgres database?")
session_bob(question="How do I run a parameterized query?")

# Merge all sessions into one training set
trainset = Session.merge_examples(session_alice, session_bob)
print(f"Total training examples: {len(trainset)}")
# Total training examples: 4
```

---

### 6. Save, load, and resume conversations

Persist session state to disk and restore it later — useful for long-running
services, user session management, or picking up where you left off.

```python
import dspy
from dspy_session import Session, sessionify

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

class Therapist(dspy.Signature):
    """You are a supportive therapist. Be empathetic and ask follow-up questions."""
    message: str = dspy.InputField()
    response: str = dspy.OutputField()

# --- Day 1: initial conversation ---
session = sessionify(dspy.Predict(Therapist))
session(message="I've been feeling overwhelmed at work lately.")
session(message="My manager keeps adding tasks without checking my bandwidth.")

# Save to disk (module weights are NOT saved — only turn history and config)
session.save("therapy_session.json")
print(f"Saved {len(session.turns)} turns to disk")

# --- Day 2: resume the conversation ---
fresh_module = dspy.Predict(Therapist)
restored = Session.load_from("therapy_session.json", fresh_module)

print(f"Restored {len(restored.turns)} turns")
# Restored 2 turns

# Continue the conversation — the model sees the full prior context
out = restored(message="I tried setting boundaries like you suggested.")
print(out.response)
```

#### Starting a session with seed history

Pre-load context without recording it as a regular turn — useful for system
prompts or prior conversation summaries:

```python
import dspy
from dspy_session import sessionify

seed = dspy.History(messages=[
    {"message": "I have anxiety about public speaking.", "response": "That's very common..."},
    {"message": "I have a presentation next week.", "response": "Let's work on some strategies..."},
])

session = sessionify(dspy.Predict(Therapist), initial_history=seed)

# First call already sees the seed history
out = session(message="I tried the breathing exercises. They helped a little.")
print(len(session.turns))            # 1 (seed messages don't count as turns)
print(len(session.session_history.messages))  # 3 (2 seed + 1 new)
```

---

### 7. Forking a conversation

Create a branch of the conversation to explore different directions without
affecting the original — great for A/B testing responses or "what if" scenarios.

```python
import dspy
from dspy_session import sessionify

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

class StoryWriter(dspy.Signature):
    """Continue the collaborative story."""
    prompt: str = dspy.InputField()
    continuation: str = dspy.OutputField()

session = sessionify(dspy.Predict(StoryWriter))
session(prompt="A detective arrives at an abandoned mansion on a rainy night.")
session(prompt="She finds a locked room on the second floor.")

# Fork: explore two different story directions
branch_a = session.fork()
branch_b = session.fork()

branch_a(prompt="She picks the lock and finds a hidden laboratory.")
branch_b(prompt="She hears footsteps coming from inside the room.")

print(f"Original: {len(session.turns)} turns")    # 2
print(f"Branch A: {len(branch_a.turns)} turns")   # 3
print(f"Branch B: {len(branch_b.turns)} turns")   # 3

# Each branch is fully independent — original is untouched
```

---

### 8. Manual turn editing

Manually add, remove, or correct turns. Useful for building curated training
data or correcting a bad model response before continuing.

```python
import dspy
from dspy_session import sessionify

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

class QA(dspy.Signature):
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

session = sessionify(dspy.Predict(QA))

# Manually inject a turn (e.g., from a human-written gold example)
session.add_turn(
    inputs={"question": "What is the capital of France?"},
    outputs={"answer": "The capital of France is Paris."},
)

# The next call sees this manually-added turn in its history
out = session(question="What about Germany?")
print(out.answer)
# The capital of Germany is Berlin.

# Oops — undo the last model response
session.pop_turn()
print(f"Turns after pop: {len(session.turns)}")
# Turns after pop: 1

# Or undo multiple turns at once
session.undo(steps=1)
print(f"Turns after undo: {len(session.turns)}")
# Turns after undo: 0
```

---

### 9. Hot-swapping an optimized module

Keep the user's conversation going while upgrading the underlying model or
loading optimized weights — the session state carries over seamlessly.

```python
import dspy
from dspy_session import sessionify

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

class Support(dspy.Signature):
    """You are a customer support agent for a SaaS product."""
    question: str = dspy.InputField()
    reply: str = dspy.OutputField()

base_module = dspy.Predict(Support)
session = sessionify(base_module)

# User starts a conversation
session(question="I can't log in to my account.")
session(question="I've already tried resetting my password.")

# Meanwhile, you've optimized a better module offline
trainset = session.to_examples()
optimized_module = dspy.BootstrapFewShot().compile(
    dspy.Predict(Support), trainset=trainset
)

# Swap in the optimized module — conversation state is preserved
session.update_module(optimized_module)

# Continue with the better model — it still sees the full conversation
out = session(question="The reset email never arrived.")
print(out.reply)
print(f"Total turns (uninterrupted): {len(session.turns)}")
# Total turns (uninterrupted): 3
```

---

### 10. Async conversations

For async applications (web servers, Discord bots, etc.), use `acall` with
an async lock to prevent race conditions from concurrent requests.

```python
import asyncio
import dspy
from dspy_session import sessionify

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

class Chat(dspy.Signature):
    """A friendly conversational assistant."""
    message: str = dspy.InputField()
    reply: str = dspy.OutputField()

async def main():
    session = sessionify(dspy.Predict(Chat), lock="async")

    out1 = await session.acall(message="Hey! What's a good beginner programming language?")
    print(out1.reply)

    out2 = await session.acall(message="Why do you recommend that one?")
    print(out2.reply)

    print(f"Turns: {len(session.turns)}")

asyncio.run(main())
```

Lock options:
- `"none"` (default) — no synchronization
- `"thread"` — `threading.Lock` for multi-threaded apps
- `"async"` — `asyncio.Lock` for async apps

---

### 11. History policies for optimizer replay

When an optimizer calls your session with an explicit `history` argument (e.g.,
during `BootstrapFewShot` replay), you need to control what happens. History
policies handle this:

```python
import dspy
from dspy_session import sessionify

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

class QA(dspy.Signature):
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

# --- "override" (default) — stateless pass-through for optimizer replay ---
session = sessionify(dspy.Predict(QA), history_policy="override")
session(question="Hi!")  # recorded as turn 1

# Optimizer replays with explicit history → NOT recorded, session state unchanged
result = session(
    question="Follow-up",
    history=dspy.History(messages=[{"question": "Hi!", "answer": "Hello!"}])
)
print(len(session.turns))  # still 1

# --- "use_if_provided" — use provided history AND record the turn ---
session2 = sessionify(dspy.Predict(QA), history_policy="use_if_provided")
session2(
    question="Follow-up",
    history=dspy.History(messages=[{"question": "Hi!", "answer": "Hello!"}])
)
print(len(session2.turns))  # 1 (turn was recorded)

# --- "replace_session" — replace seed history, clear turns, start fresh ---
session3 = sessionify(dspy.Predict(QA), history_policy="replace_session")
session3(question="Turn 1")
session3(question="Turn 2")
print(len(session3.turns))  # 2

# Replacing: clears all turns, installs new seed history, then records this call
session3(
    question="Fresh start",
    history=dspy.History(messages=[{"question": "Seed", "answer": "Context"}])
)
print(len(session3.turns))  # 1 (old turns cleared, new turn recorded)
```

---

## TemplateAdapter integration

`dspy-session` and `dspy-template-adapter` work extremely well together:

- `dspy-session` handles state/history lifecycle
- `TemplateAdapter` gives exact prompt-layout control

👉 See the full integration guide (including contrived layouts like "all history in system", "all in user", and split strategies):

- [`docs/template-adapter-integration.md`](docs/template-adapter-integration.md)

---

## API reference

```python
sessionify(module, **kwargs) -> Session

Session(
    module,
    history_field="history",
    max_turns=None,
    max_stored_turns=None,
    exclude_fields=None,
    history_input_fields=None,      # alias: input_field_override
    initial_history=None,
    history_policy="override",     # override | use_if_provided | replace_session
    on_metric_error="zero",        # zero | raise
    strict_history_annotation=False,
    copy_mode="deep",              # deep | shallow | none
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
- For very long sessions, use `max_turns` and `max_stored_turns` to control prompt and memory growth.
- If your module is not deepcopy-friendly, use `copy_mode="shallow"` or `copy_mode="none"`.

## License

MIT
