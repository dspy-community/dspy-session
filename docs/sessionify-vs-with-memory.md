# `sessionify` vs `with_memory`

This guide explains the difference between `sessionify(...)` and `with_memory(...)`, and exactly when to use each.

---

## TL;DR

- Use **`sessionify`** when you want a straightforward stateful wrapper for one module/program (notebook scripts, local tools, quick experiments).
- Use **`with_memory`** when you want a **shared blueprint** with **per-user/per-request external state**, recursive child policies, and production concurrency safety.

---

## Mental model

## 1) `sessionify`: instance-centric state

`sessionify(module)` gives you a `Session` where state naturally lives with that session object (or in a bound external state if you choose to use `use_state`).

Think: **"this wrapped object is my chat/session"**.

```python
chat = sessionify(my_module)
chat(question="Hi")
chat(question="Follow-up")
print(len(chat.turns))  # same object owns this state
```

Best for:
- notebooks
- single-user scripts
- offline dataset generation
- quick prototyping

---

## 2) `with_memory`: blueprint + external state

`with_memory(module, ...)` also returns a `Session`, but it is configured as a **shared blueprint**:
- `copy_mode="none"` by default (reuse one module instance)
- recursive wrapping/policy control (`child_configs`, `isolation`, `lifespan`, `consolidator`)
- intended to be paired with `SessionState` per user/request

Think: **"this is my model runtime blueprint; user memory lives outside"**.

```python
app = with_memory(my_agent)           # shared blueprint
state = app.new_state()               # per-user memory

with app.use_state(state):
    app(question="Hi")
```

Best for:
- FastAPI/Django/ASGI servers
- multi-tenant chat systems
- workflows with per-node memory policies
- long-lived services with persisted user state

---

## Side-by-side comparison

| Concern | `sessionify` | `with_memory` |
|---|---|---|
| Primary intent | Simple stateful wrapper | Production blueprint + policy engine |
| Default copy behavior | `copy_mode="deep"` | `copy_mode="none"` |
| Default recursive wrapping | Off (unless you pass it manually) | On (`recursive=True`) |
| Per-node policy controls | Available, but not the main point | First-class (`child_configs`, lifespan/isolation) |
| External state binding | Supported | Core pattern |
| Best fit | Local / notebook / single flow | Server / concurrent multi-user |

---

## Decision rule

If you are asking:

- **"I just need memory for this one wrapped program"** → use `sessionify`
- **"I need one global app with isolated state per user/request"** → use `with_memory`
- **"I need per-child policies (episodic/stateless/shared/consolidator)"** → use `with_memory`

---

## Typical usage patterns

## Pattern A — quick local session (`sessionify`)

```python
chat = sessionify(dspy.Predict("question -> answer"))
chat(question="Q1")
chat(question="Q2")
examples = chat.to_examples()
```

## Pattern B — production shared blueprint (`with_memory`)

```python
app = with_memory(MyAgent())

# request handler
state = SessionState.from_dict(db.load(user_id)) if db.exists(user_id) else app.new_state()

async with app.use_state(state):
    out = await app.acall(question=message)

# persist user memory only (lightweight)
db.save(user_id, state.to_dict())
```

## Pattern C — policy-driven agent memory (`with_memory`)

```python
app = with_memory(
    MyAgent(),
    child_configs={
        "query_generator": {
            "lifespan": "episodic",
            "consolidator": my_consolidator,
        },
        "retriever": {
            "lifespan": "stateless",
        },
        "critic": {
            "isolation": "shared",
        },
    },
)
```

---

## Important caveats

## 1) Don’t use a single unbound stateful object for all users

If you serve a multi-user API and call a shared session without `use_state(...)`, all traffic can accumulate into the same default state.

Use per-user `SessionState` + `with app.use_state(state): ...`.

## 2) `with_memory` is not a different class

It returns a `Session` too. The difference is **defaults + intended architecture**:
- blueprint defaults (`copy_mode="none"`)
- recursive wrapping enabled
- policy-centric usage

## 3) You can still use external state with `sessionify`

`sessionify` also supports `new_state()` / `use_state()`. The recommended distinction is about **intent**:
- `sessionify`: simple wrapper first
- `with_memory`: blueprint/runtime split first

---

## Migration guide

## From local prototype to production

Start:

```python
chat = sessionify(MyAgent())
chat(question="Hi")
```

Move to production:

```python
app = with_memory(MyAgent())
state = app.new_state()
with app.use_state(state):
    app(question="Hi")
```

Add policies later without changing app code structure:

```python
app = with_memory(MyAgent(), child_configs={...})
```

---

## Summary

- `sessionify` = easiest path to stateful behavior.
- `with_memory` = architecture for scalable, policy-driven, concurrent serving.

If in doubt: start with `sessionify`, switch to `with_memory` once you need per-user state persistence and per-node policy control.
