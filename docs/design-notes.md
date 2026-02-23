# dspy-session Design Notes

## Overview

`dspy-session` provides a stateful session wrapper for DSPy that:

1. Accumulates turn history automatically
2. Injects history into predictor calls
3. Linearizes multi-turn sessions into optimizer-ready examples
4. Works across adapters (Chat/JSON/XML/Template)

The central design goal is to make multi-turn behavior **orthogonal** to adapters while preserving DSPy-native optimization workflows.

---

## Core Design Principles

### 1) History is a Signature-level concern, not an Adapter concern

All DSPy adapters ultimately consume module inputs and signatures. `dspy.History` lives in this layer.

`dspy-session` therefore:
- patches predictor signatures to include a history field when missing
- injects history in predictor calls
- does **not** depend on any specific adapter implementation

### 2) Program support via predictor-level injection

Programs often have `forward(self, question)` and do not accept a `history` kwarg.

To support these programs, `dspy-session` wraps nested predictors and injects history using a contextvar. This avoids requiring top-level `forward()` signature changes.

### 3) Optimizer-safe behavior

When explicit history is passed in a call, session can run in stateless bypass mode (`history_policy="override"`).

This enables optimizer replay of examples without polluting session state.

### 4) Linearization-first data model

Each turn stores a history snapshot at call time. `to_examples()` emits independent `dspy.Example`s:

- inputs: current turn inputs (+ history snapshot, unless disabled)
- labels: turn outputs

This aligns with DSPy optimizers that expect independent examples.

---

## API Surface (v2 format)

### Construction

```python
Session(
    module,
    history_field="history",
    max_turns=None,
    exclude_fields=None,
    history_input_fields=None,     # alias: input_field_override
    initial_history=None,
    history_policy="override",    # override | use_if_provided | replace_session
    on_metric_error="zero",       # zero | raise
    strict_history_annotation=False,
    lock="none",                  # none | thread | async
)
```

### Runtime calls

- `session(...)`
- `session.forward(...)`
- `await session.acall(...)`
- `await session.aforward(...)`

### Session editing

- `add_turn(inputs, outputs)`
- `pop_turn()`
- `undo(steps=1)`
- `reset()`
- `fork()`

### Data extraction

- `score(metric, gold=None)`
- `to_examples(...)`
- `to_trainset(...)`
- `Session.merge_examples(...)`

### Persistence / lifecycle

- `save(path)`
- `save_state()`
- `Session.load_from(path, module)`
- `update_module(new_module)`

---

## History Policies

`history_policy` controls behavior when caller provides explicit history:

- `override` (default): stateless pass-through, no turn recording
- `use_if_provided`: use explicit history and record turn
- `replace_session`: replace seed history, clear turns, continue statefully

---

## Scoring + Filtering

`to_examples()` supports:

- metric-based filtering (`min_score`)
- `strict_trajectory=True`:
  - if a turn fails threshold, that turn and all subsequent turns are dropped
  - useful when later turns are contaminated by bad earlier context

---

## Serialization Format

Current format version: **2**

Stored fields:
- session config (`history_field`, window/filter params, policy)
- optional `initial_history`
- turns (`index`, `inputs`, `outputs`, `score`)

On load, history snapshots are rebuilt via the same `_build_history()` logic used at runtime, ensuring consistency with exclusion and windowing rules.

---

## Review-driven Decisions

The implementation incorporates feedback from multi-model review:

- predictor-level injection for true program support
- robust annotation handling for history fields
- explicit history policy modes
- strict trajectory filtering mode
- deep-copy recording for mutable input safety
- module hot-swap support
- lock modes for concurrent environments
- README usage examples validated by tests

---

## Known Tradeoffs / Future Work

1. Advanced memory reduction:
   - token-budget reducers
   - summarization reducers

2. Optional lazy snapshot storage for extremely long sessions

3. Optional plugin hooks (pre-turn/post-turn callbacks)

4. Expanded multimodal history controls

---

## Upstreaming Considerations

If proposing upstream to DSPy:

- keep adapter-agnostic design intact
- preserve optimizer bypass behavior
- preserve predictor-level program support
- align history annotation detection with DSPy internal helpers where possible
- keep `to_examples()` output fully compatible with DSPy `Example` conventions
