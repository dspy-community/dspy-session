# dspy-session v2: Recursive & Nested Sessions

> **Collaborative design doc.** Edit freely, run code cells, leave comments.
> Status: **DRAFT — exploring the design space**

---

## Table of Contents

- [1. Problem Statement](#1-problem-statement)
- [2. Three Invariants](#2-three-invariants)
- [3. Design Decisions](#3-design-decisions)
  - [3.1 What does "recursive" mean?](#31-what-does-recursive-mean)
  - [3.2 Nesting policy: what happens when Session calls Session?](#32-nesting-policy-what-happens-when-session-calls-session)
  - [3.3 History mode: per-predictor vs outer-conversation](#33-history-mode-per-predictor-vs-outer-conversation)
  - [3.4 Call-level tracing (CallRecord)](#34-call-level-tracing-callrecord)
- [4. Proposed API](#4-proposed-api)
  - [4.1 `sessionify()` — extended signature](#41-sessionify--extended-signature)
  - [4.2 `SessionContainer` — explicit composition](#42-sessioncontainer--explicit-composition)
  - [4.3 Navigation & introspection](#43-navigation--introspection)
  - [4.4 Cascading operations](#44-cascading-operations)
  - [4.5 Dev/optimization extraction](#45-devoptimization-extraction)
- [5. Concrete Scenarios](#5-concrete-scenarios)
  - [5.1 Single predictor — baseline (already works)](#51-single-predictor--baseline-already-works)
  - [5.2 Composed module — recursive sessionify](#52-composed-module--recursive-sessionify)
  - [5.3 Pre-sessionified components — explicit composition](#53-pre-sessionified-components--explicit-composition)
  - [5.4 RAG with exclude_fields inside recursive tree](#54-rag-with-exclude_fields-inside-recursive-tree)
- [6. Wire Format: What Hits the LLM](#6-wire-format-what-hits-the-llm)
  - [6.1 Recommended Default: `"keep"` = Strategy 1](#61-recommended-default-nested_sessionskeep--strategy-1)
  - [6.2 Opt-in: `"inherit"` + TemplateAdapter = Strategies 3-5](#62-opt-in-nested_sessionsinherit--templateadapter--strategies-3-5)
  - [6.3 Strategy Selection Flowchart](#63-summary-the-strategy-selection-flowchart)
  - [6.4 State Scope: Within a Forward Pass vs Across Them](#64-state-scope-within-a-forward-pass-vs-across-them)
- [7. Optimizer Time-Travel](#7-optimizer-time-travel)
- [8. Implementation Skeleton](#8-implementation-skeleton)
- [9. Open Questions](#9-open-questions)
- [10. Migration & Rollout](#10-migration--rollout)

---

## 1. Problem Statement

Today `dspy-session` handles **one level**: you sessionify a module, and its inner predictors get history injected via a contextvar. This works great for flat pipelines.

But real agents are **trees**:

```
AgentOrchestrator
├── planner (ChainOfThought)
├── researcher (Module)
│   ├── retriever (Predict)
│   └── summarizer (Predict)
├── executor (ReAct)
└── writer (Predict)
```

Two patterns emerge that the current API doesn't handle:

**Pattern A — "Sessionify the whole tree"**: I have a `dspy.Module` with nested sub-modules. I want `sessionify(module, recursive=True)` to make the whole thing stateful. Each inner component should track its own history or share the parent's — I need control.

**Pattern B — "Compose pre-sessionified pieces"**: I built and tested `planner`, `researcher`, `executor` as independent sessions. Now I want to compose them into an orchestrator without double-history-injection or conflicting state.

Both patterns need to work cleanly in **production** (zero-boilerplate, persistence, concurrency) and **dev/optimization** (per-component traces, targeted extraction, optimizer replay).

---

## 2. Three Invariants

These rules make the system predictable. If we break any of them, things get confusing.

1. **One "turn" per user-facing call.** The boundary where multi-turn state lives is the outermost session. Everything inside that call is a "step", not a turn.

2. **Inner modules don't mutate their own turn state by default** when called inside a parent session context. (This is `nested_sessions="inherit"` — the default.)

3. **The `history_snapshot` on each turn is exactly the history that was sent to the LLM for that turn.** This makes `to_examples()` produce faithful, replayable training data.

---

## 3. Design Decisions

### 3.1 What does "recursive" mean?

Three scopes, chosen by a single parameter:

| Value | What gets wrapped | Traversal method |
|---|---|---|
| `recursive=False` | Nothing new (current behavior) | — |
| `recursive="predictors"` | Every `dspy.Predict` / `CoT` / `ReAct` / etc. | `module.named_predictors()` |
| `recursive="modules"` | All `dspy.Module` attributes (including non-predictor sub-modules) | Deep attribute walk |

**Default: `False`** (backward compatible). We recommend `"predictors"` for most use cases.

**Filtering** — only wrap what needs history:

```python
sessionify(
    agent,
    recursive="predictors",
    include={"planner", "writer"},              # allowlist by name
    exclude={"retriever", "reranker"},           # blocklist by name
    # OR predicate:
    where=lambda path, obj: "retriever" not in path,
)
```

> **DECISION NEEDED:** Should `recursive=True` be an alias for `"predictors"`? It reads better as a boolean for the simple case.

<!-- VOTE: yes / no / True maps to "predictors", "all" maps to "modules" -->


### 3.2 Nesting policy: what happens when Session calls Session?

This is the crux. When a `Session` is invoked while another `Session`'s forward is active:

| Policy | Behavior | Use case |
|---|---|---|
| `"inherit"` | Inner session does NOT record turns. Uses parent's history snapshot. Inner calls are traced as `CallRecord`s on the parent turn. | Default — safe composition |
| `"keep"` | Inner session records its own turns independently. Uses its own accumulated history. | Sub-agent with private memory (planner/critic with evolving state) |
| `"unwrap"` | Treat inner `Session` as bare `session.module`. Parent's contextvar injection applies. | Accidental double-wrap, or library modules that ship pre-sessionified |
| `"error"` | Raise with helpful message | Strict mode for catching mistakes |

**Default: `"keep"`** — each inner predictor maintains its own session state (see Section 6 for why).

**Detection mechanism:** A contextvar `_ACTIVE_SESSION_CONTEXT` tells inner sessions they're nested.

```python
# "keep" example (default) — each component has its own memory
agent = sessionify(SupportAgent(), recursive="predictors")
agent(question="I can't log in")
agent(question="Reset email never arrives")
# → classify has 2 turns of its own history (question → intent pairs)
# → respond has 2 turns of its own history (question+intent → reply)
# → outer session has 2 turns with CallRecords linking to inner calls

# "inherit" example — composition shares outer history
chat = sessionify(agent, nested_sessions="inherit")
chat(task="...")  
# → planner and writer see chat's history (outer conversation)
# → planner and writer do NOT record their own turns
# → chat records ONE turn with CallRecords for planner + writer
```

> **DECISION NEEDED:** Should `nested_sessions` be configurable per-child? e.g. `child_configs={"planner": {"nested_sessions": "keep"}}` for a planner that maintains its own evolving memory while the rest inherits.


### 3.3 History mode: per-predictor vs outer-conversation

When recursive sessionification wraps inner predictors, what history do they see?

This turns out to be the hardest design question. **See Section 6 for the full wire-format analysis**, but the summary is:

**`"keep"` — per-predictor history (recommended default for recursive):**
Each inner predictor accumulates its own prior input/output pairs. The classifier sees its past `question → intent` pairs. The writer sees its past `question + intent → reply` pairs. Works perfectly with any adapter — no schema mismatch.

**`"inherit"` — outer conversation history:**
All inner predictors see the parent session's conversation. The adapter filters history through each predictor's signature, which means:
- With **ChatAdapter**: works only if the outer module returns all intermediate fields (fragile)
- With **TemplateAdapter**: user controls rendering via `{"role": "history", "user": "...", "assistant": "..."}` or `{history(style='yaml')}` (robust)

The mapping:
- `"keep"` → independent per-predictor history → always schema-aligned → **safe default**
- `"inherit"` → shared outer history → powerful but needs either rich outer outputs or TemplateAdapter

> **DECISION NEEDED:** Should the default for `recursive="predictors"` be `"keep"` (safe) or `"inherit"` (simpler mental model)? See Section 6 for the tradeoffs. Current leaning: `"keep"` because it works with any adapter and produces clean per-predictor traces for optimization.


### 3.4 Call-level tracing (CallRecord)

Turns track the outer boundary. But for optimization you need per-predictor data.

```python
@dataclass
class CallRecord:
    path: str                       # "agent.planner" or "agent.researcher.summarizer"
    predictor_type: str             # "Predict", "ChainOfThought", "ReAct"
    inputs: dict[str, Any]
    outputs: dict[str, Any]
    history_snapshot: History        # the history THIS call saw
    score: float | None = None
    meta: dict = field(default_factory=dict)
```

Each `Turn` gets a `calls: list[CallRecord]` field (populated when `record="calls"` or `record="all"`).

```python
session = sessionify(agent, recursive="predictors", record="all")
session(task="...")

# Per-predictor extraction
by_path = session.to_examples(level="call", by="path")
# → {"agent.planner": [Example, ...], "agent.writer": [Example, ...]}
```

> **DECISION NEEDED:** Should `record="calls"` be the default for recursive sessions? It's cheap (just captures kwargs/outputs during the contextvar-wrapped call) but adds memory. Maybe default `"turns"` and opt-in `"all"` for dev mode.

---

## 4. Proposed API

### 4.1 `sessionify()` — extended signature

Backward-compatible. All new params have defaults that match current behavior.

```python
def sessionify(
    module: dspy.Module,
    *,
    # --- existing (unchanged) ---
    history_field: str = "history",
    max_turns: int | None = None,
    max_stored_turns: int | None = None,
    exclude_fields: set[str] | None = None,
    history_input_fields: set[str] | None = None,
    initial_history: History | None = None,
    history_policy: Literal["override", "use_if_provided", "replace_session"] = "override",
    on_metric_error: Literal["zero", "raise"] = "zero",
    strict_history_annotation: bool = False,
    copy_mode: Literal["deep", "shallow", "none"] = "deep",
    lock: Literal["none", "thread", "async"] = "none",
    on_turn: Callable | None = None,

    # --- NEW: recursive sessionification ---
    recursive: bool | Literal["predictors", "modules"] = False,
    include: set[str] | None = None,       # allowlist of predictor/module paths
    exclude: set[str] | None = None,       # blocklist of predictor/module paths (not the same as exclude_fields!)
    where: Callable[[str, dspy.Module], bool] | None = None,
    child_configs: dict[str, dict] | None = None,  # per-child overrides

    # --- NEW: nesting ---
    nested_sessions: Literal["inherit", "keep", "unwrap", "error"] = "keep",

    # --- NEW: inner state scope (only applies when nested_sessions="keep") ---
    inner_scope: Literal["cross_turn", "per_turn"] = "cross_turn",

    # --- NEW: tracing ---
    record: Literal["turns", "calls", "all"] = "turns",
    on_call: Callable | None = None,       # callback(session, call_record)
) -> Session:
    ...
```


### 4.2 `SessionContainer` — explicit composition

For when you build the orchestrator from pre-sessionified pieces.

```python
from dspy_session import SessionContainer

class SessionContainer:
    """Named collection of sessions with unified operations."""

    def __init__(self, sessions: dict[str, Session]):
        ...

    # Attribute access → session lookup
    def __getattr__(self, name: str) -> Session: ...

    # Unified operations
    def reset(self, name: str | None = None): ...
    def save(self, directory: str | Path): ...
    @classmethod
    def load(cls, directory: str | Path, modules: dict[str, dspy.Module]) -> SessionContainer: ...
    def fork(self) -> SessionContainer: ...
    def to_examples(self, *, recursive: bool = True, **kwargs) -> dict[str, list[dspy.Example]]: ...
    def score_all(self, metric: Callable): ...

    # Hot-swap
    def update(self, name: str, new_session_or_module): ...

    # Share history across specific children
    def share_history(self, names: list[str]): ...

    @property
    def named_sessions(self) -> dict[str, Session]: ...
    def __len__(self) -> int: ...  # total turns across all sessions
```

Usage:

```python
planner   = sessionify(dspy.ChainOfThought("task -> plan"), max_turns=20)
researcher = sessionify(ResearchModule(), recursive="predictors")
writer    = sessionify(dspy.Predict("plan, research -> draft"))

sessions = SessionContainer({
    "planner": planner,
    "researcher": researcher,
    "writer": writer,
})

# Use in an orchestrator
class Agent(dspy.Module):
    def __init__(self, sessions: SessionContainer):
        self.sessions = sessions

    def forward(self, task: str):
        plan = self.sessions.planner(task=task).plan
        research = self.sessions.researcher(query=plan).summary
        return self.sessions.writer(plan=plan, research=research).draft
```


### 4.3 Navigation & introspection

```python
# Tree navigation (for recursive sessions)
session.children                    # dict[str, Session] — immediate children
session.named_sessions              # dict[str, Session] — flattened tree
session.is_root                     # bool — True if no active parent context

# Turn introspection
session.turns[-1]                   # last turn
session.turns[-1].calls             # list[CallRecord] — inner predictor calls
session.turns[-1].calls[0].path     # "agent.planner"

# Pretty printing
session.turns[-1].print_trace_tree()
# ▼ [Turn 2] SupportAgent
#   Inputs: question="The reset email never arrives."
#   ├─ classify (Predict)  → intent="technical"
#   └─ respond (Predict)   → reply="Let's check your spam folder..."
#   Outputs: reply="Let's check your spam folder..."
```

> **DESIGN NOTE:** `session[i]` as shorthand for `session.turns[i]`? Slicing like `session[-2:]`? Nice for notebooks but might confuse with module indexing semantics.


### 4.4 Cascading operations

When a recursive session has children, operations cascade:

```python
session.reset()                     # resets self only (current behavior)
session.reset(cascade=True)         # resets self + all children

session.undo(steps=1, cascade=True) # undo outer turn + revert children by
                                    # the exact number of calls they made in that turn

session.save("agent_state.json")    # saves hierarchical: outer turns + child turns
Session.load_from("agent_state.json", module)  # restores everything

session.fork(cascade=True)          # deep-fork the whole tree
```


### 4.5 Dev/optimization extraction

```python
# Turn-level (existing, unchanged)
examples = session.to_examples(metric=my_metric, min_score=0.5)

# Call-level (NEW — requires record="all")
call_examples = session.to_examples(level="call")
# → flat list of Example from all CallRecords across all turns

by_predictor = session.to_examples(level="call", by="path")
# → {"agent.planner": [...], "agent.researcher.summarizer": [...]}

# Per-child session examples (for recursive sessions)
planner_examples = session.children["planner"].to_examples()

# Targeted optimization workflow
optimized_planner = dspy.BootstrapFewShot(metric=plan_quality).compile(
    session.children["planner"].module,
    trainset=by_predictor["agent.planner"],
)
session.children["planner"].update_module(optimized_planner)
# → conversation continues with optimized planner, state preserved

# Batch conversation generation
conversations = [
    [{"task": "Book a flight"}, {"task": "Change to business class"}],
    [{"task": "Find a hotel"}, {"task": "Add breakfast"}],
]
sessions = Session.batch_run(agent, conversations)
trainset = Session.merge_examples(*sessions, metric=quality, min_score=0.7)
```

---

## 5. Concrete Scenarios

### 5.1 Single predictor — baseline (already works)

```python
import dspy
from dspy_session import sessionify

dspy.configure(lm=dspy.LM("groq/moonshotai/kimi-k2-instruct-0905"))

session = sessionify(dspy.Predict("question -> answer"))
session(question="What is a derivative?")
session(question="Give me an example with f(x) = x²")
print(len(session.turns))  # 2 — no change from v1
```

### 5.2 Composed module — recursive sessionify

```python
class SupportAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.classify = dspy.Predict("question -> intent")
        self.respond = dspy.Predict("question, intent -> reply")

    def forward(self, question):
        intent = self.classify(question=question).intent
        reply = self.respond(question=question, intent=intent).reply
        return dspy.Prediction(reply=reply, intent=intent)

# Recursive: both inner predictors become session-aware
agent = sessionify(SupportAgent(), recursive="predictors", record="all")

agent(question="I can't log in")
agent(question="The reset email never arrives")

# One outer turn per call
print(len(agent.turns))  # 2

# But we can see what happened inside
print(agent.turns[1].calls)
# [CallRecord(path="classify", ...), CallRecord(path="respond", ...)]

# Extract per-predictor trainsets
by_pred = agent.to_examples(level="call", by="path")
print(by_pred.keys())  # dict_keys(["classify", "respond"])
```


### 5.3 Pre-sessionified components — explicit composition

```python
from dspy_session import sessionify, SessionContainer

planner = sessionify(dspy.ChainOfThought("task -> plan"))
writer  = sessionify(dspy.Predict("plan -> draft"))

# Each works standalone
planner(task="Build a REST API")
print(len(planner.turns))  # 1

# Now compose
sessions = SessionContainer({"planner": planner, "writer": writer})

class Orchestrator(dspy.Module):
    def __init__(self, sessions):
        self.sessions = sessions
    def forward(self, task):
        plan = self.sessions.planner(task=task).plan
        return self.sessions.writer(plan=plan).draft

# Wrap the orchestrator — inner sessions detected, "inherit" by default
app = sessionify(Orchestrator(sessions), nested_sessions="inherit")
app(task="Build a REST API")

# Only the outer session recorded a turn
print(len(app.turns))            # 1
print(len(planner.turns))        # still 1 (from standalone call — not mutated by nested call)
```


### 5.4 RAG with exclude_fields inside recursive tree

```python
class RAGAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.retriever = dspy.Predict("question -> documents")
        self.answerer = dspy.Predict("question, documents -> answer")

    def forward(self, question):
        docs = self.retriever(question=question).documents
        return self.answerer(question=question, documents=docs)

agent = sessionify(
    RAGAgent(),
    recursive="predictors",
    exclude={"retriever"},                    # don't inject history into retriever
    child_configs={
        "answerer": {"exclude_fields": {"documents"}},  # don't put docs in history
    },
)
```

---

## 6. Wire Format: What Hits the LLM

This is the crux of the whole design. Every strategy we pick for "what history do inner predictors see" ultimately boils down to: **what `messages[]` array arrives at the inference provider?**

We use `dspy-template-adapter`'s template language as our notation throughout — it gives us exact control and makes the options concrete. Even if users use `ChatAdapter`, the underlying rendering question is the same.

### The Two Scenarios

We use two scenarios throughout this section. They differ in a crucial way: how much the inner predictors' field names overlap with the outer module's interface.

#### Scenario A — SupportAgent (partial field overlap)

```python
class ClassifyIntent(dspy.Signature):
    """Classify the customer's intent."""
    question: str = dspy.InputField()
    intent: str = dspy.OutputField(desc="One of: billing, technical, general")

class GenerateReply(dspy.Signature):
    """Generate a helpful support reply."""
    question: str = dspy.InputField()
    intent: str = dspy.InputField()
    reply: str = dspy.OutputField()

class SupportAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.classify = dspy.Predict(ClassifyIntent)
        self.respond  = dspy.Predict(GenerateReply)

    def forward(self, question):
        intent = self.classify(question=question).intent
        return self.respond(question=question, intent=intent)

agent = sessionify(SupportAgent(), recursive="predictors")
```

**Outer interface:** `question → reply` (+ `intent` as intermediate)
**Field overlap:** `question` appears in both inner sigs. `intent` is the classify output AND a respond input. `reply` is only in respond.

**Prior state (1 completed turn):**
- Turn 0 inputs: `{question: "I can't log in"}`
- Turn 0 outputs: `{reply: "Try resetting your password.", intent: "technical"}`

**Current call (Turn 1):** `agent(question="The reset email never arrives")`

#### Scenario B — CorrectThenTranslate (zero field overlap between predictors)

This is the existing `decomposed-translator` example from the docs:

```python
class CorrectText(dspy.Signature):
    text: str = dspy.InputField()
    corrected: str = dspy.OutputField()

class TranslateText(dspy.Signature):
    corrected: str = dspy.InputField()
    target_language: str = dspy.InputField()
    translated: str = dspy.OutputField()

class CorrectThenTranslate(dspy.Module):
    def __init__(self, target_language: str):
        super().__init__()
        self.target_language = target_language
        self.corrector = dspy.Predict(CorrectText)
        self.translator = dspy.Predict(TranslateText)

    def forward(self, text: str):
        corrected_pred = self.corrector(text=text)
        translated_pred = self.translator(
            corrected=corrected_pred.corrected,
            target_language=self.target_language,
        )
        return dspy.Prediction(
            corrected=corrected_pred.corrected,
            translated=translated_pred.translated,
        )

session = sessionify(CorrectThenTranslate(target_language="French"), recursive="predictors")
```

**Outer interface:** `text → translated` (+ `corrected` as intermediate)
**Field overlap:** **ZERO** between the two inner predictors. Corrector uses `text → corrected`. Translator uses `corrected, target_language → translated`. The outer history has `text` and `translated`. Neither predictor's output fields match the outer output.

**Prior state (2 completed turns):**
- Turn 0: `{text: "This plant is red"}` → `{corrected: "This plant is red.", translated: "Cette plante est rouge."}`
- Turn 1: `{text: "Can I have it?"}` → `{corrected: "Can I have it?", translated: "Puis-je l'avoir ?"}`

**Outer session history** (what `session.session_history` returns):
```python
History(messages=[
    {"text": "This plant is red", "corrected": "This plant is red.", "translated": "Cette plante est rouge."},
    {"text": "Can I have it?", "corrected": "Can I have it?", "translated": "Puis-je l'avoir ?"},
])
```

> **Note:** `corrected` appears in outer history because `forward()` returns it in the `Prediction`. If it only returned `translated`, the outer history would be `{text, translated}` — making things even worse for Strategy 2.

**Current call (Turn 2):** `session(text="No it to precious, I want to keep it.")`

Inside `forward()`, two LLM calls: `corrector(text=...)` then `translator(corrected=..., target_language="French")`.

---

Each strategy below shows **both scenarios** so you can see how they handle the easy case (partial overlap) and the hard case (zero overlap).

---

### The Rendering Problem, Precisely

The outer session's history is:
```python
History(messages=[
    {"question": "I can't log in", "reply": "Try resetting your password.", "intent": "technical"}
])
```

Now watch what happens when we pass this to each adapter:

**ChatAdapter** calls `format_user_message_content(sig, msg)` + `format_assistant_message_content(sig, msg)` using the **predictor's own signature** (with history field removed). It iterates over `signature.input_fields` and `signature.output_fields` looking for matches in the history message dict.

**For `classify` (sig: `question → intent`):**
- User: looks for `question` in msg → found → `[[ ## question ## ]]\nI can't log in` ✓
- Assistant: looks for `intent` in msg → found → `[[ ## intent ## ]]\ntechnical\n\n[[ ## completed ## ]]` ✓
- **Lucky!** The outer turn happened to contain `intent` because `_extract_outputs` captured it.

**For `respond` (sig: `question, intent → reply`):**
- User: looks for `question`, `intent` → both found → `[[ ## question ## ]]\nI can't log in\n\n[[ ## intent ## ]]\ntechnical` ✓
- Assistant: looks for `reply` in msg → found → `[[ ## reply ## ]]\nTry resetting your password.\n\n[[ ## completed ## ]]` ✓
- **Also works!**

But this only works because the outer module's `_extract_outputs` captured **all** fields (`reply` + `intent`). If `forward()` returned only `reply` (common pattern), the history message would be `{question: "...", reply: "..."}` and:

**For `classify` — BROKEN:**
- User: `question` found ✓
- Assistant: `intent` NOT in history → renders as `None` → `[[ ## intent ## ]]\nNone\n\n[[ ## completed ## ]]` ❌

**This is the fundamental tension:** the outer conversation's field shape doesn't match the inner predictor's schema. Whether it works or breaks depends on accident (what the outer module returns).

**TemplateAdapter** has the same structural issue in `_expand_history`, but its `{"role": "history"}` directive with custom templates gives us an escape hatch.

---

### Strategy 1: Per-Predictor History (each predictor remembers its own calls)

This is `nested_sessions="keep"`. Each inner predictor accumulates its own turn history from its own prior calls. No schema mismatch possible.

#### Scenario A — SupportAgent

**History state after Turn 0:**
- classify's history: `[{question: "I can't log in", intent: "technical"}]`
- respond's history: `[{question: "I can't log in", intent: "technical", reply: "Try resetting..."}]`

**LLM Call 1 — classify:**

With ChatAdapter:
```
[system]  Classify the customer's intent.
          ...Inputs: question: str / Outputs: intent: str...

[user]    [[ ## question ## ]]
          I can't log in

[asst]    [[ ## intent ## ]]
          technical

          [[ ## completed ## ]]

[user]    [[ ## question ## ]]
          The reset email never arrives

          Respond with the corresponding output fields...
```

With TemplateAdapter:
```python
# Template for classify
messages=[
    {"role": "system", "content": "{instruction}"},
    {"role": "history"},                              # expands classify's own history
    {"role": "user",   "content": "{question}"},
]
```
Renders to:
```
[system]  Classify the customer's intent.

[user]    I can't log in                              ← history user (question found)
[asst]    technical                                   ← history asst (intent found)

[user]    The reset email never arrives                ← current input
```

**LLM Call 2 — respond:**

With ChatAdapter:
```
[system]  Generate a helpful support reply.
          ...Inputs: question, intent / Outputs: reply...

[user]    [[ ## question ## ]]
          I can't log in

          [[ ## intent ## ]]
          technical

[asst]    [[ ## reply ## ]]
          Try resetting your password.

          [[ ## completed ## ]]

[user]    [[ ## question ## ]]
          The reset email never arrives

          [[ ## intent ## ]]
          technical

          Respond with...
```

With TemplateAdapter:
```python
messages=[
    {"role": "system", "content": "{instruction}"},
    {"role": "history"},
    {"role": "user",   "content": "{inputs(style='yaml')}"},
]
```
Renders to:
```
[system]  Generate a helpful support reply.

[user]    question: I can't log in                    ← history user
          intent: technical
[asst]    Try resetting your password.                ← history asst (reply found)

[user]    question: The reset email never arrives      ← current input
          intent: technical
```

#### Scenario B — CorrectThenTranslate

**History state after 2 turns:**
- corrector's history: `[{text: "This plant is red", corrected: "This plant is red."}, {text: "Can I have it?", corrected: "Can I have it?"}]`
- translator's history: `[{corrected: "This plant is red.", target_language: "French", translated: "Cette plante est rouge."}, {corrected: "Can I have it?", target_language: "French", translated: "Puis-je l'avoir ?"}]`

**LLM Call 1 — corrector:**

With ChatAdapter:
```
[system]  ...Inputs: text: str / Outputs: corrected: str...

[user]    [[ ## text ## ]]
          This plant is red

[asst]    [[ ## corrected ## ]]
          This plant is red.

          [[ ## completed ## ]]

[user]    [[ ## text ## ]]
          Can I have it?

[asst]    [[ ## corrected ## ]]
          Can I have it?

          [[ ## completed ## ]]

[user]    [[ ## text ## ]]
          No it to precious, I want to keep it.

          Respond with...
```

With TemplateAdapter:
```python
messages=[
    {"role": "system", "content": "{instruction}"},
    {"role": "history"},
    {"role": "user",   "content": "{text}"},
]
```
Renders to:
```
[system]  Correct grammar and spelling in the input sentence.

[user]    This plant is red                           ← history turn 1
[asst]    This plant is red.

[user]    Can I have it?                              ← history turn 2
[asst]    Can I have it?

[user]    No it to precious, I want to keep it.       ← current input
```

✅ The corrector sees a clean history of `text → corrected` pairs. It can learn the pattern.

**LLM Call 2 — translator:**

With ChatAdapter:
```
[system]  ...Inputs: corrected, target_language / Outputs: translated...

[user]    [[ ## corrected ## ]]
          This plant is red.

          [[ ## target_language ## ]]
          French

[asst]    [[ ## translated ## ]]
          Cette plante est rouge.

          [[ ## completed ## ]]

[user]    [[ ## corrected ## ]]
          Can I have it?

          [[ ## target_language ## ]]
          French

[asst]    [[ ## translated ## ]]
          Puis-je l'avoir ?

          [[ ## completed ## ]]

[user]    [[ ## corrected ## ]]
          No, it's too precious. I want to keep it.

          [[ ## target_language ## ]]
          French

          Respond with...
```

With TemplateAdapter:
```python
messages=[
    {"role": "system", "content": "{instruction}"},
    {"role": "history"},
    {"role": "user",   "content": "{inputs(style='yaml')}"},
]
```
Renders to:
```
[system]  Translate corrected text into the target language.

[user]    corrected: This plant is red.               ← history turn 1
          target_language: French
[asst]    Cette plante est rouge.

[user]    corrected: Can I have it?                   ← history turn 2
          target_language: French
[asst]    Puis-je l'avoir ?

[user]    corrected: No, it's too precious. I want to keep it.  ← current
          target_language: French
```

✅ The translator sees a clean history of `(corrected, target_language) → translated` triples. Perfect for few-shot learning and optimization.

**Verdict:** ✅ Clean, schema-aligned, each predictor sees exactly its own prior calls. Works perfectly even with zero field overlap.
**Tradeoff:** The classifier/corrector doesn't know what the agent actually *said* to the user — it only sees its own prior calls. Fine for a classifier or a corrector, but what about a predictor that *needs* the full conversational context?

---

### Strategy 2: Outer History, Signature-Filtered (what ChatAdapter does natively)

This is `nested_sessions="inherit"` with no special projection. We pass the outer session's `History` object to each predictor, and the adapter's rendering logic filters through each predictor's signature fields.

#### Scenario A — SupportAgent

**Outer history:** `[{question: "I can't log in", reply: "Try resetting...", intent: "technical"}]`

Whether this works depends on **what the outer `_extract_outputs` captured.**

**Best case** — outer module returns all intermediate fields:

classify sees:
```
[user]    [[ ## question ## ]]                        ← input field match
          I can't log in
[asst]    [[ ## intent ## ]]                          ← output field match (intent was in outer outputs)
          technical
          [[ ## completed ## ]]
```
respond sees:
```
[user]    [[ ## question ## ]]
          I can't log in
          [[ ## intent ## ]]                          ← input field match
          technical
[asst]    [[ ## reply ## ]]                           ← output field match
          Try resetting your password.
          [[ ## completed ## ]]
```

**Worst case** — outer module returns only `reply`:

Outer history: `[{question: "I can't log in", reply: "Try resetting..."}]`

classify sees:
```
[user]    [[ ## question ## ]]
          I can't log in
[asst]    [[ ## intent ## ]]                          ← NOT in history → renders as None ❌
          None
          [[ ## completed ## ]]
```

**Scenario A verdict:** ⚠️ Fragile but can work if all intermediate fields are returned.

#### Scenario B — CorrectThenTranslate

**Outer history (best case, all fields returned):**
```python
[{text: "This plant is red", corrected: "This plant is red.", translated: "Cette plante est rouge."},
 {text: "Can I have it?",    corrected: "Can I have it?",    translated: "Puis-je l'avoir ?"}]
```

**corrector** (sig: `text → corrected`) — ChatAdapter filters through its signature:
- User: looks for `text` in msg → found ✓
- Assistant: looks for `corrected` in msg → found ✓
```
[user]    [[ ## text ## ]]
          This plant is red
[asst]    [[ ## corrected ## ]]
          This plant is red.
          [[ ## completed ## ]]

[user]    [[ ## text ## ]]
          Can I have it?
[asst]    [[ ## corrected ## ]]
          Can I have it?
          [[ ## completed ## ]]

[user]    [[ ## text ## ]]
          No it to precious, I want to keep it.
          Respond with...
```
✅ Works! But only because `forward()` returns `corrected` in the Prediction.

**translator** (sig: `corrected, target_language → translated`) — ChatAdapter filters:
- User: looks for `corrected` → found ✓, looks for `target_language` → **NOT in outer history** → missing ❌
- Assistant: looks for `translated` → found ✓
```
[user]    [[ ## corrected ## ]]
          This plant is red.

          [[ ## target_language ## ]]                 ← NOT in history! target_language
          None                                          is a module attribute, not a turn field

[asst]    [[ ## translated ## ]]
          Cette plante est rouge.
          [[ ## completed ## ]]

...same pattern for turn 2...

[user]    [[ ## corrected ## ]]
          No, it's too precious. I want to keep it.

          [[ ## target_language ## ]]
          French                                      ← current call has it, but history didn't

          Respond with...
```

❌ `target_language` is `None` in history turns! The LLM sees two prior conversations where it supposedly translated without knowing the target language. Confusing and likely to degrade quality.

**Even worse case** — outer module returns only `translated` (not `corrected`):

Outer history: `[{text: "This plant is red", translated: "Cette plante est rouge."}, ...]`

corrector sees:
```
[user]    [[ ## text ## ]]                            ← found ✓
          This plant is red
[asst]    [[ ## corrected ## ]]                       ← NOT in history ❌
          None
          [[ ## completed ## ]]
```

translator sees:
```
[user]    [[ ## corrected ## ]]                       ← NOT in history ❌
          None
          [[ ## target_language ## ]]                  ← NOT in history ❌
          None
[asst]    [[ ## translated ## ]]                      ← found ✓
          Cette plante est rouge.
          [[ ## completed ## ]]
```

💀 Almost entirely `None`. The LLM sees garbage history.

**Scenario B verdict:** ❌ Breaks badly. Even the best case has missing `target_language`. The zero-overlap nature of this pipeline makes signature-filtered outer history nearly useless.

**Overall Strategy 2 Verdict:** ⚠️ Fragile. Works only when inner predictor fields happen to appear in the outer history. Fails silently with `None` values. **Not recommended as a default.**

---

### Strategy 3: Outer History as Inline Context (not turn-based)

Instead of expanding history into user/assistant pairs, render it as text inside a message. The template adapter's `{history()}` function does exactly this.

#### Scenario A — SupportAgent

```python
# Template for classify
messages=[
    {"role": "system", "content": (
        "{instruction}\n\n"
        "Prior conversation for context:\n{history(style='yaml')}"
    )},
    {"role": "user", "content": "{question}"},
]
```

Renders to:
```
[system]  Classify the customer's intent.

          Prior conversation for context:
          - turn: 1
            question: I can't log in
            reply: Try resetting your password.
            intent: technical

[user]    The reset email never arrives
```

For respond:
```python
messages=[
    {"role": "system", "content": (
        "{instruction}\n\n"
        "Prior conversation for context:\n{history(style='yaml')}"
    )},
    {"role": "user", "content": "{inputs(style='yaml')}"},
]
```
Renders to:
```
[system]  Generate a helpful support reply.

          Prior conversation for context:
          - turn: 1
            question: I can't log in
            reply: Try resetting your password.
            intent: technical

[user]    question: The reset email never arrives
          intent: technical
```

#### Scenario B — CorrectThenTranslate

```python
# Template for corrector
messages=[
    {"role": "system", "content": (
        "{instruction}\n\n"
        "Prior conversation for context:\n{history(style='yaml')}"
    )},
    {"role": "user", "content": "{text}"},
]
```

Renders to:
```
[system]  Correct grammar and spelling in the input sentence.

          Prior conversation for context:
          - turn: 1
            text: This plant is red
            corrected: This plant is red.
            translated: Cette plante est rouge.
          - turn: 2
            text: Can I have it?
            corrected: Can I have it?
            translated: Puis-je l'avoir ?

[user]    No it to precious, I want to keep it.
```

✅ The corrector sees all prior context. It can see what the user originally wrote, what was corrected, and what translation resulted. Everything is visible.

For translator:
```python
messages=[
    {"role": "system", "content": (
        "{instruction}\n\n"
        "Prior conversation for context:\n{history(style='yaml')}"
    )},
    {"role": "user", "content": "{inputs(style='yaml')}"},
]
```
Renders to:
```
[system]  Translate corrected text into the target language.

          Prior conversation for context:
          - turn: 1
            text: This plant is red
            corrected: This plant is red.
            translated: Cette plante est rouge.
          - turn: 2
            text: Can I have it?
            corrected: Can I have it?
            translated: Puis-je l'avoir ?

[user]    corrected: No, it's too precious. I want to keep it.
          target_language: French
```

✅ The translator sees the full conversation, including what the user originally typed (`text`), the corrections, and its own prior translations. Tone-consistent translation is easier with this context.

**Verdict:** ✅ Never breaks. All fields visible to all predictors regardless of schema mismatch. But loses native multi-turn structure — the LLM doesn't see it as a "conversation" it's continuing, just as background text. Less effective for models tuned on multi-turn chat.

---

### Strategy 4: Outer History with Custom Field Mapping via `{"role": "history"}` Directive

The template adapter's history directive lets you control exactly which fields map to user/assistant, regardless of the predictor's signature:

#### Scenario A — SupportAgent

```python
# Template for classify — use the OUTER module's field names
messages=[
    {"role": "system", "content": "{instruction}"},
    {"role": "history", "user": "{question}", "assistant": "{reply}"},
    {"role": "user",   "content": "{question}"},
]
```

Renders to:
```
[system]  Classify the customer's intent.

[user]    I can't log in                              ← {question} from history entry
[asst]    Try resetting your password.                ← {reply} from history entry

[user]    The reset email never arrives                ← current input
```

The classifier sees the conversation as a proper multi-turn exchange. It doesn't matter that `reply` isn't in the classifier's signature — the template accesses history entry keys directly.

For respond:
```python
messages=[
    {"role": "system", "content": "{instruction}"},
    {"role": "history", "user": "{question}", "assistant": "{reply}"},
    {"role": "user",   "content": "{inputs(style='yaml')}"},
]
```
Renders to:
```
[system]  Generate a helpful support reply.

[user]    I can't log in                              ← question from history
[asst]    Try resetting your password.                ← reply from history

[user]    question: The reset email never arrives
          intent: technical
```

✅ Both predictors see a clean user/assistant conversation. The mapping is explicit.

#### Scenario B — CorrectThenTranslate

Here the choice of what maps to "user" and "assistant" is more interesting. The outer conversation is `text → translated`:

```python
# Template for corrector
messages=[
    {"role": "system", "content": "{instruction}"},
    {"role": "history", "user": "{text}", "assistant": "{translated}"},
    {"role": "user",   "content": "{text}"},
]
```

Renders to:
```
[system]  Correct grammar and spelling in the input sentence.

[user]    This plant is red                           ← user's original text
[asst]    Cette plante est rouge.                     ← the French translation (!)

[user]    Can I have it?
[asst]    Puis-je l'avoir ?

[user]    No it to precious, I want to keep it.       ← current input
```

⚠️ The corrector sees the **French translation** as the "assistant response" — but the corrector's job is English correction, not translation. The LLM might get confused: "am I supposed to output French?" The mapping `{text} → {translated}` doesn't semantically match the corrector's role.

We could use `{corrected}` instead:
```python
{"role": "history", "user": "{text}", "assistant": "{corrected}"}
```
Renders to:
```
[user]    This plant is red
[asst]    This plant is red.                          ← corrected English ✓

[user]    Can I have it?
[asst]    Can I have it?                              ← corrected English ✓

[user]    No it to precious, I want to keep it.
```

✅ Now the corrector sees semantically correct history — its own corrections. But this requires `corrected` to be in the outer history (i.e., `forward()` must return it).

For translator:
```python
messages=[
    {"role": "system", "content": "{instruction}"},
    {"role": "history", "user": "{text}", "assistant": "{translated}"},
    {"role": "user",   "content": "{inputs(style='yaml')}"},
]
```
Renders to:
```
[system]  Translate corrected text into the target language.

[user]    This plant is red                           ← user's original text
[asst]    Cette plante est rouge.                     ← French translation ✓

[user]    Can I have it?
[asst]    Puis-je l'avoir ?                           ← French translation ✓

[user]    corrected: No, it's too precious. I want to keep it.  ← current
          target_language: French
```

✅ The translator sees a proper translation history. The user/assistant structure matches what it does.

**Scenario B verdict:** ⚠️ Works, but **each predictor needs a different field mapping** to be semantically correct. The corrector wants `{text} → {corrected}`, the translator wants `{text} → {translated}` (or `{corrected} → {translated}`). This means the mapping should be a per-child config, not a global one.

**Overall Strategy 4 Verdict:** ✅ Proper multi-turn structure. No schema mismatch — we explicitly choose what maps to user/assistant. But requires per-predictor configuration of the mapping, and TemplateAdapter. Powerful escape hatch, not a default.

---

### Strategy 5: Hybrid — Outer Context + Per-Predictor Turns

The most information-rich option. Each predictor gets:
1. The outer conversation as inline context (for awareness)
2. Its own prior calls as proper user/assistant turns (for schema-aligned continuation)

This requires two history sources — the template adapter's `{outer_history()}` custom helper for the context, and `{"role": "history"}` for the predictor's own turns.

#### Scenario A — SupportAgent

```python
# Template for classify
messages=[
    {"role": "system", "content": (
        "{instruction}\n\n"
        "Full conversation context:\n{outer_history(style='yaml')}"  # custom helper
    )},
    {"role": "history"},                              # predictor's own prior calls
    {"role": "user", "content": "{question}"},
]
```

Renders to (Turn 1, classify has 1 prior call + outer context):
```
[system]  Classify the customer's intent.

          Full conversation context:
          - turn: 1
            question: I can't log in
            reply: Try resetting your password.
            intent: technical

[user]    I can't log in                              ← classify's own history
[asst]    technical                                   ← classify's own output

[user]    The reset email never arrives                ← current input
```

The classifier has full situational awareness (it knows the agent said "Try resetting your password") AND clean schema-aligned turns for its own task.

#### Scenario B — CorrectThenTranslate

```python
# Template for corrector
messages=[
    {"role": "system", "content": (
        "{instruction}\n\n"
        "Full conversation context:\n{outer_history(style='yaml')}"
    )},
    {"role": "history"},                              # corrector's own prior calls
    {"role": "user", "content": "{text}"},
]
```

Renders to (Turn 2, corrector has 2 prior calls + outer context):
```
[system]  Correct grammar and spelling in the input sentence.

          Full conversation context:
          - turn: 1
            text: This plant is red
            corrected: This plant is red.
            translated: Cette plante est rouge.
          - turn: 2
            text: Can I have it?
            corrected: Can I have it?
            translated: Puis-je l'avoir ?

[user]    This plant is red                           ← corrector's own turn 1
[asst]    This plant is red.

[user]    Can I have it?                              ← corrector's own turn 2
[asst]    Can I have it?

[user]    No it to precious, I want to keep it.       ← current input
```

The corrector knows the full pipeline context (the user is having a conversation that gets translated to French), AND has clean `text → corrected` examples for its specific task.

For translator:
```python
messages=[
    {"role": "system", "content": (
        "{instruction}\n\n"
        "Full conversation context:\n{outer_history(style='yaml')}"
    )},
    {"role": "history"},
    {"role": "user", "content": "{inputs(style='yaml')}"},
]
```
Renders to:
```
[system]  Translate corrected text into the target language.

          Full conversation context:
          - turn: 1
            text: This plant is red
            corrected: This plant is red.
            translated: Cette plante est rouge.
          - turn: 2
            text: Can I have it?
            corrected: Can I have it?
            translated: Puis-je l'avoir ?

[user]    corrected: This plant is red.               ← translator's own turn 1
          target_language: French
[asst]    Cette plante est rouge.

[user]    corrected: Can I have it?                   ← translator's own turn 2
          target_language: French
[asst]    Puis-je l'avoir ?

[user]    corrected: No, it's too precious. I want to keep it.  ← current
          target_language: French
```

✅ The translator can see the user's original text ("No it to precious...") in the context for tone awareness, AND has clean `(corrected, target_language) → translated` examples for consistency.

**Verdict:** ✅ Maximum context + maximum schema alignment. Best of both worlds. But uses the most tokens and adds complexity (two history sources, requires both `outer_history` helper and per-predictor state). Worthwhile for complex agents where inner predictors genuinely benefit from knowing the full conversation.

---

### Strategy Comparison Matrix

| | Schema safe | Multi-turn structure | Full context visible | Token cost | ChatAdapter | TemplateAdapter | Scenario A | Scenario B |
|---|---|---|---|---|---|---|---|---|
| **1: Per-predictor** | ✅ always | ✅ native turns | ❌ own calls only | ✅ minimal | ✅ | ✅ | ✅ | ✅ |
| **2: Outer, sig-filtered** | ❌ fragile | ✅ native turns | ⚠️ depends on fields | ✅ | ✅ | ✅ | ⚠️ lucky | 💀 broken |
| **3: Outer inline text** | ✅ always | ❌ flat text | ✅ everything | ⚠️ verbose | ❌ needs template | ✅ | ✅ | ✅ |
| **4: Outer field mapping** | ✅ explicit | ✅ native turns | ✅ outer conversation | ✅ | ❌ needs template | ✅ | ✅ | ⚠️ per-child config |
| **5: Hybrid** | ✅ both | ✅ both | ✅ everything | ❌ most tokens | ❌ needs template | ✅ | ✅ | ✅ |

**Scenario B is the litmus test.** The CorrectThenTranslate pipeline has zero field overlap between inner predictors, which is common in real decomposed modules. Strategy 2 fails catastrophically here. Strategy 1 handles it effortlessly.

---

### What This Means for `dspy-session`'s Design

The two scenarios reveal a clear hierarchy:

**Strategy 1 (per-predictor) is the only strategy that works universally.** It handles both the easy case (SupportAgent with partial field overlap) and the hard case (CorrectThenTranslate with zero overlap) without any configuration, any special adapter, or any hope that the outer module returns the right fields.

**Strategy 2 (outer, sig-filtered) is an attractive nuisance.** It sometimes works (Scenario A with rich outputs) and catastrophically fails (Scenario B), with the failure mode being silent `None` values in history. We should not default to this.

**Strategies 3-5 are TemplateAdapter power tools.** They solve the rendering problem through explicit template control rather than implicit signature matching. They're available to users who need them, but can't be the default because they require TemplateAdapter.

The strategies split cleanly along two axes:

**Axis 1: Where does history state live?**
- **Per-predictor** (Strategy 1) = `nested_sessions="keep"`
- **Outer session** (Strategies 2-5) = `nested_sessions="inherit"`

**Axis 2: How is history rendered for inner predictors?**

This is **the adapter's job, not the session's job.** The session's responsibility is:
1. Decide what `History` object to pass to each predictor
2. Let the adapter render it

For `"keep"`: each predictor gets its own `History` built from its own prior calls. The adapter renders it natively through the predictor's own signature. Always works. ✅

For `"inherit"`: the session passes the outer `History` to inner predictors. How it renders depends on which adapter is active:
- **ChatAdapter**: Strategy 2 (signature-filtered). Breaks on zero-overlap pipelines like CorrectThenTranslate. ❌
- **TemplateAdapter**: Strategies 3, 4, or 5 — user controls rendering via their template. ✅

**The practical implication:**

1. **Default: `"keep"` (per-predictor history).** Works with any adapter, any module structure. Each predictor's history is always schema-aligned. Great for optimization (clean per-predictor traces). No configuration needed.

2. **Opt-in: `"inherit"` for users who know what they're doing.** Best paired with TemplateAdapter for full control. For ChatAdapter users choosing `"inherit"`, we should warn/document that it only works if the outer module returns all intermediate fields.

3. **Power user: Strategy 5 (hybrid) via TemplateAdapter integration.** For users who want inner predictors to see BOTH the outer conversation AND their own schema-aligned turns. `dspy-session` exposes `outer_history` as a template helper:
   ```python
   adapter.register_helper("outer_history", lambda ctx, sig, demos, **kw:
       TemplateAdapter._render_history_inline(ctx.get("__outer_history__"), sig, **kw))
   ```

> **KEY INSIGHT FROM SCENARIO B:** The CorrectThenTranslate pipeline (zero field overlap) is not a corner case — it's the **common case** for well-decomposed modules. A corrector and a translator *should* have different signatures. The API must handle this gracefully by default, which means per-predictor history.

---

### 6.1 Recommended Default: `nested_sessions="keep"` = Strategy 1

Each predictor maintains its own call history. No schema mismatch. Works with any adapter.

```python
# This is what you get by default with recursive sessionification
agent = sessionify(SupportAgent(), recursive="predictors")
# equivalent to:
agent = sessionify(SupportAgent(), recursive="predictors", nested_sessions="keep")
```

See Strategy 1 above for the exact wire format for both scenarios.

### 6.2 Opt-in: `nested_sessions="inherit"` + TemplateAdapter = Strategies 3-5

For users who want inner predictors to see the outer conversation. Pair with TemplateAdapter for safe, explicit rendering control.

```python
agent = sessionify(SupportAgent(), recursive="predictors", nested_sessions="inherit")
```

- With **ChatAdapter**: you get Strategy 2 (signature-filtered). Works for Scenario A if outer outputs are rich. **Breaks for Scenario B.** Use at your own risk.
- With **TemplateAdapter**: you get Strategies 3, 4, or 5 depending on your template. Full control, always safe.

### 6.3 Summary: The Strategy Selection Flowchart

```
Do inner predictors need to see the outer conversation?
├── No → nested_sessions="keep" (Strategy 1) ← DEFAULT
│         Works always. Schema-aligned. Optimizer-friendly.
│
└── Yes → nested_sessions="inherit"
          │
          ├── Using ChatAdapter?
          │   └── Ensure forward() returns ALL intermediate fields
          │       ⚠️ Fragile. Test with zero-overlap pipelines.
          │
          └── Using TemplateAdapter?
              ├── Want flat context? → {history(style='yaml')} in system msg (Strategy 3)
              ├── Want multi-turn?  → {"role": "history", "user": "...", "assistant": "..."} (Strategy 4)
              └── Want both?        → {outer_history()} + {"role": "history"} (Strategy 5)
```

---

## 6.4 State Scope: Within a Forward Pass vs Across Them

Strategy 1 (`"keep"`) says each inner predictor maintains its own session. But **when does that session accumulate state?** This question has three possible answers, and they produce very different behaviors.

### The Three Scopes

| Scope | Inner predictor state lives... | When it resets |
|---|---|---|
| **A: Cross-turn** | Across all outer `forward()` calls for the session's lifetime | Only on explicit `reset()` |
| **B: Per-outer-turn** | Only within one outer `forward()` call | Auto-resets before each outer turn |
| **C: Hybrid** | Across outer turns, but with outer-turn boundaries tracked | Never auto-resets, but `undo(cascade=True)` knows the boundaries |

### Scenario A — SupportAgent (1 call per predictor per turn)

The outer `forward()` calls `classify` once and `respond` once per turn.

**Scope A (cross-turn):** After 3 outer turns, classify has 3 turns of history. Each new classification sees all prior classifications.

```
Outer Turn 0:  classify("I can't log in")           → classify has 0 prior turns
Outer Turn 1:  classify("Reset email never arrives") → classify sees turn 0 (its own prior)
Outer Turn 2:  classify("It's in my spam folder")    → classify sees turns 0 + 1
```

The classifier builds up a thread of prior classifications. On Turn 2, it sees:
```
[user] I can't log in          →  [asst] technical
[user] Reset email never arrives →  [asst] technical
[user] It's in my spam folder   →  current input
```

✅ The classifier can see the pattern ("this has been technical all along") and classify consistently.

**Scope B (per-outer-turn):** The classifier NEVER sees history (it's called once per outer turn, and state resets between turns).

```
Outer Turn 0:  classify("I can't log in")           → classify has 0 prior turns
Outer Turn 1:  classify("Reset email never arrives") → classify has 0 prior turns (reset!)
Outer Turn 2:  classify("It's in my spam folder")    → classify has 0 prior turns (reset!)
```

❌ The classifier treats every message in isolation. No consistency benefit. **This defeats the purpose of recursive sessionification.** If you wanted stateless inner predictors, you wouldn't sessionify them.

### Scenario B — CorrectThenTranslate (1 call per predictor per turn)

Same dynamic. The corrector is called once per outer turn.

**Scope A (cross-turn):** After 3 outer turns, the corrector has seen 3 texts. On Turn 2:

```
[user] This plant is red               →  [asst] This plant is red.
[user] Can I have it?                   →  [asst] Can I have it?
[user] No it to precious, I want to keep it.  →  current input
```

✅ The corrector sees the conversational thread. It can maintain consistent style/tone in corrections.

The translator on Turn 2:

```
[user] corrected: This plant is red.  / target_language: French   →  [asst] Cette plante est rouge.
[user] corrected: Can I have it?      / target_language: French   →  [asst] Puis-je l'avoir ?
[user] corrected: No, it's too precious...  / target_language: French   →  current input
```

✅ The translator sees its prior translations. It can maintain consistent terminology ("plant" → "plante" throughout). This is exactly why multi-turn translation is better than turn-by-turn.

**Scope B (per-outer-turn):** Both predictors are stateless every turn. No consistency. ❌

### Scenario C — Agent with Retry Loop (N calls per predictor per turn)

This is where within-turn state matters:

```python
class RetryAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.planner = dspy.ChainOfThought("task -> plan")
        self.validator = dspy.Predict("plan -> is_valid: bool, feedback")

    def forward(self, task):
        for attempt in range(3):
            plan = self.planner(task=task)
            result = self.validator(plan=plan.plan)
            if result.is_valid:
                return plan
            task = f"{task}\n\nPrevious attempt was invalid: {result.feedback}"
        return plan  # best effort
```

The planner is called up to 3 times **within a single outer forward()**.

**Scope A (cross-turn):** The planner accumulates ALL calls — both within-turn retries AND cross-turn. After 2 outer turns where each needed 2 retries:

```
Planner internal state: 4 turns total
  Turn 0: (from outer turn 0, attempt 1)
  Turn 1: (from outer turn 0, attempt 2) ← sees attempt 1's failure
  Turn 2: (from outer turn 1, attempt 1) ← sees both attempts from outer turn 0
  Turn 3: (from outer turn 1, attempt 2) ← sees everything
```

The planner on outer Turn 1, attempt 2 sees 3 prior plans — including the retry from a completely different task. Is that helpful? Maybe. The planner learns "I tend to make plans that fail validation" which could be useful. But it also sees stale context from an unrelated task.

**Scope B (per-outer-turn):** The planner sees only within-turn retries:

```
Outer Turn 0:
  Attempt 1: planner has 0 prior turns
  Attempt 2: planner has 1 prior turn (sees attempt 1's failure) ✅

Outer Turn 1:
  Attempt 1: planner has 0 prior turns (reset!)
  Attempt 2: planner has 1 prior turn (sees attempt 1's failure) ✅
```

✅ Retries benefit from seeing prior attempts. ❌ No cross-turn learning.

**Scope C (hybrid):** Same as Scope A (all calls accumulate), but outer turn boundaries are tracked:

```
Planner internal state: 4 turns total
  [outer_turn=0] Turn 0: attempt 1
  [outer_turn=0] Turn 1: attempt 2
  [outer_turn=1] Turn 2: attempt 1
  [outer_turn=1] Turn 3: attempt 2
```

Now `undo(cascade=True)` on the outer session knows: "outer Turn 1 produced planner turns 2 and 3 → remove those."

### Scenario D — ReAct-like Multi-Step (many calls, state is essential)

```python
class StepAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.thinker = dspy.Predict("task, observation -> thought, action")

    def forward(self, task):
        observation = "No observation yet."
        for step in range(5):
            result = self.thinker(task=task, observation=observation)
            observation = execute_action(result.action)
            if is_done(observation):
                return result
        return result
```

The thinker is called 5 times per outer turn. **Within-turn state accumulation is essential** — step 3 must see steps 0-2.

**Scope B works perfectly here:** thinker accumulates within the turn, resets between turns. Each outer turn is a fresh reasoning chain.

**Scope A also works** but the thinker sees stale reasoning chains from prior outer turns, which might confuse it.

### The Design Answer

| Scope | Within-turn | Cross-turn | Best for |
|---|---|---|---|
| **A: Cross-turn** | ✅ accumulates | ✅ accumulates | Simple pipelines (1 call/predictor/turn): corrector, translator, classifier |
| **B: Per-outer-turn** | ✅ accumulates | ❌ resets | Multi-step agents (ReAct, retry loops): each turn is a fresh episode |
| **C: Hybrid** | ✅ accumulates | ✅ accumulates + tracked | Any pattern. Supports `undo(cascade=True)`. |

**Scope C (hybrid) is the right default.** It gives you cross-turn persistence (the whole point of multi-turn) AND within-turn accumulation (essential for multi-step). The outer turn boundaries are tracked so cascading operations work.

But Scope B should be easy to opt into for the multi-step/ReAct case:

```python
agent = sessionify(
    StepAgent(),
    recursive="predictors",
    child_configs={
        "thinker": {"scope": "per_turn"},  # reset between outer turns
    },
)
```

Or as a shorthand:

```python
agent = sessionify(StepAgent(), recursive="predictors", inner_scope="per_turn")
```

### Implementation: How Hybrid Scope Works

Each inner Session gets a `_turn_groups: list[tuple[int, int]]` that maps outer turn index → range of inner turn indices:

```python
@dataclass
class TurnGroup:
    outer_turn_index: int
    inner_turn_start: int  # inclusive
    inner_turn_end: int    # exclusive
```

When the outer Session starts a forward pass, it notifies children of the current outer turn index (via the `SessionContext` contextvar). When the outer turn completes, each child records: "outer turn N produced inner turns [start, end)."

**`undo(steps=1, cascade=True)`** on the outer session:
1. Pops the last outer turn
2. For each child: looks up the `TurnGroup` for that outer turn → removes exactly those inner turns

**`save()` / `load()`** on the outer session:
1. Saves outer turns + each child's turns + turn group mappings
2. On load, restores everything with correct associations

**`fork(cascade=True)`:**
1. Deep-copies outer turns + each child's full state including turn groups

### Wire Format Implications

Scope C doesn't change the wire format at all — it's purely a state management concern. The wire format is still determined by Strategy 1/2/3/4/5 from the previous section.

What changes is **how much history** the inner predictor sees on a given call:

```
Outer Turn 2, corrector call:

Scope A/C (cross-turn):
  [user] This plant is red          → [asst] This plant is red.       (from outer turn 0)
  [user] Can I have it?             → [asst] Can I have it?           (from outer turn 1)
  [user] No it to precious...       → current input                   (outer turn 2)

Scope B (per-outer-turn):
  [user] No it to precious...       → current input                   (only current turn)
```

### Summary

> **Default: Scope C (hybrid).** Inner predictors accumulate state across all calls (within-turn AND cross-turn). Outer turn boundaries are tracked for cascading undo/save/fork. Users who want per-turn scoping can set `inner_scope="per_turn"` globally or per-child.

> **DECISION NEEDED:** Should `inner_scope` be a top-level param or only available via `child_configs`?
<!-- [ ] Top-level param (simpler for the common case) -->
<!-- [ ] child_configs only (keeps the API surface smaller) -->
<!-- [ ] Both (top-level sets the default, child_configs overrides per-child) -->

---

## 7. Optimizer Time-Travel

The hardest problem. When an optimizer replays Turn 3, all inner sub-sessions must be at their exact state from Turn 3 — without corrupting production state.

### How it works today (single level)

`history_policy="override"` + explicit history → stateless pass-through, no turn recorded. This already works.

### How it must work for recursive sessions

When the optimizer calls `session(question="...", history=snapshot_from_turn_3)`:

1. The root `Session` detects explicit history → enters override/replay mode
2. For `"inherit"` children: they receive the historical snapshot as their contextvar history. No state mutation. ✓ Already works.
3. For `"keep"` children: **problem** — they have their own accumulated turns. We need to time-travel them too.

**Solution:** Each outer `Turn` stores `child_snapshots: dict[str, History]` — a snapshot of every child session's history at that point in time.

```python
@dataclass
class Turn:
    index: int
    inputs: dict[str, Any]
    outputs: dict[str, Any]
    history_snapshot: History
    score: float | None = None
    calls: list[CallRecord] | None = None
    # NEW: for recursive sessions with nested_sessions="keep"
    child_snapshots: dict[str, History] | None = None
```

During optimizer replay:
1. Root session enters a thread-safe `_time_travel` context
2. Children temporarily load their `child_snapshots` history
3. Children are forced into `history_policy="override"` (don't record)
4. Optimizer gets deterministic replay
5. Context exits → children revert to real state

```python
# This "just works" — no special user code needed
trainset = session.to_examples()  # snapshots are embedded

optimized = dspy.BootstrapFewShot(metric=quality).compile(
    session,  # Session is a dspy.Module, so optimizers can use it directly
    trainset=trainset,
)
```

> **DECISION NEEDED:** Is the complexity of `child_snapshots` worth it? Only needed for `nested_sessions="keep"`. For `"inherit"`, the parent's history snapshot is sufficient. Maybe we defer `"keep"` time-travel to a later version?

---

## 8. Implementation Skeleton

### New contextvar for nesting detection

```python
_ACTIVE_SESSION_CONTEXT: contextvars.ContextVar[SessionContext | None] = contextvars.ContextVar(
    "dspy_session_context", default=None
)

@dataclass
class SessionContext:
    """Tracks the active session call stack."""
    session: Session
    history: History
    depth: int
    call_records: list[CallRecord]  # mutable, appended to during forward
```

### Modified `_forward_impl` (sketch)

```python
def _forward_impl(self, **kwargs):
    parent_ctx = _ACTIVE_SESSION_CONTEXT.get()

    # Am I nested inside another session?
    if parent_ctx is not None:
        return self._handle_nested_call(parent_ctx, kwargs)

    # I'm the root session — normal turn logic
    explicit_history = kwargs.pop(self.history_field, None)
    # ... existing policy handling ...

    ctx = SessionContext(
        session=self,
        history=run_history,
        depth=0,
        call_records=[],
    )
    token = _ACTIVE_SESSION_CONTEXT.set(ctx)
    try:
        result = self._invoke_inner(run_history, kwargs)
    finally:
        _ACTIVE_SESSION_CONTEXT.reset(token)

    # Record turn with call records
    self._record_turn(kwargs, result, run_history, calls=ctx.call_records)
    return result

def _handle_nested_call(self, parent_ctx, kwargs):
    policy = self._nested_sessions_policy  # or parent's config

    if policy == "inherit":
        # Don't record own turn, use parent's history, trace the call
        result = self._invoke_with_history(parent_ctx.history, kwargs)
        parent_ctx.call_records.append(CallRecord(
            path=self._session_path,
            predictor_type=type(self.module).__name__,
            inputs=kwargs, outputs=self._extract_outputs(result),
            history_snapshot=parent_ctx.history,
        ))
        return result

    elif policy == "keep":
        # Record own turn, use own history
        return self._forward_as_standalone(**kwargs)

    elif policy == "unwrap":
        # Bypass session, call module directly
        return self.module(**kwargs)

    elif policy == "error":
        raise RuntimeError(
            f"Session '{self._session_path}' called inside active session "
            f"'{parent_ctx.session._session_path}'. Set nested_sessions to "
            f"'inherit', 'keep', or 'unwrap' to allow this."
        )
```

### Recursive predictor wrapping

```python
def _prepare_recursive(self):
    """Walk module tree and wrap predictors/submodules as child sessions."""
    if self._recursive == "predictors":
        for path, predictor in self.module.named_predictors():
            if self._should_include(path):
                child_config = self._child_configs.get(path, {})
                child_session = Session(predictor, **{**self._default_child_config, **child_config})
                self._children[path] = child_session
                # Replace attribute on parent module
                self._set_nested_attr(self.module, path, child_session)

    elif self._recursive == "modules":
        for name, submodule in self._walk_modules(self.module):
            if self._should_include(name):
                child_config = self._child_configs.get(name, {})
                child_session = Session(submodule, **{**self._default_child_config, **child_config})
                self._children[name] = child_session
                setattr(self.module, name, child_session)
```

---

## 9. Open Questions

Mark your preference inline or add notes.

### Q1: Should `recursive=True` be an alias for `"predictors"`?
<!-- [ ] Yes, True → "predictors" (simpler for users) -->
<!-- [ ] No, keep it strictly as the literal strings -->

### Q2: Default `record` mode for recursive sessions?
<!-- [ ] "turns" always (opt-in "all") — less memory overhead -->
<!-- [ ] "all" when recursive is enabled (you probably want call traces) -->

### Q3: `nested_sessions` per-child or global only?
<!-- [ ] Global only (simpler) -->
<!-- [ ] Per-child via child_configs (more flexible) -->

### Q4: Default nesting policy for recursive sessions?
See Section 6 for the full wire-format analysis.
<!-- [ ] "keep" (per-predictor history) — safe, schema-aligned, works with any adapter -->
<!-- [ ] "inherit" (outer history) — simpler mental model, but fragile with ChatAdapter -->
<!-- [ ] "keep" as default, but make "inherit" easy to opt into -->

### Q5: Inner state scope — where should the `inner_scope` param live?
See Section 6.4. Default is `"cross_turn"` (hybrid with tracking). Alternative is `"per_turn"` (reset between outer turns).
<!-- [ ] Top-level sessionify param (simpler for common case: all children same scope) -->
<!-- [ ] child_configs only (keeps API smaller, per-child control) -->
<!-- [ ] Both: top-level sets default, child_configs overrides per-child -->

### Q6: Should `SessionContainer` be a `dspy.Module`?
If yes, optimizers can discover its contents. If no, it's just a dict-like convenience.
<!-- [ ] Yes, subclass dspy.Module -->
<!-- [ ] No, keep it a plain utility class -->

### Q7: `session.replay()` — should it exist in v2?
```python
results = session.replay(optimized_module)  # re-run same inputs, new module
```
<!-- [ ] Yes, add it now -->
<!-- [ ] Defer to v3 -->

### Q8: Context manager form?
```python
async with sessionify(agent, lock="async") as chat:
    await chat.acall(user="Hello")
    # auto-finalize on exit
```
<!-- [ ] Yes, add __aenter__/__aexit__ -->
<!-- [ ] Not needed, explicit .save() is fine -->

### Q9: Decorator form for classes?
```python
@sessionify
class MyAgent(dspy.Module):
    ...
agent = MyAgent()  # already a Session
```
<!-- [ ] Yes, nice for module authors -->
<!-- [ ] No, confuses class vs instance semantics -->

### Q10: Bounded memory beyond max_turns?
```python
sessionify(module, history_budget=8000, compressor=my_summarizer)
```
<!-- [ ] Yes, important for production -->
<!-- [ ] Defer, max_turns is sufficient for now -->

---

## 10. Migration & Rollout

### Phase 1: Foundation (non-breaking)
- Add `recursive` param (default `False`)
- Add `nested_sessions` param (default `"inherit"`)
- Add `_ACTIVE_SESSION_CONTEXT` contextvar for nesting detection
- Add `children` property and `child_configs`
- Ship `CallRecord` dataclass and `record` param
- All existing tests pass unchanged

### Phase 2: SessionContainer
- `SessionContainer` class with unified operations
- `save`/`load` for hierarchical state
- `to_examples(level="call", by="path")`

### Phase 3: Polish
- `cascade=True` for reset/undo/fork
- `session.replay()`
- `Turn.print_trace_tree()`
- Notebook `_repr_html_`
- `batch_run`

### Phase 4: Evaluate changing defaults
- Consider making `recursive="predictors"` the default
- Consider making `record="all"` the default for recursive sessions
- These would be the v2 breaking changes (if any)

---

## Scratch Space

_Use this area for quick experiments, alternative ideas, or code you want to test._

```python
# Test: does the current contextvar approach already handle nested Session calls?
# (spoiler: it doesn't — inner sessions will try to record their own turns)
```

```python
# Test: what does named_predictors() return for a module containing a Session?
# (important for understanding how optimizers see the tree)
```
