# dspy-session API Usage Examples (one example per parameter/API)

This page is a practical cookbook for every public parameter and API in `dspy-session`.
Each snippet uses real DSPy signatures and modules that reflect the scenario being described.

---

## Shared setup used in snippets

Every snippet below assumes this setup block has been run first. It configures an LLM and imports everything you need.

`dspy.LM("openai/gpt-4o-mini")` creates a lightweight language model client. Under the hood, DSPy uses [LiteLLM](https://docs.litellm.ai/) as a universal routing layer, which means the string you pass follows the format `"provider/model-name"`. For example, `"openai/gpt-4o-mini"`, `"anthropic/claude-3-haiku-20240307"`, or `"together_ai/meta-llama/Llama-3-8b-chat-hf"`. LiteLLM translates these into the correct API calls for each provider, so you only need to set the appropriate API key as an environment variable (e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`) and swap the model string — nothing else in your code changes.

We use `gpt-4o-mini` here because it is cheap and fast, making it a good default for running examples and experiments. If you want to use a different model, just change the string. If you don't have an API key at all and want to run these examples purely locally, you can point to a local model via Ollama (e.g., `"ollama/llama3"`) or any other LiteLLM-compatible provider.

```python
import asyncio
import dspy
from dspy.adapters.types.history import History

from dspy_session import (
    Session,
    SessionState,
    Turn,
    sessionify,
    with_memory,
    get_current_history,
    get_outer_history,
    get_node_memory,
    get_child_l1_ledger,
    get_execution_trace,
)

# Configure a cheap LLM — swap this for any provider/model you prefer.
dspy.configure(lm=dspy.LM("groq/moonshotai/kimi-k2-instruct-0905"))
```

---

## Factory APIs

### `sessionify(module, **kwargs)`

This is the quickest way to give memory to a single DSPy module. Think of it as wrapping your predictor in a thin shell that records every question and answer, then feeds the conversation back on the next call.

**Realistic scenario:** You built a `dspy.ChainOfThought` that answers customer questions. Right now every call is stateless — the model forgets what the customer said 10 seconds ago. Wrapping it with `sessionify` instantly gives it multi-turn memory. The customer asks "What's your return policy?", then follows up with "And how long does the refund take?" — the model now knows what "the refund" refers to.

**When not to use:** If your program is a complex pipeline with multiple sub-modules (planner, researcher, writer), `sessionify` only gives memory to the top-level module. The sub-modules remain stateless. Use `with_memory` instead for that.

```python
class CustomerSupport(dspy.Signature):
    """Answer customer questions about our products and policies."""
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

support_bot = sessionify(dspy.ChainOfThought(CustomerSupport))
support_bot(question="What's your return policy?")
support_bot(question="How long does the refund take?")
# The model remembers the previous question — it knows "the refund" refers to returns
print(len(support_bot.turns))  # 2
```

### `with_memory(module, ...)`

This is designed for complex DSPy programs that contain multiple nested modules. It walks your module tree and can wrap each sub-component with its own independent conversation ledger, each with its own memory policy.

**Realistic scenario:** You have a research agent with three sub-modules: a `planner` that decides what to investigate, a `researcher` that runs web searches, and a `writer` that drafts the final answer. Each one needs to remember what it has already done. The planner needs to remember its past plans so it doesn't repeat itself. The researcher needs to know which queries it has already tried. The writer needs to see its previous drafts to improve. `with_memory(agent, recursive=True)` gives each of them their own memory in one call.

**When not to use:** If you just have a single `dspy.Predict` or `dspy.ChainOfThought` with no sub-modules, this is overkill. Use `sessionify` instead.

```python
class Plan(dspy.Signature):
    """Decide what to research next given the user's question."""
    question: str = dspy.InputField()
    plan: str = dspy.OutputField()

class Search(dspy.Signature):
    """Generate a focused search query based on the plan."""
    plan: str = dspy.InputField()
    query: str = dspy.OutputField()

class WriteReport(dspy.Signature):
    """Write a clear answer based on the research findings."""
    question: str = dspy.InputField()
    findings: str = dspy.InputField()
    report: str = dspy.OutputField()

class ResearchAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.planner = dspy.Predict(Plan)
        self.searcher = dspy.Predict(Search)
        self.writer = dspy.Predict(WriteReport)

    def forward(self, question):
        plan = self.planner(question=question)
        results = self.searcher(plan=plan.plan)
        return self.writer(question=question, findings=results.query)

app = with_memory(ResearchAgent(), recursive=True)
state = app.new_state()
with app.use_state(state):
    app(question="What are the latest advances in quantum computing?")
print(len(state.turns))                          # 1 root turn
print(len(state.node_states["planner"].turns))   # 1 planner turn
print(len(state.node_states["searcher"].turns))  # 1 searcher turn
```

---

## `Session(...)` parameters

### `history_field`

By default, `dspy-session` injects conversation history into a kwarg called `history`. This parameter lets you change that name.

**Realistic scenario:** Your team already has a codebase where all signatures expect history under the name `chat_history`. Renaming everything would be a large refactor. Instead, you just set `history_field="chat_history"` and the session aligns with your existing code.

**When not to use:** If you are starting a project from scratch, just leave the default `"history"`. There is no benefit to changing it unless you have a naming conflict or an existing convention to match.

```python
class ChatBot(dspy.Signature):
    """A friendly conversational assistant."""
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

# Your team's codebase uses "chat_history" everywhere — match that convention
bot = sessionify(dspy.Predict(ChatBot), history_field="chat_history")
bot(question="Hi there!")
print(bot.history_field)  # chat_history
```

### `max_turns`

This limits the number of historical messages the LLM sees in its prompt. It truncates from the oldest, keeping only the N most recent turns.

**Realistic scenario:** You are running a customer support bot. Conversations can go 50+ turns as the user describes their issue, provides order numbers, and asks follow-ups. But your LLM has a 4K context window and the prompt template itself takes up 500 tokens. You set `max_turns=10` so only the last 10 exchanges are sent, keeping the prompt under budget.

**Edge case:** Old context is silently dropped. If the user mentioned their order number in turn 1 and you are now on turn 15 with `max_turns=10`, the model no longer sees the order number. For this reason, if you need long-term memory, pair `max_turns` with a `consolidator` that summarizes old turns into persistent facts, or use `initial_history` to pin critical information.

**When not to use:** Short conversations (under 10 turns) where context window is not a concern. Leaving it unset lets the model see everything.

```python
class SupportAgent(dspy.Signature):
    """Answer customer support questions."""
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

# Long conversations: only keep last 2 turns in the prompt to fit context window
bot = sessionify(dspy.Predict(SupportAgent), max_turns=2)
bot(question="I need help with my order")
bot(question="The order number is #12345")
bot(question="It arrived damaged")
# Only the 2 most recent turns are sent to the LLM
print(len(bot.session_history.messages))  # 2
```

### `max_stored_turns`

Unlike `max_turns` (which limits what the LLM sees), `max_stored_turns` limits how many turns the session keeps in memory at all. Old turns are permanently deleted, not just hidden from the prompt.

**Realistic scenario:** You have a 24/7 production chatbot. Some power users send hundreds of messages per day. After a few months, a single user's session object holds 10,000+ turns, eating up RAM and making serialization slow. Setting `max_stored_turns=200` keeps the in-memory footprint bounded.

**When not to use:** If you plan to call `to_examples()` later to extract a complete training dataset from the conversation, you need all turns. Leaving this unset preserves the full history for later extraction.

```python
class SupportAgent(dspy.Signature):
    """Answer customer support questions."""
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

# Prevent memory bloat in long-running sessions
bot = sessionify(dspy.Predict(SupportAgent), max_stored_turns=2)
for i in range(4):
    bot(question=f"Question number {i}")
# Older turns are permanently deleted from memory
print(len(bot.turns))  # 2
```

### `exclude_fields`

This prevents specific fields from being written into the rolling history ledger. The field is still available during the current turn (the model sees it right now), and it is still saved in `turn.inputs` for the optimizer, but it will not appear in any future turn's history.

**Realistic scenario — per-turn search results:** Your agent calls a search API each turn and gets back a big chunk of text in a `search_results` field. The answer already summarizes the useful bits. You don't need the raw search results repeated in the history for every past turn — that would eat your entire context window by turn 5. Excluding `search_results` keeps the history clean (just question/answer pairs) while the model still sees the current turn's results.

**Realistic scenario — ChainOfThought rationale:** You are using `dspy.ChainOfThought`, which produces a `rationale` output on every turn. After 10 turns, the history would contain 10 verbose rationale paragraphs. The user never sees these, and the model doesn't need old rationales to answer the next question. Excluding `rationale` keeps the history lean.

**Common misconception — "I'll pass context once and exclude it":** If you are thinking "I'll pass a document as `context` in turn 1 and exclude it so it doesn't repeat" — be careful. Excluding it means the model will completely forget the document by turn 2. If you want the model to reference that document across the entire conversation, do NOT exclude it. Just let it stay in history. The scenario where `exclude_fields` shines is data that is **different every turn and bulky** — like fresh search results, new rationale, or new tool output — not data you want to persist.

**Optimizer behavior:** Even though the field is excluded from history, `turn.inputs` still faithfully records it. When you call `to_examples()`, the optimizer sees that `search_results` was present during that specific turn. The `history_snapshot` in the example accurately reflects that the LLM did not see past search results — giving the optimizer a perfect trace of what was actually in the context window at each moment.

```python
class ContextualQA(dspy.Signature):
    """Answer questions using search results retrieved fresh each turn."""
    question: str = dspy.InputField()
    search_results: str = dspy.InputField(desc="Fresh results retrieved for this question")
    answer: str = dspy.OutputField()

bot = sessionify(dspy.Predict(ContextualQA), exclude_fields={"search_results"})
bot(question="What's your return policy?", search_results="[Doc: 30-day return window for most items...]")
bot(question="Does that apply to electronics?", search_results="[Doc: Electronics have a 15-day window...]")
# History only shows question/answer — no bulky search results repeated
print(bot.session_history.messages[0].keys())  # dict_keys(['question', 'answer'])
```

### `input_field_override` (legacy alias)

This is a legacy alias for `history_input_fields`. You might encounter it in older codebases. Prefer `history_input_fields` in new code.

```python
class ChatBot(dspy.Signature):
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

bot = sessionify(dspy.Predict(ChatBot), input_field_override={"question"})
bot(question="Hello")
```

### `history_input_fields`

By default, all input fields are recorded into the session history. This parameter lets you explicitly allow-list only specific input fields.

**Realistic scenario:** Your signature takes `question`, `user_id`, `timestamp`, and `locale` as inputs. Only `question` is meaningful for conversation context. You don't want the history to be cluttered with `{"question": "Hi", "user_id": "u_abc", "timestamp": "2025-01-01T00:00:00", "locale": "en-US", "answer": "Hello!"}` on every turn. Setting `history_input_fields={"question"}` keeps the history clean: just `{"question": "Hi", "answer": "Hello!"}`.

**Difference from `exclude_fields`:** `exclude_fields` removes fields from both input and output recording in history. `history_input_fields` is an allow-list that only affects inputs. If you need to exclude an output field (like `rationale`), use `exclude_fields`.

```python
class SupportWithMetadata(dspy.Signature):
    """Answer support questions. Metadata is passed for routing but shouldn't clutter history."""
    question: str = dspy.InputField()
    user_id: str = dspy.InputField(desc="Internal routing ID")
    timestamp: str = dspy.InputField(desc="Request timestamp")
    locale: str = dspy.InputField(desc="User's locale")
    answer: str = dspy.OutputField()

# Only record the question in history — not user_id, timestamp, or locale
bot = sessionify(dspy.Predict(SupportWithMetadata), history_input_fields={"question"})
bot(question="What's your return policy?", user_id="u_abc", timestamp="2025-01-15T10:00:00", locale="en-US")
# History is clean: just {"question": "What's your return policy?", "answer": "..."}
print(bot.session_history.messages[0].keys())
```

### `initial_history`

This seeds the session with a pre-existing conversation before any real turns happen. The model sees this history as if those exchanges already occurred.

**Realistic scenario — resuming a conversation:** A user chatted with your bot yesterday. You saved their session to a database. Today they come back. You load the old turns and pass them as `initial_history` so the model picks up right where they left off.

**Realistic scenario — system context as a fake turn:** You want the model to always "remember" a piece of context, like "You are a helpful cooking assistant. The user is vegetarian." You encode this as a synthetic first turn in `initial_history` — the model sees it at the start of every conversation without you having to pass it every call.

**Edge case:** If you later use `history_policy="replace_session"` and pass a new history, the initial history is overwritten and the old turns are cleared.

```python
class Assistant(dspy.Signature):
    """A helpful assistant that remembers prior context."""
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

# Resume a conversation from yesterday — loaded from your database
yesterday = History(messages=[
    {"question": "I'm looking for a new laptop", "answer": "What's your budget and main use case?"},
    {"question": "About $1500, mostly for programming", "answer": "I'd recommend the MacBook Pro M3 or ThinkPad X1 Carbon."},
])
bot = sessionify(dspy.Predict(Assistant), initial_history=yesterday)
# The model picks up right where you left off
bot(question="Which one has better Linux support?")
print(len(bot.turns[0].history_snapshot.messages))  # 2 (the seeded messages)
```

### `history_policy="override"`

This is the default policy. When someone passes an explicit `history` kwarg during a call, the session uses that history as-is, does not inject its own, and does not record the turn.

**Realistic scenario — DSPy optimizer replay:** During optimization, DSPy replays specific traces. It passes its own carefully constructed `history` to see how the model responds to that exact context. The session must be transparent here — it should not add its own turns on top of the optimizer's history, and it should not record the replay as a real turn. This is exactly what `override` does.

**When not to use:** If you want the session to keep recording even when explicit history is provided, use `use_if_provided` instead.

```python
class QA(dspy.Signature):
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

bot = sessionify(dspy.Predict(QA), history_policy="override")
bot(question="What's 2+2?")
# Optimizer passes its own history — session steps aside and doesn't record this turn
bot(question="What's 3+3?", history=History(messages=[]))
print(len(bot.turns))  # 1 (the optimizer call was not recorded)
```

### `history_policy="use_if_provided"`

With this policy, if explicit history is provided, the model uses it for the current turn, but the session still records the turn normally.

**Realistic scenario — supervisor correction:** A human supervisor is monitoring a support bot. They notice the conversation went off-track. They inject a corrected history for the next turn ("actually, the user's issue is about shipping, not billing") but still want the session to keep recording from this point forward.

```python
class QA(dspy.Signature):
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

# Supervisor can inject corrected context, but the session keeps recording
bot = sessionify(dspy.Predict(QA), history_policy="use_if_provided")
bot(question="Tell me about billing")
corrected = History(messages=[
    {"question": "The user's issue is about shipping, not billing", "answer": "Understood, switching to shipping."},
])
bot(question="When will my package arrive?", history=corrected)
print(len(bot.turns))  # 2 (both turns recorded)
```

### `history_policy="replace_session"`

This policy completely overwrites the session's memory when explicit history is provided. All previous turns are cleared, and the new history becomes the seed.

**Realistic scenario — "load this old conversation":** Your UI has a sidebar showing past conversations. The user clicks one to resume it. You pass that old conversation as `history`, and the session drops whatever was happening before and starts fresh from the loaded conversation.

```python
class QA(dspy.Signature):
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

bot = sessionify(dspy.Predict(QA), history_policy="replace_session")
bot(question="Current topic discussion")
# User clicks "load old conversation" in the sidebar
old_conversation = History(messages=[
    {"question": "We were discussing project deadlines", "answer": "The deadline is March 15th."},
])
bot(question="Can we push that back a week?", history=old_conversation)
print(len(bot.turns))  # 1 (old turns cleared, fresh start from loaded conversation)
```

### `on_metric_error`

Controls what happens when a metric function crashes during `score()`.

**Realistic scenario:** You are evaluating 10,000 production sessions overnight. Some sessions have corrupted data (a user sent empty strings, or an output field is missing). Your metric crashes on those. With `on_metric_error="zero"`, the bad turns get a score of 0.0 and the evaluation continues. You review the warnings in the morning. With `"raise"`, the entire evaluation would have crashed on the first bad turn at 2 AM and you'd wake up to nothing.

**When to use `"raise"`:** When you are developing a new metric and need to see the full stack trace to debug it.

```python
class QA(dspy.Signature):
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

bot = sessionify(dspy.Predict(QA), on_metric_error="zero")
bot(question="Hello")

def buggy_metric(example, pred, trace=None):
    raise RuntimeError("metric crashed on corrupted data")

# Evaluation continues despite the crash — scores 0.0 instead of blowing up
scores = bot.score(buggy_metric)
print(scores)  # [0.0]
```

### `strict_history_annotation`

When enabled, the session identifies history fields by their Python type annotation (`History`) rather than by name heuristics.

**Realistic scenario:** Your signature has a field called `search_history: list[str]` (a list of past search queries, not a DSPy History object) and a field called `history: History` (the actual conversation history). Without strict mode, the session might confuse the two because both contain "history" in the name. With `strict_history_annotation=True`, it only picks the field annotated with `History`.

**When not to use:** Most projects have a single history field with a clear name. Strict mode adds no value in that case and can cause issues if your history field has a non-standard annotation.

```python
class QA(dspy.Signature):
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

# Only identify history fields by their History type annotation, not by name
bot = sessionify(dspy.Predict(QA), strict_history_annotation=True)
bot(question="Hello")
```

### `copy_mode`

Determines whether the wrapped module is deep-copied, shallow-copied, or used by reference.

**Realistic scenario — production serving:** You have a FastAPI server with one global agent. The model weights are 500MB in memory. You certainly don't want to deep-copy them for every user request. You use `copy_mode="none"` so all requests share the exact same module weights, and use `SessionState` objects (via `new_state()` / `use_state()`) to keep per-user conversation data separate.

**Realistic scenario — notebook experimentation:** You are experimenting in a Jupyter notebook. You create multiple sessions from the same base module and tweak each one differently. `copy_mode="deep"` (the default) ensures your experiments don't interfere with each other.

**When not to use `"none"`:** If you are mutating the module between sessions (different prompts, different demos), `"none"` means all sessions share the mutation. Use `"deep"` to keep them independent.

```python
class QA(dspy.Signature):
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

base_module = dspy.Predict(QA)
# Production: share the same module weights across sessions
shared = sessionify(base_module, copy_mode="none")
# Notebook: get an independent copy for experimentation
independent = sessionify(base_module, copy_mode="deep")
print(shared.module is base_module)      # True  — same object in memory
print(independent.module is base_module) # False — completely separate copy
```

### `lock`

Adds thread or async locking around session execution to prevent race conditions.

**Realistic scenario:** You have a FastAPI async endpoint. Two requests for the same user arrive at nearly the same time. Both read the turn ledger, both append a turn, and one overwrites the other. Setting `lock="async"` ensures only one coroutine touches the ledger at a time.

**When not to use:** Single-threaded scripts, notebooks, or batch evaluations. Locking adds overhead and complexity that you don't need when there is no concurrency.

```python
class QA(dspy.Signature):
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

bot = sessionify(dspy.Predict(QA), lock="async")

async def handle_requests():
    await bot.acall(question="First concurrent request")
    await bot.acall(question="Second concurrent request")

asyncio.run(handle_requests())
print(len(bot.turns))  # 2
```

### `on_turn`

A callback function that fires every time a turn finishes recording.

**Realistic scenario — auto-save to Redis:** After every turn, you want to persist the session state to Redis so that if the server crashes, no conversation data is lost. Your `on_turn` hook serializes the state and writes it to the cache.

**Realistic scenario — streaming UI:** You have a WebSocket-based chat UI. Every time a turn completes, your hook pushes the new turn to the frontend so the chat bubbles update in real time.

**Edge case:** Keep the callback fast. If your hook does a slow database write, it blocks the session's response. For heavy I/O, consider queuing the work to a background task inside the hook.

```python
class QA(dspy.Signature):
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

save_log = []

def auto_save(session, turn):
    # In production: save to Redis, push to WebSocket, etc.
    save_log.append({"turn": turn.index, "question": turn.inputs["question"]})

bot = sessionify(dspy.Predict(QA), on_turn=auto_save)
bot(question="What's your return policy?")
print(save_log)  # [{'turn': 0, 'question': "What's your return policy?"}]
```

### `isolation`

Determines whether a nested child module maintains its own private history (`"isolated"`) or reads from its parent's history (`"shared"`).

**Realistic scenario — shared translator:** Your agent has a `translator` sub-module. It needs to see the main conversation to know what language the user is speaking. Setting it to `"shared"` lets it read the parent's history. It doesn't build up its own turn ledger — it just piggybacks on the parent context.

**Realistic scenario — isolated code executor:** Your agent has a `code_executor` that iteratively writes and debugs code over several internal calls. You want it to have a private scratchpad of its own attempts, invisible to the parent conversation. Setting it to `"isolated"` gives it its own ledger.

```python
class Translate(dspy.Signature):
    """Translate text, using conversation context to infer the target language."""
    text: str = dspy.InputField()
    answer: str = dspy.OutputField()

class ChatAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.translator = dspy.Predict(Translate)

    def forward(self, question):
        return self.translator(text=question)

# Translator reads the parent's history to know the conversation language
app = with_memory(
    ChatAgent(),
    child_configs={"translator": {"isolation": "shared"}},
)
state = app.new_state()
with app.use_state(state):
    app(question="Bonjour, comment ça va?")
    app(question="Quelle est la météo?")
# Shared mode: translator has no private ledger, it reads the parent's history
print(len(state.node_states["translator"].turns))  # 0
```

### `lifespan`

Controls how long a nested module retains its turns.

**Realistic scenario — episodic web searcher:** Your agent has a `web_searcher` sub-module. On each user turn, the searcher runs 3-5 internal queries to find information. You don't want the searcher to remember queries from 5 user turns ago — that's stale and wastes context. Setting `lifespan="episodic"` clears the searcher's memory at the end of every top-level turn. Each new user message gives the searcher a fresh slate.

**Realistic scenario — stateless calculator:** Your agent has a `calculator` sub-module. It takes a math expression and returns a number. There is zero reason for it to remember past calculations. Setting `lifespan="stateless"` prevents it from recording any turns at all.

**Realistic scenario — persistent planner:** Your agent's `planner` needs to remember its entire history of plans across the whole conversation so it doesn't repeat strategies. Leave it as `"persistent"` (the default).

```python
class SearchQuery(dspy.Signature):
    """Generate a search query to find relevant information."""
    question: str = dspy.InputField()
    query: str = dspy.OutputField()

class SearchAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.searcher = dspy.Predict(SearchQuery)

    def forward(self, question):
        # Searcher runs multiple queries per user turn
        self.searcher(question=question)
        self.searcher(question=f"More details on: {question}")
        return dspy.Prediction(answer="done")

# Searcher resets after each user turn — no stale queries from old turns
app = with_memory(
    SearchAgent(),
    child_configs={"searcher": {"lifespan": "episodic"}},
)
state = app.new_state()
with app.use_state(state):
    app(question="quantum computing advances")
# Episodic: searcher's turns were cleared at the end of the macro-turn
print(len(state.node_states["searcher"].turns))  # 0
```

### `consolidator`

A DSPy module that runs at the end of an episode (when `lifespan="episodic"`) to summarize the episode's turns into long-term facts stored in `l2_memory`.

**Realistic scenario:** Your `researcher` sub-module runs 10 web searches per user turn (episodic). At the end of the turn, a consolidator summarizes: "Key facts found: The company was founded in 2019. Revenue grew 40% YoY. CEO is Jane Smith." This summary is stored in `l2_memory`. On the next user turn, the researcher starts with a fresh scratchpad but has access to the consolidated facts via `get_node_memory()`, so it doesn't repeat work.

**When not to use:** If the sub-module is `"persistent"` (it keeps all turns anyway) or `"stateless"` (it has no turns to consolidate), a consolidator has nothing to do.

```python
class SearchQuery(dspy.Signature):
    """Generate a search query."""
    question: str = dspy.InputField()
    query: str = dspy.OutputField()

class Summarize(dspy.Signature):
    """Summarize research findings into key facts for long-term memory."""
    past_memory: str = dspy.InputField()
    episode_transcript: str = dspy.InputField()
    updated_memory: str = dspy.OutputField()

class ResearchAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.searcher = dspy.Predict(SearchQuery)

    def forward(self, question):
        self.searcher(question=question)
        return dspy.Prediction(answer="done")

# Searcher is episodic with a consolidator that extracts key facts
app = with_memory(
    ResearchAgent(),
    child_configs={"searcher": {
        "lifespan": "episodic",
        "consolidator": dspy.Predict(Summarize),
    }},
)
state = app.new_state()
with app.use_state(state):
    app(question="What is quantum computing?")
# The consolidator summarized the searcher's episode into long-term memory
print(state.node_states["searcher"].l2_memory)
```

---

## `with_memory(...)` policy parameters

### `recursive`

Determines whether `with_memory` should traverse the module tree and wrap nested predictors with their own sessions.

**Realistic scenario:** You have a research agent with a `planner`, `researcher`, and `writer`. You want all three to independently remember their past actions. Setting `recursive=True` wraps all of them in one call.

**When to set `False`:** If you only want the top-level agent to have memory (for example, to track user messages) but want sub-modules to remain stateless. Or if you want fine-grained control and prefer to use `child_configs` to wrap only specific children.

```python
class Plan(dspy.Signature):
    question: str = dspy.InputField()
    plan: str = dspy.OutputField()

class PlanningAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.planner = dspy.Predict(Plan)

    def forward(self, question):
        return self.planner(question=question)

app = with_memory(PlanningAgent(), recursive=True)
print(isinstance(app.module.planner, Session))  # True — planner got its own session
```

### `include`

An explicit allow-list of nested module paths to wrap with memory. Only these paths get sessions; everything else stays stateless.

**Realistic scenario:** You have a large agent with 8 sub-modules: `planner`, `researcher`, `writer`, `formatter`, `validator`, `translator`, `summarizer`, `calculator`. Only `planner` and `writer` actually benefit from conversation memory. The rest are pure functions or one-shot tools. Setting `include={"planner", "writer"}` avoids wrapping the other 6 with unnecessary session overhead.

```python
class Plan(dspy.Signature):
    question: str = dspy.InputField()
    plan: str = dspy.OutputField()

class Format(dspy.Signature):
    text: str = dspy.InputField()
    formatted: str = dspy.OutputField()

class PipelineAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.planner = dspy.Predict(Plan)
        self.formatter = dspy.Predict(Format)

    def forward(self, question):
        plan = self.planner(question=question)
        return self.formatter(text=plan.plan)

# Only the planner needs memory — formatter is a pure function
app = with_memory(PipelineAgent(), include={"planner"})
print(isinstance(app.module.planner, Session))    # True
print(isinstance(app.module.formatter, Session))  # False
```

### `exclude`

The inverse of `include`. Everything gets wrapped except the listed paths.

**Realistic scenario:** Almost all your sub-modules benefit from memory, except `formatter` which is a deterministic template — giving it history would just waste tokens. Setting `exclude={"formatter"}` keeps it stateless while wrapping everything else.

```python
# Using the PipelineAgent class from the include example above
app = with_memory(PipelineAgent(), exclude={"formatter"})
print(isinstance(app.module.formatter, Session))  # False
```

### `where`

A programmatic filter function `(path, module) -> bool` that decides which nested modules get memory.

**Realistic scenario:** Your team follows a naming convention where all stateless tool modules are prefixed with `tool_` (like `tool_calculator`, `tool_formatter`). You write `where=lambda path, obj: not path.startswith("tool_")` and every tool is automatically excluded without maintaining a hardcoded list.

**When not to use:** If you have a fixed, small set of modules to include or exclude, `include`/`exclude` are simpler and more readable.

```python
# Using the PipelineAgent class from the include example above
app = with_memory(
    PipelineAgent(),
    where=lambda path, obj: path != "formatter",
)
print(isinstance(app.module.planner, Session))    # True
print(isinstance(app.module.formatter, Session))  # False
```

### `child_configs`

Per-path configuration dictionaries for nested modules. This is the most powerful knob — you can set isolation, lifespan, consolidators, or completely disable wrapping for each specific sub-module.

**Realistic scenario:** You are building a research agent. The `planner` should be persistent (remembers all past plans). The `researcher` should be episodic with a consolidator (does fresh searches each turn but remembers key facts). The `formatter` should be disabled entirely (pure function, no memory needed). `child_configs` lets you express all of this in one place.

```python
# Using the PipelineAgent class from the include example above
app = with_memory(
    PipelineAgent(),
    child_configs={
        "planner": {"lifespan": "episodic"},
        "formatter": {"enabled": False},
    },
)
print(isinstance(app.module.planner, Session))    # True
print(isinstance(app.module.formatter, Session))  # False
```

---

## Session object APIs

### `new_state()` / `use_state()`

These are the cornerstone of production serving. Instead of creating a new session object per user (which would duplicate model weights), you keep one global session and create lightweight `SessionState` containers for each user.

**Realistic scenario — FastAPI endpoint:** You have one global `agent = with_memory(MyAgent(), copy_mode="none")`. When a request arrives, you load the user's `SessionState` from Redis, bind it with `with agent.use_state(user_state):`, run the agent, then save the updated state back to Redis. Thousands of concurrent users share the same model weights but each has their own private conversation.

**Edge case:** If you forget to bind a state, the session falls back to its internal default state. This is fine for notebooks but dangerous in production — two users would share the same conversation.

```python
class QA(dspy.Signature):
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

# One global agent shared by all users
agent = sessionify(dspy.Predict(QA), copy_mode="none")

# Each user gets their own lightweight state container
alice = agent.new_state()
bob = agent.new_state()

with agent.use_state(alice):
    agent(question="I need help with billing")
with agent.use_state(bob):
    agent(question="How do I reset my password?")

# Completely separate conversations, same model weights
print(len(alice.turns), len(bob.turns))  # 1 1
```

### `turns`, `session_history`, `len(session)`

These properties let you inspect what the session has recorded.

**Realistic scenario — rendering a chat UI:** Your frontend needs to display chat bubbles. You iterate over `session.turns` to get each turn's `inputs` and `outputs` and render them. `session_history` gives you the formatted `History` object that the LLM would actually see, which is useful for debugging prompt issues. `len(session)` is a quick check for empty conversations (e.g., showing a welcome message when `len(session) == 0`).

```python
class QA(dspy.Signature):
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

bot = sessionify(dspy.Predict(QA))
bot(question="What's your return policy?")
print(len(bot))                      # 1
print(bot.turns[-1].inputs)          # {'question': "What's your return policy?"}
print(bot.session_history.messages)  # [{'question': "...", 'answer': "..."}]
```

### `add_turn`, `pop_turn`, `undo`

Manual manipulation of the conversation ledger.

**Realistic scenario — "regenerate" button:** The user doesn't like the last response and clicks "Regenerate." You call `pop_turn()` to remove the bad response, then re-run the module to get a fresh answer. The model sees the same history it saw before and produces a different response (due to temperature).

**Realistic scenario — importing a conversation:** You are migrating from another chat system. You have a JSON export of old conversations. You use `add_turn()` to manually inject each past exchange into the session so the model can continue from where the old system left off.

**Realistic scenario — undo:** A human reviewer is doing quality control on a bot conversation. They realize the last 3 turns went off-track. They call `undo(steps=3)` to roll the conversation back and try a different approach.

```python
class QA(dspy.Signature):
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

bot = sessionify(dspy.Predict(QA))

# Import a turn from another system
bot.add_turn(
    inputs={"question": "What's your return policy?"},
    outputs={"answer": "You can return items within 30 days."},
)
print(len(bot.turns))  # 1

# User clicks "regenerate" — remove the last response
removed = bot.pop_turn()
print(removed.inputs["question"])  # What's your return policy?

# Normal conversation, then undo the last turn
bot(question="Hello")
bot(question="What products do you sell?")
bot.undo(steps=1)
print(len(bot.turns))  # 1
```

### `reset`

Clears all turns from the current session state. The module, configuration, and initial history are preserved — only the recorded turns are wiped.

**Realistic scenario:** The user clicks "New Chat" in your UI. You don't need to create a new session object or reload model weights. Just call `reset()` and the conversation starts fresh. If you have `initial_history` set (like a system prompt), it is still there after the reset.

```python
class QA(dspy.Signature):
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

bot = sessionify(dspy.Predict(QA))
bot(question="What's your return policy?")
# User clicks "New Chat"
bot.reset()
print(len(bot.turns))  # 0
```

### `fork`

Creates a completely independent copy of the session, including all its current turns and state. Changes to the fork do not affect the original.

**Realistic scenario — exploring alternatives:** A user asks "Should I use React or Vue?" You want to show them two possible continuations. You fork the session, run one branch with "Let's explore React" and the other with "Let's explore Vue." Each branch has full context of the original conversation but diverges from here.

**Realistic scenario — evaluation:** You want to test how the model responds to the same conversation under different prompts. Fork the session, `update_module()` on the fork with a different prompt, and compare.

```python
class QA(dspy.Signature):
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

bot = sessionify(dspy.Predict(QA))
bot(question="Should I use React or Vue for my project?")

# Explore two different directions from the same starting point
react_branch = bot.fork()
vue_branch = bot.fork()
react_branch(question="Tell me more about React's ecosystem")
vue_branch(question="Tell me more about Vue's learning curve")

print(len(bot.turns))           # 1 (original unchanged)
print(len(react_branch.turns))  # 2
print(len(vue_branch.turns))    # 2
```

### `update_module`

Hot-swaps the underlying DSPy module while keeping the entire conversation state intact.

**Realistic scenario — deploying optimized weights:** You ran DSPy optimization overnight and produced a new, better-performing module. Users are in active conversations. Instead of killing their sessions and starting over, you call `update_module(optimized_module)` and they seamlessly continue their conversations with the improved model.

**Realistic scenario — dynamic capability upgrade:** A user starts chatting with a general assistant. Midway, they ask a complex coding question. You swap in a code-specialized module with `update_module()` without losing the conversation context.

```python
class QA(dspy.Signature):
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

bot = sessionify(dspy.Predict(QA))
bot(question="What's your return policy?")

# Deploy optimized module overnight — swap it in without losing conversation
optimized_module = dspy.ChainOfThought(QA)
bot.update_module(optimized_module)
bot(question="Does that apply to electronics?")  # continues with full history
```

### `score`

Runs an evaluation metric against every turn in the session, returning a list of scores.

**Realistic scenario — post-conversation quality review:** After a customer support conversation ends, you score each turn for helpfulness. Turns scoring below 0.5 are flagged for human review. This helps you identify exactly where in a 20-turn conversation the bot went off the rails, rather than just scoring the conversation as a whole.

**When not to use:** If you only care about a single final answer (not the trajectory), a simple metric on the last turn's output is simpler.

```python
class QA(dspy.Signature):
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

bot = sessionify(dspy.Predict(QA))
bot(question="What's 2+2?")
bot(question="And what's that times 3?")

def helpfulness(example, pred, trace=None):
    # In production: use an LLM-as-judge or custom logic
    return 1.0 if len(pred.answer) > 5 else 0.0

scores = bot.score(helpfulness)
print(scores)  # e.g., [1.0, 1.0]
```

### `to_examples` / `to_trainset`

Converts the session's internal ledger into standard `dspy.Example` objects, ready for DSPy optimization.

**Realistic scenario:** You have deployed a chatbot for a month. You have 1,000 customer conversations stored as sessions. You call `to_examples(metric=quality_metric, min_score=0.8)` to extract only the high-quality turns. These become your training set for DSPy prompt optimization — the optimizer sees exactly what history was in the prompt at each moment and learns to produce better responses in that context.

**When not to use:** If you are only serving and never plan to optimize, you don't need this. But it's one of the key reasons sessions exist — they bridge runtime usage and offline training.

```python
class QA(dspy.Signature):
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

bot = sessionify(dspy.Predict(QA))
bot(question="What's your return policy?")
bot(question="How long does a refund take?")

examples = bot.to_examples()    # list of dspy.Example objects
trainset = bot.to_trainset()    # same thing, alias
print(len(examples), len(trainset))  # 2 2
```

### `Session.merge_examples`

Combines examples from multiple independent sessions into a single list.

**Realistic scenario:** You have 50 customer support sessions from different users. Each has 10-20 turns. You want to aggregate all the good turns into one training dataset. Instead of manually concatenating lists, `merge_examples(session_a, session_b, ..., min_score=0.7)` does it in one call.

```python
class QA(dspy.Signature):
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

# Two separate customer conversations
alice_session = sessionify(dspy.Predict(QA))
bob_session = sessionify(dspy.Predict(QA))
alice_session(question="Help with billing")
bob_session(question="Help with shipping")

# Combine into one training dataset
merged = Session.merge_examples(alice_session, bob_session)
print(len(merged))  # 2
```

### `save`, `save_state`, `load_from`

Serialization APIs for persisting session state to disk or a database.

**Realistic scenario — nightly backup:** Every night, your server serializes all active sessions to JSON files on S3. If the server crashes, you reload them on startup with `Session.load_from()`. Users resume their conversations as if nothing happened.

**Realistic scenario — conversation logging:** For compliance or debugging, you call `save_state()` after every conversation to store the complete transcript (with all turn metadata and scores) in a Postgres JSONB column.

**`save_state()` vs `save()`:** `save_state()` returns a Python dictionary — useful when you want to store it in a database or send it over an API. `save(path)` is a convenience that writes the dict to a JSON file directly.

```python
class QA(dspy.Signature):
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

bot = sessionify(dspy.Predict(QA))
bot(question="What's your return policy?")

# Get a dict for database storage
state_dict = bot.save_state()
print(state_dict["version"])  # 2

# Or save straight to a file
bot.save("conversation.json")
loaded = Session.load_from("conversation.json", dspy.Predict(QA))
print(len(loaded.turns))  # 1
```

---

## Projection helper APIs

### `get_current_history()`

Retrieves the `History` object for the currently executing turn from inside a running module. This is a low-level escape hatch.

**Realistic scenario — custom prompt formatting:** You are building a custom module that needs to know how far into the conversation it is, or needs to format the history in a special way before passing it to a sub-predictor. Inside your module's `forward()`, you call `get_current_history()` to get the raw history object and process it however you want.

**When not to use:** In most cases, the session handles history injection automatically. You only need this when you are doing something unusual with how history is rendered or processed inside a custom module.

```python
class QA(dspy.Signature):
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

class TurnAwareBot(dspy.Module):
    def __init__(self):
        super().__init__()
        self.qa = dspy.Predict(QA)

    def forward(self, question):
        # Access current history to add turn-awareness
        h = get_current_history()
        turn_number = len(h.messages) + 1
        return self.qa(question=f"[Turn {turn_number}] {question}")

bot = sessionify(TurnAwareBot())
bot(question="What's your return policy?")
bot(question="How long does a refund take?")
```

### `get_outer_history()`

Retrieves the history of the parent or root session from inside a nested child module.

**Realistic scenario:** Your `writer` sub-agent is deeply nested inside a research pipeline. It has its own isolated history of drafts. But to write a good response, it needs to know what the user originally asked — that information lives in the root session's history. Inside `writer.forward()`, you call `get_outer_history()` to read the root conversation and include the user's original question in the prompt.

```python
class WriterSig(dspy.Signature):
    """Write a response informed by the full user conversation."""
    task: str = dspy.InputField()
    user_context: str = dspy.InputField(desc="Summary of the user's conversation so far")
    result: str = dspy.OutputField()

class Writer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.draft = dspy.Predict(WriterSig)

    def forward(self, task):
        # Read the root conversation to understand the user's original question
        outer = get_outer_history()
        context = " | ".join(msg.get("question", "") for msg in outer.messages)
        return self.draft(task=task, user_context=context or "No prior context")

class MainAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.writer = Writer()

    def forward(self, question):
        return self.writer(task=question)

app = with_memory(MainAgent())
state = app.new_state()
with app.use_state(state):
    app(question="Tell me about quantum computing")
```

### `get_node_memory()`

Retrieves the consolidated long-term facts (`l2_memory`) for the currently executing node.

**Realistic scenario:** Your `researcher` sub-module is episodic — its scratchpad resets every user turn. But a consolidator saved "Key facts: User prefers Python. User is building a REST API." into `l2_memory` from previous episodes. Inside `researcher.forward()`, you call `get_node_memory()` and inject these facts into the prompt so the researcher doesn't re-discover things it already learned.

```python
class ResearchSig(dspy.Signature):
    """Research a topic, informed by previously learned facts."""
    question: str = dspy.InputField()
    prior_knowledge: str = dspy.InputField(desc="Facts from previous research episodes")
    answer: str = dspy.OutputField()

class Researcher(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(ResearchSig)

    def forward(self, question):
        # Inject consolidated facts from previous episodes
        facts = get_node_memory()
        return self.predict(question=question, prior_knowledge=facts or "None yet")

app = with_memory(Researcher())
state = app.new_state(l2_memory="User prefers Python. User is building a REST API.")
with app.use_state(state):
    out = app(question="What web framework should I use?")
print(out.answer)  # Response informed by the stored facts
```

### `get_child_l1_ledger(path)`

Lets a parent module read the internal turn history of a specific nested child.

**Realistic scenario — manager reviewing worker:** Your agent has a `manager` that delegates research to a `searcher` sub-module. The searcher runs several queries internally. Before deciding whether to search more or move to writing, the manager calls `get_child_l1_ledger("searcher")` to see exactly what the searcher found. If the results look thin, the manager tells the searcher to keep going.

```python
class SearchSig(dspy.Signature):
    query: str = dspy.InputField()
    result: str = dspy.OutputField()

class DecideSig(dspy.Signature):
    question: str = dspy.InputField()
    searcher_log: str = dspy.InputField(desc="What the searcher found so far")
    decision: str = dspy.OutputField()

class ManagerAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.searcher = sessionify(dspy.Predict(SearchSig))
        self.decide = dspy.Predict(DecideSig)

    def forward(self, question):
        self.searcher(query=question)
        # Manager reviews what the searcher found before deciding next steps
        ledger = get_child_l1_ledger("searcher")
        return self.decide(question=question, searcher_log=ledger)

app = with_memory(ManagerAgent())
state = app.new_state()
with app.use_state(state):
    out = app(question="Latest AI research papers")
```

### `get_execution_trace()`

Returns a structured view of the entire session hierarchy's call stack and state.

**Realistic scenario — debugging memory routing:** Your multi-agent system has 5 nested modules. The writer is producing off-topic responses. You call `get_execution_trace()` inside the writer's forward and print it. The trace reveals that the writer's history doesn't include the user's original question because it was set to `"isolated"` when it should have been `"shared"`. You fix the config and the issue disappears.

**When not to use:** In production. This is a development and debugging tool. It's verbose and includes internal state that you wouldn't want to send to an LLM or expose to users.

```python
class SearchSig(dspy.Signature):
    query: str = dspy.InputField()
    result: str = dspy.OutputField()

class DebugAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.searcher = sessionify(dspy.Predict(SearchSig))

    def forward(self, question):
        self.searcher(query=question)
        # Debug: inspect the full hierarchy of sessions and their states
        trace = get_execution_trace()
        print(trace)  # Use during development to verify memory routing
        return dspy.Prediction(answer=trace)

app = with_memory(DebugAgent())
state = app.new_state()
with app.use_state(state):
    out = app(question="quantum computing")
print("root" in out.answer and "searcher" in out.answer)  # True
```

---

## Quick checklist for production

- Use `with_memory(..., copy_mode='none')` (default) for shared blueprints.
- Keep per-user state in `SessionState` (`to_dict()` / `from_dict()`).
- Wrap each request with `with session.use_state(user_state): ...`.
- Use `child_configs` for policy control (`shared`, `episodic`, `stateless`, `consolidator`).
- Use TemplateAdapter helpers for advanced projections (`outer_history`, `node_memory`, `child_l1_ledger`, `execution_trace`).
