# TemplateAdapter + dspy-session Integration

This guide shows how `dspy-session` and `dspy-template-adapter` fit together.

- `dspy-session` manages turn state and `dspy.History` accumulation
- `TemplateAdapter` controls exact prompt layout

## 1) Recommended baseline (`{"role": "history"}`)


```python
%pip install --upgrade dspy-template-adapter dspy
```

```output:exec-1771899680891-w4uo6
Running: uv pip install --upgrade dspy-template-adapter dspy
--------------------------------------------------
[2mResolved [1m74 packages[0m [2min 574ms[0m
[2mPrepared [1m1 package[0m [2min 153ms[0m
[2mUninstalled [1m1 package[0m [2min 0.49ms[0m
[2mInstalled [1m1 package[0m [2min 3ms[0m
 [31m-[0m [1mhf-xet[0;2m==1.2.0[0m
 [32m+[0m [1mhf-xet[0;2m==1.3.0[0m
--------------------------------------------------
Note: Restart kernel to use newly installed packages.
```


```python
import dspy
from dspy_session import sessionify
from dspy_template_adapter import TemplateAdapter, Predict

adapter = TemplateAdapter(
    messages=[
        {"role": "system", "content": "{instruction}"},
        {"role": "user", "content": "{user}"},
    ],
    parse_mode="full_text",
)

dspy.configure(lm=dspy.LM("groq/moonshotai/kimi-k2-instruct-0905"), adapter=adapter)

chat = sessionify(dspy.Predict("user -> assistant"))

chat(user="My name is Max")
```

```output:exec-1771932407683-a1uil
Prediction(
    assistant='Nice to meet you, Max!'
)
```


```python
chat(user = 'What is my name?')
```

```output:exec-1771899313771-amuyq
Prediction(
    assistant='Your name is Max.'
)
```


```python
chat
```

```output:exec-1771899316172-pokvv
Out[79]: Session(Predict, turns=2, history_field='history')
```


```python
dspy.inspect_history()
```

```output:exec-1771899318153-be9r8




[34m[2026-02-23T21:15:13.959202][0m

[31mSystem message:[0m




[31mUser message:[0m

user: My name is Max


[31mAssistant message:[0m

Got it—thanks, Max! What can I help you with today?


[31mUser message:[0m

What is my name?


[31mResponse:[0m

[32mYour name is Max.[0m
```


This is the most explicit and reliable placement strategy.

---

## 2) Inline history with `{history()}` (new helper)

`TemplateAdapter` now supports a built-in template function:

- `{history()}`
- `{history(style='json')}`
- `{history(style='yaml')}`
- `{history(style='xml')}`

### Everything in system

```python
class ChatSig(dspy.Signature):
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

adapter = TemplateAdapter(
    messages=[
        {
            "role": "system",
            "content": "Conversation so far:\n{history(style='yaml')}",
        },
        {"role": "user", "content": "{question}"},
    ],
    parse_mode="full_text",
)
```

### Everything in user

```python
adapter = TemplateAdapter(
    messages=[
        {"role": "system", "content": "Answer directly."},
        {
            "role": "user",
            "content": "Conversation:\n{history(style='json')}\n\nQuestion:\n{question}",
        },
    ],
    parse_mode="full_text",
)
```

### Half-and-half (contrived)

```python
adapter = TemplateAdapter(
    messages=[
        {
            "role": "system",
            "content": "Long-term memory:\n{history(style='xml')}",
        },
        {
            "role": "user",
            "content": "Recent context:\n{history(style='yaml')}\n\nQ: {question}",
        },
    ],
    parse_mode="full_text",
)
```

These are intentionally unusual, but they demonstrate total prompt control.

---

## 3) Preview without LM calls

```python
preview_msgs = adapter.preview(
    signature=session.module.signature,
    inputs={
        "question": "Follow-up question",
        "history": session.session_history,
    },
)
```

Use this to verify exact prompt structure before runtime.

---

## 4) Optimizer loop snippet (session + template adapter)

```python
# collect conversational data
session(question="Turn 1")
session(question="Turn 2")

trainset = session.to_examples(
    metric=my_metric,
    min_score=0.5,
    strict_trajectory=True,
)

# compile the session-aware module
optimized = dspy.BootstrapFewShot().compile(session, trainset=trainset)
```

If you want single-turn optimization only, use `include_history=False` in `to_examples()`.

---

## 5) Notes

- If template contains `{"role": "history"}`, history is expanded at that message position.
- If template uses `{history(...)}`, history is consumed inline and not auto-injected again.
- Session remains adapter-agnostic; TemplateAdapter is purely prompt-shaping.
