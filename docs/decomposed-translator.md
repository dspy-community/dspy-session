# Composed translator workflow: correct → translate

This guide shows a **decomposed `dspy.Module` pipeline** that first corrects incoming
text, then translates it, and how to keep that conversation stateful with
`sessionify`.

You get two-step reasoning with clean separation of concerns:

1. **Corrector**: fix grammar/spelling and lightly normalize
2. **Translator**: translate the corrected text to a target language

Then we wrap the composed module so session history is managed automatically.

---

## 1) Define the decomposed pipeline module

```python
import dspy
from dspy_session import sessionify

class CorrectText(dspy.Signature):
    """Correct grammar and spelling in the input sentence."""

    text: str = dspy.InputField(desc="Raw user-provided text")
    corrected: str = dspy.OutputField(desc="Corrected text")


class TranslateText(dspy.Signature):
    """Translate corrected text into the requested target language."""

    corrected: str = dspy.InputField(desc="Already-corrected text")
    target_language: str = dspy.InputField(desc="Language code or name, e.g. 'French' or 'es'")
    translated: str = dspy.OutputField(desc="Translated output")


class CorrectThenTranslate(dspy.Module):
    def __init__(self):
        super().__init__()
        self.corrector = dspy.ChainOfThought(CorrectText)
        self.translator = dspy.ChainOfThought(TranslateText)

    def forward(self, text: str, target_language: str):
        corrected_pred = self.corrector(text=text)
        translated_pred = self.translator(
            corrected=corrected_pred.corrected,
            target_language=target_language,
        )

        # Return both stages for easier debugging and richer training turns
        return dspy.Prediction(
            corrected=corrected_pred.corrected,
            translated=translated_pred.translated,
        )
```

### Plain (non-session) usage

```python
# Configure your LM
dspy.configure(lm=dspy.LM("groq/moonshotai/kimi-k2-instruct-0905"))

module = CorrectThenTranslate()
out = module(text="I have heard from the cliente yesterday, they want the price", target_language="French")
print(out.corrected)
# → I have heard from the client yesterday, they want the price.
print(out.translated)
# → J'ai entendu parler du client hier, ils veulent le prix.
```

This works, but does **not** keep conversation state for follow-up turns.

---

## 2) Sessionify the composed module

```python
chat_translate = sessionify(CorrectThenTranslate())
```

Now each turn is automatically recorded and replayed as history.

```python
first = chat_translate(text="I have heard from the cliente yesterday, they want the price", target_language="French")
print(first.corrected)
# I have heard from the client yesterday, they want the price.
print(first.translated)
# J'ai entendu parler du client hier, ils veulent le prix.

second = chat_translate(
    text="And make it in Spanish too?",
    target_language="Spanish"
)
print(second.corrected)
print(second.translated)
```

Because `chat_translate` is sessionized, the second request can inherit relevant
context (depending on your model/prompt behavior), while your `forward()` method
stays untouched.

```python
print(chat_translate.turns)
```

```python
print(chat_translate.session_history)
```

---

## 3) Why this pattern works well with `dspy-session`

- **No signature changes**: your existing `CorrectThenTranslate.forward(text, target_language)`
  method is unchanged.
- **Nested predictors are wrapped**: session can record history around modules that
  contain internal modules (`corrector`, `translator`).
- **Richer turn data**: returned `Prediction` contains both `corrected` and
  `translated`, so downstream metrics and training examples can target either stage.
- **Optimizer-ready**: turn snapshots can be turned into `dspy.Example`s via
  `session.to_examples()` and used with your favorite optimizers.

---

## 4) Full runnable snippet (fixed target language setup)

This variant stores a fixed `target_language` at module init (the error in the
previous example came from not storing it before using it in `forward()`).

```python
import dspy
from dspy_session import sessionify

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


dspy.configure(lm=dspy.LM("openai/gpt-4.1"))
session = sessionify(CorrectThenTranslate(target_language="French"))

print(session(text="This plant is red").translated)
print(session(text="Can I have it?").translated)

print(session.session_history)
```
