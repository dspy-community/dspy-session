---
title: "Multiple Signatures"
---


```py
# USECASE 4: Multiple predicts using a shared history inside of a module
class CorrectThenTranslate(dspy.Module):
    def __init__(self):
        super().__init__()
        self.correct = dspy.Predict(
            "text, history: dspy.History -> corrected_text", append_history=True
        )
        self.translate = dspy.Predict(
            "text, target_language: str, history: dspy.History -> translation",
            append_history=True,
        )

    def forward(
        self, text: str, history: dspy.History = dspy.History()
    ) -> dspy.Prediction:
        outputs = self.correct(text=text, history=history)
        outputs = self.translate(
            text=outputs.corrected_text,
            target_language="Spanish",
            history=outputs.history,
        )
        return outputs


correct_then_translate = CorrectThenTranslate()
outputs = correct_then_translate(text="What is the capital of France")
outputs = correct_then_translate(
    text="How many people live there?", history=outputs.history
)
```


```py
# Current behavior:
# Sys prompt for translate
# User: Inputs for first translate (What is the capital of France?, Spanish)
# Assistant: Outputs for first translate (¿Cuál es la capital de Francia?)
# User: Inputs for translate (How many people live there?, Spanish)
# Assistant: Outputs for translate (¿Cuántas personas viven allí?)

It could be:
  # Sys prompt for correct
  # User: System + user Instructions for correct (Given the fields `text`, `history`, produce the fields `corrected_text`.
  # User: Inputs for first correct (WhatisthecapitalofFrance)
  # Assistant: Outputs for first correct[0] (What is the capital of France?)
  # User: Instructions for second signature (Given the fields `text`, `target_language`, `history`, produce the fields `translation`.
  # User: Inputs for first translate (What is the capital of France?, Spanish)
  # Assistant: Outputs for first translate (¿Cuál es la capital de Francia?)
  # QUESTION: Do we repeat instructions for correct??
  # User: Inputs for second correct (Howmanypeoplelivethere?)
  # Assistant: Outputs for second correct (How many people live there?)
  # NOTE: We do not repeat instructions for correct, since that is in the system prompt.
```
