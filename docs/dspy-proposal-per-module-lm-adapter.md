# Proposal: Per-module LM and adapter configuration with scoped propagation

## Summary

Allow setting `lm` and `adapter` on any `dspy.Module`, not just on individual `Predict` instances or globally via `dspy.configure()`. When a module with an explicit LM or adapter is called, its setting automatically propagates to all nested predictors via `settings.context()` — unless a nested module or predictor has its own explicit override.

**Priority chain** (highest to lowest):
1. Call-time kwarg: `predict(question="...", lm=my_lm)`
2. Predictor-level: `predict.lm = my_lm`
3. Nearest ancestor module: `module.lm = my_lm` (via context propagation)
4. Global: `dspy.configure(lm=my_lm)`
5. Fallback: `None` for LM (raises), `ChatAdapter()` for adapter

## Motivation

### Current state

**LM** already supports per-predictor override — `Predict` has `self.lm`, and resolution is `kwargs.pop("lm", self.lm) or settings.lm`. But there's no way to set an LM on a composed `dspy.Module` and have it propagate to nested predictors.

**Adapter** has no per-predictor or per-module support at all. It's hardcoded to global:

```python
# predict.py, line 191 (DSPy 3.1.3)
adapter = settings.adapter or ChatAdapter()
```

This means you can't do things like:

```python
# ❌ Not possible today
class MyAgent(dspy.Module):
    def __init__(self):
        self.planner = dspy.ChainOfThought(PlanSig)     # should use XML adapter
        self.executor = dspy.Predict(ExecSig)            # should use JSON adapter

agent = MyAgent()
agent.planner.adapter = XMLAdapter()    # ← attribute doesn't exist on Predict
agent.executor.adapter = JSONAdapter()  # ← same
```

The only workarounds are:
- Change `dspy.settings.adapter` globally before each call (fragile, not thread-safe)
- Use a third-party `Predict` subclass that injects adapter via `settings.context()` (what `dspy-template-adapter` does today)

### What users want

```python
# Global default
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"), adapter=ChatAdapter())

# Per-module override — propagates to all nested predictors
outer = MyPipeline()
outer.lm = dspy.LM("anthropic/claude-sonnet-4-20250514")
outer.adapter = XMLAdapter()

# Per-predictor override — takes priority over module-level
outer.critic.lm = dspy.LM("openai/o1")
outer.critic.adapter = JSONAdapter()

# When outer() runs:
#   outer.critic uses o1 + JSONAdapter (its own explicit settings)
#   everything else uses claude + XMLAdapter (from outer's scope)
```

This is especially important for:
- **Multi-model pipelines** (cheap LM for drafting, expensive for judging)
- **Per-predictor prompt control** (different adapters for different signatures)
- **Agentic systems** where sub-agents have different LM/adapter requirements
- **Template adapters** that need per-predictor binding without subclassing Predict

## Proposed changes

### 1. `Predict`: add `self.adapter` (parallel to existing `self.lm`)

```python
# dspy/predict/predict.py

class Predict(Module, Parameter):
    def reset(self):
        self.lm = None
        self.adapter = None      # ← NEW
        self.traces = []
        self.train = []
        self.demos = []

    def forward(self, **kwargs):
        lm, config, signature, demos, kwargs = self._forward_preprocess(**kwargs)

        # ── Changed: resolve adapter like LM ──
        adapter = self.adapter or settings.adapter or ChatAdapter()

        if self._should_stream():
            with settings.context(caller_predict=self):
                completions = adapter(lm, lm_kwargs=config, signature=signature, demos=demos, inputs=kwargs)
        else:
            with settings.context(send_stream=None):
                completions = adapter(lm, lm_kwargs=config, signature=signature, demos=demos, inputs=kwargs)

        return self._forward_postprocess(completions, signature, **kwargs)

    # Same change in aforward()
```

This is a minimal, backward-compatible change. `self.adapter` defaults to `None`, so existing code that relies on `settings.adapter` or `ChatAdapter()` is unaffected.

### 2. `Module`: context-scoped LM/adapter propagation

```python
# dspy/primitives/module.py

class Module(BaseModule, metaclass=ProgramMeta):
    def _base_init(self):
        self._compiled = False
        self.callbacks = []
        self.history = []
        self._module_lm = None        # ← NEW (underscore to avoid conflict with Predict.lm)
        self._module_adapter = None   # ← NEW

    @with_callbacks
    def __call__(self, *args, **kwargs):
        caller_modules = settings.caller_modules or []
        caller_modules = list(caller_modules)
        caller_modules.append(self)

        # ── NEW: build context overrides from module-level settings ──
        ctx = {"caller_modules": caller_modules}
        if self._module_lm is not None:
            ctx["lm"] = self._module_lm
        if self._module_adapter is not None:
            ctx["adapter"] = self._module_adapter

        with settings.context(**ctx):
            if settings.track_usage and ...:
                ...
            return self.forward(*args, **kwargs)

    # Same change in acall()

    # ── NEW: properties for clean access ──
    @property
    def lm(self):
        return self._module_lm

    @lm.setter
    def lm(self, value):
        self._module_lm = value

    @property
    def adapter(self):
        return self._module_adapter

    @adapter.setter
    def adapter(self, value):
        self._module_adapter = value

    # ── NEW: eager setter (parallel to existing set_lm) ──
    def set_adapter(self, adapter):
        """Set adapter on all nested predictors."""
        for _, param in self.named_predictors():
            param.adapter = adapter
```

When `outer.__call__` runs and `outer._module_lm` is set, it pushes that LM into `settings.context(lm=...)`. Since `Predict._forward_preprocess` already does `self.lm or settings.lm`, and `settings.lm` checks thread-local overrides first, the module's LM is visible to all nested predictors — unless they have their own `self.lm` set.

### 3. `Predict.lm` / `Module.lm` MRO consideration

`Predict` inherits from `Module`. If `Module.lm` becomes a property, it will shadow `Predict`'s plain `self.lm` attribute. Two approaches:

**Option A (recommended):** Use different internal names. Module uses `_module_lm` / `_module_adapter` and exposes via properties. Predict keeps its existing `self.lm` attribute. Since attribute lookup finds instance attributes before class properties, Predict's `self.lm = None` (set in `reset()`) still takes priority. The Module-level property only applies to non-Predict Module subclasses.

**Option B:** Unify — make Predict use the property too, remove `self.lm` from `reset()`, adjust `_forward_preprocess` to read `self._module_lm`. More consistent but touches more code.

### 4. Fix other hardcoded `settings.adapter` references

These places read `settings.adapter` directly, bypassing any per-module/predictor override:

| File | Line | Current code |
|---|---|---|
| `predict/predict.py` | 191, 205 | `adapter = settings.adapter or ChatAdapter()` |
| `predict/react.py` | 92 | `adapter = dspy.settings.adapter or dspy.ChatAdapter()` |
| `predict/refine.py` | 104 | `adapter = dspy.settings.adapter or dspy.ChatAdapter()` |
| `streaming/streaming_listener.py` | 116+ | `settings.adapter.__class__.__name__` |

The `predict.py` change is covered by change #1. For `react.py` and `refine.py`, the same pattern applies — resolve from `self.adapter or settings.adapter or ChatAdapter()` where `self` is the relevant predictor.

The streaming listener is trickier because it doesn't have a predictor reference. Since module-level overrides are pushed into `settings.context()`, `settings.adapter` will already reflect the correct adapter at call time. No change needed there.

## How nesting composes

```python
dspy.configure(lm=gpt4, adapter=ChatAdapter())

outer = Outer()
outer.lm = claude               # propagates via context
outer.inner.predict.lm = gemini  # this specific predictor only
outer.inner.predict.adapter = XMLAdapter()  # this specific predictor only

outer()
```

```
outer.__call__()
  └─ settings.context(lm=claude)          # pushed by Module.__call__
      │
      ├─ inner.__call__()
      │   └─ settings.context(...)        # inner has no override, no-op
      │       └─ inner.predict.forward()
      │           ├─ lm = self.lm (gemini) or settings.lm (claude) → gemini ✓
      │           └─ adapter = self.adapter (XML) or settings.adapter (Chat) → XML ✓
      │
      └─ outer.summarize.forward()
          ├─ lm = self.lm (None) or settings.lm (claude) → claude ✓
          └─ adapter = self.adapter (None) or settings.adapter (Chat) → Chat ✓
```

## Backward compatibility

- `dspy.configure(lm=..., adapter=...)` continues to work identically
- `predict.lm = ...` continues to work identically
- `module.set_lm(lm)` continues to work (eager per-predictor set)
- New: `module.lm = ...` sets scoped propagation (context-based)
- New: `predict.adapter = ...` and `module.adapter = ...`
- Code that never sets per-module/predictor adapter sees no behavior change

## Non-goals

- Adapter serialization in `dump_state`/`load_state` — adapters may not be serializable. Document that adapter must be re-attached after load.
- Changing optimizer behavior — optimizers that pass explicit `lm` kwargs already work with the resolution chain.

## Related work

- `dspy-template-adapter` (`Predict` subclass) already implements per-predictor adapter via `settings.context()` — this proposal promotes that pattern into core DSPy.
- `dspy-session` manages per-session state and would benefit from per-module adapter scoping for composed multi-agent systems.
