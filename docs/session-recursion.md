---
title: "Session Recursion"
---


```python
import dspy
from dspy_session import sessionify
```

```output:exec-1772023577541-ajopy
```


```python
lm = dspy.LM("groq/moonshotai/kimi-k2-instruct-0905")
lm('hello')
```

```output:exec-1772023582926-wdkky
Out[2]: ['Hello! How can I help you today?']
```


```python
dspy.configure(lm = lm)
```

```output:exec-1772023584383-nh12s
```


```python
prg = dspy.Predict('q -> a')
prg(q = 'what is the capital of france')
```

```output:exec-1772023586290-wz4pc
```


```python
s = sessionify(prg)
s(q = 'what is the capital of france')
```

```output:exec-1772023588694-8ekiz
[31m---------------------------------------------------------------------------[0m
[31mNameError[0m                                 Traceback (most recent call last)
[36mCell [32mIn[1], line 1[0m
[32m----> 1[0m s = [43msessionify[0m(prg)
[32m      2[0m s(q = [33m'what is the capital of france'[0m)

[31mNameError[0m: name 'sessionify' is not defined
[Error: name 'sessionify' is not defined]
```

```python

```

