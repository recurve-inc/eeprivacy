
This is why we pursue the OO model:

* Awkwardness of long function names

```python
laplace_mechanism_epsilon_for_confidence_interval
gaussian_mechanism_sensitivity_for_mean
private_mean_with_gaussian_mechanism_epsilon_for_confidence_interval
```

* Confusing `sensitivity` (counts vs sums vs vector functions)

Instead, we get:

```python
private_mean = PrivateClampedMean(lower_bound=0, upper_bound=100,).execute(
    values=values, epsilon=0.1
)
```

It looks the same as this:

```python
private_mean_with_laplace(
    values=values, epsilon=epsilon, lower_bound=lower_bound, upper_bound=upper_bound
)
```

Except you also have the useful algorithm design functions attached:

```python
ci = private_mean.confidence_interval(epsilon=0.1)

epsilon = private_mean.epsilon_for_confidence_interval(
    target_ci=5, confidence=0.99
)
```

Why repeat epsilon in `execute` and `confidence_interval`?

Why not pass them as part of `__init__` for an `Operation`?

```python
def execute(self, vectors=None, epsilon=None, delta=None) -> float:
    pass


def confidence_interval(self, epsilon=None, delta=None, confidence=0.95) -> float:
    pass
```

Because `epsilon_for_confidence_interval` doesn't need them:


```python
def epsilon_for_confidence_interval(
    self, target_ci=None, delta=None, confidence=0.95
) -> float:
    pass
```

I wonder if we will find the mixing the concerns of algorithm execution
and algorithm design in the `Operation` class it a mistake? Are these
tasks so fundamentally different that it is better to implement them
separately? Or will it be convenient to have the unified interface? I
recall often flipping from design to execution in the same notebook.

