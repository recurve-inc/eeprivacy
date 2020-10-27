eeprivacy
=========

Operations
----------

An `Operation` encapsulates the full workflow of common analytics tasks: running statistics (like means, histograms, and quantiles), reporting on accuracy, and supporting analysis of privacy/accuracy tradeoffs.

.. automodule:: eeprivacy.operations
	:members:

Mechanisms
----------

`Mechanisms` are differential privacy primitives. The raw implementations should be used with care by non-experts.

A Note on Confidence Intervals
##############################

The confidence intervals returned by `Mechanism` classes are two-sided.

For example, in the algorithm design helper function `epsilon_for_confidence_interval` the ε value returned for the Laplace Mechanism at the default `confidence` = 0.95 and `target_ci` = 5, the distribution of outputs for the returned ε would be the following::

      2.5% │      █      │ 2.5%
    ◀──────│      █      │──────▶
           │      █      │
           │     ███     │
           │     ███     │
           │     ███     │
           │     ███     │
           │     ███     │
           │    █████    │
           │   ███████   │
           │  █████████  │
           │█████████████│
        ███│█████████████│███
      ─────┼──────┬──────┼────
           │      │      │
                True
          -5    Count    5


.. automodule:: eeprivacy.mechanisms
	:members: