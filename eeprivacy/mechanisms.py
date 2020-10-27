from typing import List, Union
import numpy as np  # type: ignore
from scipy.special import erfinv  # type: ignore


class Mechanism:
    pass


class LaplaceMechanism(Mechanism):
    """
    The Laplace Mechanism.
    """

    @staticmethod
    def scale(*, sensitivity: float, epsilon: float):
        return sensitivity / epsilon

    @staticmethod
    def execute(*, value: float, epsilon: float, sensitivity: float,) -> float:
        """
        Run the Laplace Mechanism, adding noise to `value` to realize differential
        private at `epsilon` for the provided `sensitivity`.
        """
        b = LaplaceMechanism.scale(sensitivity=sensitivity, epsilon=epsilon)
        return value + np.random.laplace(0, b)

    @staticmethod
    def execute_batch(
        *, values: List[float], epsilon: float, sensitivity: float,
    ) -> List[float]:
        """
        Run the Laplace Mechanism, adding noise to `value` to realize differential
        private at `epsilon` for the provided `sensitivity`.

        Runs the Laplace Mechanism multiple times, once for each item in the list.
        """
        b = LaplaceMechanism.scale(sensitivity=sensitivity, epsilon=epsilon)
        return values + np.random.laplace(0, b, size=len(values))

    @staticmethod
    def confidence_interval(
        *, epsilon: float, sensitivity: float, confidence: float = 0.95
    ) -> float:
        """Determine the two-sided confidence interval for a given privacy parameter.
        """
        b = sensitivity / epsilon

        # Convert the ``confidence`` into a quantile for two-sided error.
        # For example, for a 95% confidence, 2.5% of values will be below
        # the true value and 2.5% above. We want the 97.5% quantile so that
        # when we report +/- CI, it covers 95% of the outcomes.
        quantile = 1.0 - (1.0 - confidence) / 2.0

        if quantile <= 0.5:
            Q = b * np.log(2 * quantile)
        else:
            Q = -b * np.log(2 - 2 * quantile)
        return Q

    @staticmethod
    def epsilon_for_confidence_interval(
        *, target_ci: float, sensitivity: float, confidence: float = 0.95
    ) -> float:
        """Determine the privacy parameter for a desired accuracy.
        """
        quantile = 1.0 - (1.0 - confidence) / 2.0
        Q = target_ci
        if quantile <= 0.5:
            epsilon = sensitivity * np.log(2 * quantile) / Q
        else:
            epsilon = -sensitivity * np.log(2 - 2 * quantile) / Q
        return epsilon


class GaussianMechanism(object):
    """
    The Gaussian Mechanism.
    """

    @staticmethod
    def scale(*, sensitivity: float, epsilon: float, delta: float) -> float:
        return sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon

    @staticmethod
    def confidence_interval(
        *, epsilon: float, delta: float, sensitivity: float, confidence: float = 0.95,
    ) -> float:
        """
        Return the confidence interval for the Gaussian Mechanism at a given
        `epsilon`, `delta`, and `sensitivity`.
        """

        sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
        Q = sigma * np.sqrt(2) * erfinv(confidence)
        return Q

    @staticmethod
    def execute(
        *, value: float, epsilon: float, delta: float, sensitivity: float,
    ) -> float:
        """
        Run the Gaussian Mechanism, adding noise to `value` to realize differential
        private at (`epsilon`, `delta`) for the provided `sensitivity`.
        """
        b = GaussianMechanism.scale(
            sensitivity=sensitivity, epsilon=epsilon, delta=delta
        )
        return value + np.random.normal(0, b)

    @staticmethod
    def execute_batch(
        *, values: List[float], epsilon: float, delta: float, sensitivity: float,
    ) -> List[float]:
        """
        Run the Gaussian Mechanism, adding noise to `value` to realize differential
        private at (`epsilon`, `delta`) for the provided `sensitivity`.

        Runs the Gaussian Mechanism multiple times, once for each item in the list.
        """
        b = GaussianMechanism.scale(
            sensitivity=sensitivity, epsilon=epsilon, delta=delta
        )
        return values + np.random.normal(0, b, size=len(values))

    @staticmethod
    def epsilon_for_confidence_interval(
        target_ci: float, sensitivity: float, delta: float, confidence: float = 0.95
    ) -> float:
        """
        Returns the ε for the Gaussian Mechanism that will produce outputs
        +/-`target_ci` at `confidence` for queries with `sensitivity` and `delta`.
        """
        quantile = 1.0 - (1.0 - confidence) / 2.0
        Q = target_ci
        sigma = Q / (np.sqrt(2) * erfinv(2 * quantile - 1))
        epsilon = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / sigma
        return epsilon
