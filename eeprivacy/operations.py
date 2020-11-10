from typing import List
import numpy as np  # type: ignore
from eeprivacy.mechanisms import LaplaceMechanism, GaussianMechanism


class Operation:
    pass


class PrivateCount(Operation):
    def __init__(self):
        pass

    def execute(self, *, values, epsilon):
        pass


class PrivateClampedSum(Operation):
    """
    Compute a sum, bounding sensitivity with clamping. Employs the Laplace Mechanism.
    """

    def __init__(self, *, lower_bound: float, upper_bound: float):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def execute(self, *, values: List[float], epsilon: float) -> float:
        """
        Computes the mean of `values` privately using a clamped sum.

        Return:
            The private sum.
        """
        a = self.lower_bound
        b = self.upper_bound

        S = np.sum(np.clip(values, a, b))

        private_sum = S + np.random.laplace(0, (b - a) / epsilon)

        return private_sum

    def confidence_interval(self, *, epsilon, N, confidence=0.95):
        """Compute the two-sided confidence interval for the mean."""
        return LaplaceMechanism.confidence_interval(
            epsilon=epsilon,
            sensitivity=(self.upper_bound - self.lower_bound),
            confidence=confidence,
        )

    def epsilon_for_confidence_interval(self, *, target_ci, N, confidence=0.95):
        """Return epsilon for a desired confidence interval."""
        return LaplaceMechanism.epsilon_for_confidence_interval(
            target_ci=target_ci,
            sensitivity=(self.upper_bound - self.lower_bound),
            confidence=confidence,
        )


class PrivateClampedMean(Operation):
    """
    Compute a mean, bounding sensitivity with clamping. Employs the Laplace Mechanism.
    """

    def __init__(self, *, lower_bound: float, upper_bound: float):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def execute(self, *, values: List[float], epsilon: float) -> float:
        """
        Computes the mean of `values` privately using a clamped mean.

        Implements Algorithm 2.3 from [Li et al]

        Return:
            The private mean.

        References:
            - Li, N., Yang, W., Lyu, M., Su, D. (2016). Differential Privacy: From Theory to Practice. United States: Morgan & Claypool Publishers.
        """

        S = np.sum(values)
        C = len(values)
        a = self.lower_bound
        b = self.upper_bound

        if C == 0:
            p = [
                0.5 * np.exp(-epsilon / 2),
                0.5 * np.exp(-epsilon / 2),
                1 - np.exp(-epsilon / 2),
            ]

            private_mean = np.random.choice(
                [a, b, np.random.uniform(low=a, high=b)], p=p
            )
        else:
            A = S / C + np.random.laplace(0, (b - a) / epsilon) / C

            if A < a:
                private_mean = a
            elif A > b:
                private_mean = b
            else:
                private_mean = A

        return private_mean

    def confidence_interval(self, *, epsilon, N, confidence=0.95):
        """Compute the two-sided confidence interval for the mean."""
        return LaplaceMechanism.confidence_interval(
            epsilon=epsilon,
            sensitivity=(self.upper_bound - self.lower_bound) / N,
            confidence=confidence,
        )

    def epsilon_for_confidence_interval(self, *, target_ci, N, confidence=0.95):
        """Return epsilon for a desired confidence interval."""
        return LaplaceMechanism.epsilon_for_confidence_interval(
            target_ci=target_ci,
            sensitivity=(self.upper_bound - self.lower_bound) / N,
            confidence=confidence,
        )


class PrivateVectorClampedMeanGaussian(Operation):
    """
    A vector mean computes the elementwise mean of a vector-valued dataset. That is,
    a dataset in which each row is a vector of values corresponding to each
    individual. For example::

        six_hourly_energy_consumptions = [
            [4.2, 11.1, 14.1, 3.2],
            [1.4, 4.5.1, 4.8, 1.6],
            ...
        ]

    This implementation requires the size of the dataset upfront (either pass it 
    exactly if it is not private or compute with a private count).

    With the Gaussian Mechanism, noise is scaled to the L2 norm of the dataset.

    Args:
        lower_bound: Lower bound of input data
        upper_bound: Upper bound of input data
        k: Size of vectors in dataset
        N: Number of elements in dataset
    """

    def __init__(
        self, *, lower_bound: float, upper_bound: float, k: float, N: float,
    ):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.k = k
        self.N = N

    @property
    def sensitivity(self):
        return (self.upper_bound - self.lower_bound) / self.N * np.sqrt(self.k)

    def execute(self, *, vectors: List[float], epsilon: float, delta: float) -> float:
        """
        Computes the elementwise mean of `vectors` privately using a clamped sum and
        exact count with the Gaussian Mechanism.
        """

        lower_bound = self.lower_bound
        upper_bound = self.upper_bound
        k = self.k

        clamped_vectors = np.array(
            [np.clip(v, lower_bound, upper_bound) for v in vectors]
        )

        exact_sum = np.sum(clamped_vectors, axis=0)

        sensitivity = (upper_bound - lower_bound) * np.sqrt(k)

        private_sum = GaussianMechanism.execute(
            value=exact_sum, epsilon=epsilon, sensitivity=sensitivity, delta=delta
        )

        return private_sum / self.N

    def confidence_interval(
        self, *, epsilon: float, delta: float, confidence: float = 0.95
    ) -> float:
        """Compute the two-sided confidence interval for the mean."""
        return GaussianMechanism.confidence_interval(
            epsilon=epsilon,
            delta=delta,
            confidence=confidence,
            sensitivity=self.sensitivity,
        )

    def epsilon_for_confidence_interval(
        self, *, target_ci: float, delta: float, confidence: float = 0.95
    ) -> float:
        """Return epsilon for a desired confidence interval."""
        return GaussianMechanism.epsilon_for_confidence_interval(
            target_ci=target_ci,
            delta=delta,
            confidence=confidence,
            sensitivity=self.sensitivity,
        )


class PrivateVectorClampedMeanLaplace(Operation):
    """
    A simple implementation of the mean operation that requires the size of the
    dataset up front (either pass it exactly if it is not private or compute with
    a private count).

    With the Laplace Mechanism, noise is scaled to the L1 norm of the dataset.

    Args:
        lower_bound: Lower bound of input data
        upper_bound: Upper bound of input data
        k: Size of vectors in dataset
        N: Number of elements in dataset
    """

    def __init__(
        self, *, lower_bound: float, upper_bound: float, k: float, N: float,
    ):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.k = k
        self.N = N

    @property
    def sensitivity(self) -> float:
        return (self.upper_bound - self.lower_bound) / self.N * self.k

    def execute(self, *, vectors: List[float], epsilon: float, delta: float) -> float:
        """
        Computes the mean of `vectors` privately using a clamped sum and
        exact count with the Laplace Mechanism.
        """

        lower_bound = self.lower_bound
        upper_bound = self.upper_bound

        clamped_vectors = np.array(
            [np.clip(v, lower_bound, upper_bound) for v in vectors]
        )

        exact_sum = np.sum(clamped_vectors, axis=0)

        sensitivity = (upper_bound - lower_bound) * self.k

        private_sum = LaplaceMechanism.execute(
            value=exact_sum, epsilon=epsilon, sensitivity=sensitivity
        )

        return private_sum / self.N

    def confidence_interval(self, *, epsilon, confidence=0.95) -> float:
        return LaplaceMechanism.confidence_interval(
            epsilon=epsilon, sensitivity=self.sensitivity, confidence=confidence,
        )

    def epsilon_for_confidence_interval(
        self, *, target_ci, delta, confidence=0.95
    ) -> float:
        return LaplaceMechanism.epsilon_for_confidence_interval(
            target_ci=target_ci, confidence=confidence, sensitivity=self.sensitivity,
        )


class PrivateHistogram(Operation):
    """
    Compute a private histogram with the Laplace Mechanism.

    Args:
        bins: Edges of bins for results. These must be specified up-front
            otherwise information will be leaked by the choice of bins. `Read more`_
        max_individual_contribution: Maximum count any individual in the dataset
            could contribute to the histogram. Normally `1`.

    .. _Read more:
        https://desfontain.es/privacy/almost-differential-privacy.html
    """

    def __init__(self, *, bins: List[float], max_individual_contribution: int = 1):
        self.bins = bins
        self.max_individual_contribution = max_individual_contribution

    def execute(self, *, values: List[float], epsilon: float) -> List[float]:
        """
        Computes the histogram of `values` privately.

        Values are clamped to the bound specified by ``bins``.

        Return: 
            A list of private counts, one for each bin.
        """
        sensitivity = self.max_individual_contribution
        bins = np.sort(self.bins)
        values = np.clip(values, bins[0], bins[-1])
        counts, _ = np.histogram(values, bins=bins)
        noisy_counts = LaplaceMechanism.execute_batch(
            values=counts, epsilon=epsilon, sensitivity=sensitivity
        )
        return noisy_counts

    def confidence_interval(self, *, epsilon: float, confidence: float = 0.95) -> float:
        """Return the confidence interval for each bar of the histogram."""
        return LaplaceMechanism.confidence_interval(
            epsilon=epsilon,
            sensitivity=self.max_individual_contribution,
            confidence=confidence,
        )


class PrivateQuantile(Operation):
    """Find quantiles privately with the `Report Noisy Max` mechanism.

    Args:
        options: List of possible outputs
        max_individual_contribution: Normally 1, unless an individual can contribute
            multiple points to the dataset.
    """

    def __init__(self, *, options: List[float], max_individual_contribution: int = 1):
        self.options = options
        self.max_individual_contribution = max_individual_contribution

    def execute(self, *, values: List[float], epsilon: float, quantile: float):
        """
        Computes the quantile of a list of values privately.

        Implemented using Report Noisy Max (Claim 3.9) of [Dwork and Roth].

        Args:
            values: Dataset
            epsilon: Privacy parameter
            quantile: Quantile to return (between 0 and 1)

        References:
        - Dwork, C., Roth, A., 2013. The Algorithmic Foundations of Differential Privacy. FNT in Theoretical Computer Science 9, 211–407. https://doi.org/10.1561/0400000042
        """
        N = len(values)

        sensitivity = self.max_individual_contribution

        def score(values, option):
            count = np.sum(values < option)
            return -np.abs(count - N * quantile)

        scores = [score(values, option) for option in self.options]
        noisy_scores = LaplaceMechanism.execute_batch(
            values=scores, epsilon=epsilon, sensitivity=sensitivity
        )
        max_idx = np.argmax(noisy_scores)
        return self.options[max_idx]
