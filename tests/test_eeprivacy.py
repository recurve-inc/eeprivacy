import numpy as np  # type: ignore
import pytest
from eeprivacy.mechanisms import LaplaceMechanism, GaussianMechanism


def test_laplace_mechanism_invocation():
    """Test a few variations of arguments for the Laplace Mechanism"""
    LaplaceMechanism.execute(value=0, epsilon=0.1, sensitivity=1)
    LaplaceMechanism.execute(values=[], epsilon=0.1, sensitivity=1)
    LaplaceMechanism.execute(values=[0, 1], epsilon=0.1, sensitivity=1)


def test_laplace_mechanism_confidence_interval():
    ci = LaplaceMechanism.confidence_interval(
        epsilon=0.1, sensitivity=1, confidence=0.95
    )

    # Exact CI computed using scipy.stats.laplace.ppf
    assert pytest.approx(ci, abs=0.001) == 29.9573

    # epsilon = LaplaceMechanism.epsilon_for_confidence_interval(
    #     target_ci=ci, sensitivity=1, confidence=0.95
    # )

    # ci = laplace_mechanism.confidence_interval(epsilon=)


# def test_compute_laplace_epsilon():
#     assert laplace_mechanism_epsilon_for_confidence_interval(
#         5, 1, 0.95
#     ) == pytest.approx(0.32188, abs=0.01)


# def test_laplace_mechanism_output_size():
#     res = laplace_mechanism(value=0, epsilon=1.0, sensitivity=1)
#     assert isinstance(res, float)

#     res = laplace_mechanism(values=[0], epsilon=1.0, sensitivity=1)
#     assert isinstance(res, np.ndarray)
#     assert len(res) == 1

#     res = laplace_mechanism(values=[0, 0, 0], epsilon=1.0, sensitivity=1)
#     assert len(res) == 3


# def stochastic_test(epsilon=1.0, lower=0, upper=0, tolerance=0.1):
#     A = laplace_mechanism(values=np.zeros(100000), epsilon=epsilon, sensitivity=1)
#     B = laplace_mechanism(values=np.ones(100000), epsilon=epsilon, sensitivity=1)

#     A = np.clip(A, lower, upper)
#     B = np.clip(B, lower, upper)

#     bins = np.linspace(lower, upper, num=50)

#     A, _ = np.histogram(A, bins=bins)
#     B, _ = np.histogram(B, bins=bins)

#     realized_epsilon = np.abs(np.log(A / B))

#     assert pytest.approx(epsilon, abs=tolerance) == np.max(realized_epsilon)


# def test_laplace_mechanism_vec():
#     stochastic_test(epsilon=1.0, lower=-2, upper=2, tolerance=0.8)
#     stochastic_test(epsilon=0.1, lower=-2, upper=2, tolerance=0.4)


# def test_private_histogram_with_laplace():
#     res = private_histogram_with_laplace(values=[0, 0, 0], bins=[0, 1], epsilon=1.0)


# def test_laplace_mechanism_confidence_interval():
#     pass


# def test_private_mean_with_laplace():
#     A = private_mean_with_laplace(values=[], epsilon=1.0, lower_bound=0, upper_bound=1)
#     assert A == 1
