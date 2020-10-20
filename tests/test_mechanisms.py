import pytest
from eeprivacy.mechanisms import LaplaceMechanism, GaussianMechanism


def test_laplace_mechanism_invocation():
    """Test a few variations of arguments for the Laplace Mechanism"""
    LaplaceMechanism.execute(value=0, epsilon=0.1, sensitivity=1)
    LaplaceMechanism.execute_batch(values=[], epsilon=0.1, sensitivity=1)
    LaplaceMechanism.execute_batch(values=[0, 1], epsilon=0.1, sensitivity=1)


def test_laplace_mechanism_confidence_interval():
    ci = LaplaceMechanism.confidence_interval(
        epsilon=0.1, sensitivity=1, confidence=0.95
    )

    # Exact CI computed using scipy.stats.laplace.ppf
    assert pytest.approx(ci, abs=0.001) == 29.9573

    # Validate the reverse holds
    epsilon = LaplaceMechanism.epsilon_for_confidence_interval(
        target_ci=ci, sensitivity=1, confidence=0.95
    )
    assert pytest.approx(epsilon, abs=0.001) == 0.1


def test_gaussian_mechanism_invocation():
    """Test a few variations of arguments for the Gaussian Mechanism"""
    GaussianMechanism.execute(value=0, epsilon=0.1, sensitivity=1, delta=1e-12)
    GaussianMechanism.execute_batch(values=[], epsilon=0.1, sensitivity=1, delta=1e-12)
    GaussianMechanism.execute_batch(
        values=[0, 1], epsilon=0.1, sensitivity=1, delta=1e-12
    )


def test_gaussian_mechanism_confidence_interval():
    ci = GaussianMechanism.confidence_interval(
        epsilon=0.1, sensitivity=1, confidence=0.95, delta=1e-12
    )

    # Exact CI computed using scipy.stats.norm.ppf
    assert pytest.approx(ci, abs=0.001) == 146.2878

    # Validate the reverse holds
    epsilon = GaussianMechanism.epsilon_for_confidence_interval(
        target_ci=ci, sensitivity=1, confidence=0.95, delta=1e-12
    )
    assert pytest.approx(epsilon, abs=0.001) == 0.1
