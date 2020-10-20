import pytest
from eeprivacy.operations import PrivateClampedMean


def test_private_clamped_mean():
    op = PrivateClampedMean(lower_bound=0, upper_bound=1)

    op.execute()

    # op.confidence_interval(epsilon=0.1, N=100, confidence=0.95)

    # op.epsilon_for_confidence_interval(target_ci=)
