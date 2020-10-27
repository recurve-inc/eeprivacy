import os
import pathlib
import pytest

import numpy as np  # type: ignore
import opendp.whitenoise.core as wn  # type: ignore
import pandas as pd  # type: ignore
from eeprivacy.operations import PrivateClampedMean

HERE = pathlib.Path(__file__).parent.absolute()
TEST_DATA_PATH = os.path.join(HERE, "data", "PUMS_california_demographics_1000.csv")
TEST_DATA_COLUMNS = ["age", "sex", "educ", "race", "income", "married", "pid"]


def test_private_clamped_mean():
    # TODO: flesh out this test further. Or maybe it's tested in a stochastic
    #       testing harness?
    op = PrivateClampedMean(lower_bound=0, upper_bound=100)
    op.execute(values=[1, 2, 3], epsilon=0.1)


def test_private_clamped_mean_helpers():
    # Compute the CI with whitenoise
    with wn.Analysis() as analysis:
        data = wn.Dataset(path=TEST_DATA_PATH, column_names=TEST_DATA_COLUMNS)
        D = wn.to_float(data["age"])
        D_tilde = wn.resize(wn.clamp(data=D, lower=0.0, upper=100.0), number_rows=1000,)
        release = wn.dp_mean(data=wn.impute(D_tilde), privacy_usage={"epsilon": 1.0})
    whitenoise_ci = release.get_accuracy(0.05)

    # Compute the CI with eeprivacy
    op = PrivateClampedMean(lower_bound=0, upper_bound=100)
    eeprivacy_ci = op.confidence_interval(epsilon=1, N=1000, confidence=0.95)

    # Compare computed confidence intervals
    assert pytest.approx(whitenoise_ci, abs=0.001) == eeprivacy_ci

    whitenoise_epsilon = release.from_accuracy(value=1, alpha=0.05)[0]["epsilon"]
    eeprivacy_epsilon = op.epsilon_for_confidence_interval(
        target_ci=1, N=1000, confidence=0.95
    )

    # Compare computed epsilons for confidence interval
    assert pytest.approx(whitenoise_epsilon, abs=0.001) == eeprivacy_epsilon


def test_private_histogram_helpers():
    pass


def test_private_vector_clamped_mean_gaussian_helpers():
    pass
