import os
import pathlib
import pytest

import numpy as np  # type: ignore
import opendp.smartnoise.core as sn  # type: ignore
import pandas as pd  # type: ignore
from eeprivacy.mechanisms import LaplaceMechanism
from eeprivacy.operations import (
    PrivateClampedMean,
    PrivateClampedSum,
    PrivateHistogram,
    PrivateVectorClampedMeanGaussian,
)

HERE = pathlib.Path(__file__).parent.absolute()
TEST_DATA_PATH = os.path.join(HERE, "data", "PUMS_california_demographics_1000.csv")
TEST_DATA_COLUMNS = ["age", "sex", "educ", "race", "income", "married", "pid"]


def test_private_clamped_mean():
    # TODO: flesh out this test further. Or maybe it's tested in a stochastic
    #       testing harness?
    op = PrivateClampedMean(lower_bound=0, upper_bound=100)
    op.execute(values=[1, 2, 3], epsilon=0.1)


def test_private_clamped_mean_helpers():
    # Compute the CI with smartnoise
    with sn.Analysis() as analysis:
        data = sn.Dataset(path=TEST_DATA_PATH, column_names=TEST_DATA_COLUMNS)
        D = sn.to_float(data["age"])
        D_tilde = sn.resize(sn.clamp(data=D, lower=0.0, upper=100.0), number_rows=1000,)
        release = sn.dp_mean(data=sn.impute(D_tilde), privacy_usage={"epsilon": 1.0})
    smartnoise_ci = release.get_accuracy(0.05)

    # Compute the CI with eeprivacy
    op = PrivateClampedMean(lower_bound=0, upper_bound=100)
    eeprivacy_ci = op.confidence_interval(epsilon=1, N=1000, confidence=0.95)

    # Compare computed confidence intervals
    assert pytest.approx(smartnoise_ci, abs=0.001) == eeprivacy_ci

    smartnoise_epsilon = release.from_accuracy(value=1, alpha=0.05)[0]["epsilon"]
    eeprivacy_epsilon = op.epsilon_for_confidence_interval(
        target_ci=1, N=1000, confidence=0.95
    )

    # Compare computed epsilons for confidence interval
    assert pytest.approx(smartnoise_epsilon, abs=0.001) == eeprivacy_epsilon


def test_private_histogram_helpers():
    op = PrivateHistogram(bins=[], max_individual_contribution=1)
    histogram_ci = op.confidence_interval(epsilon=1, confidence=0.95)
    laplace_ci = LaplaceMechanism.confidence_interval(
        epsilon=1, sensitivity=1, confidence=0.95
    )
    assert histogram_ci == laplace_ci


def test_private_vector_clamped_mean_gaussian_helpers():
    pass


def test_private_clamped_sum_helpers():
    # Compute the CI with smartnoise
    with sn.Analysis() as analysis:
        data = sn.Dataset(path=TEST_DATA_PATH, column_names=TEST_DATA_COLUMNS)
        D = sn.to_float(data["age"])
        D_tilde = sn.resize(sn.clamp(data=D, lower=0.0, upper=100.0), number_rows=1000,)
        release = sn.dp_sum(data=sn.impute(D_tilde), privacy_usage={"epsilon": 1.0})
    smartnoise_ci = release.get_accuracy(0.05)

    op = PrivateClampedSum(lower_bound=0, upper_bound=100)
    eeprivacy_ci = op.confidence_interval(epsilon=1, confidence=0.95)

    assert pytest.approx(smartnoise_ci, abs=0.001) == eeprivacy_ci
