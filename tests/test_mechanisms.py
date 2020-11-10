import os
import pathlib
import pytest

import numpy as np  # type: ignore
import opendp.smartnoise.core as sn  # type: ignore
import pandas as pd  # type: ignore
from eeprivacy.mechanisms import LaplaceMechanism, GaussianMechanism

HERE = pathlib.Path(__file__).parent.absolute()
TEST_DATA_PATH = os.path.join(HERE, "data", "PUMS_california_demographics_1000.csv")
TEST_DATA_COLUMNS = ["age", "sex", "educ", "race", "income", "married", "pid"]


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


# def test_gaussian_mechanism_confidence_interval():
#     ci = GaussianMechanism.confidence_interval(
#         epsilon=0.1, sensitivity=1, confidence=0.95, delta=1e-12
#     )

#     # Exact CI computed using scipy.stats.norm.ppf
#     assert pytest.approx(ci, abs=0.001) == 146.2878

#     # Compare to smartnoise implementation
#     eeprivacy_ci = GaussianMechanism.confidence_interval(
#         epsilon=0.1, sensitivity=(100 / 1000), confidence=0.95, delta=0.1
#     )

#     with sn.Analysis() as analysis:
#         data = sn.Dataset(path=TEST_DATA_PATH, column_names=TEST_DATA_COLUMNS)
#         D = sn.to_float(data["age"])
#         D_tilde = sn.clamp(data=D, lower=0.0, upper=200.0)
#         release = sn.dp_mean(
#             data=sn.impute(D_tilde),
#             mechanism="Gaussian",
#             privacy_usage={"epsilon": 0.1, "delta": 0.1},
#         )

#     # 0.1:  9.04083
#     # 0.05: 4.520418818601477
#     # 0.025: 2.260209

#     # delta 0.1 epsilon 0.1: 9.04083
#     # delta 0.1 epsilon 0.2: 2.0400177

#     # rows and upper have no effect

#     smartnoise_ci = release.get_accuracy(0.05)

#     print(eeprivacy_ci, smartnoise_ci)

#     assert False

#     # Validate the reverse holds
#     epsilon = GaussianMechanism.epsilon_for_confidence_interval(
#         target_ci=ci, sensitivity=1, confidence=0.95, delta=1e-12
#     )
#     assert pytest.approx(epsilon, abs=0.001) == 0.1
