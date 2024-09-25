"""Tests PHSafe tabulation variance."""

# Copyright 2024 Tumult Labs
# 
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
# 
#        http://www.apache.org/licenses/LICENSE-2.0
# 
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import copy
from io import StringIO
from typing import Callable, Dict, Mapping

import pandas as pd
import pytest
from pyspark.sql import DataFrame
from pyspark.sql.session import SparkSession

from tmlt.common.pyspark_test_tools import pyspark  # pylint: disable=unused-import
from tmlt.phsafe.input_processing import (
    PRIVACY_BUDGET_KEY,
    PRIVACY_DEFN_FLAG,
    READER_FLAG,
    STATE_FILTER_FLAG,
    TAU_KEY,
)
from tmlt.phsafe.utils import (
    _dp_denom,
    _dp_double,
    _dp_standard,
    _zc_denom,
    _zc_double,
    _zc_standard,
    validate_tabulation_variance,
)

PRIVACY_BUDGETS: Dict[str, Dict[str, float]] = {
    "PH1_num": {
        "usa_A-G": 0.33,
        "usa_H,I": 0.33,
        "usa_*": 0.33,
        "state_A-G": 0.33,
        "state_H,I": 0.33,
        "state_*": 0.33,
    },
    "PH1_denom": {
        "usa_A-G": 0.33,
        "usa_H,I": 0.33,
        "usa_*": 0.33,
        "state_A-G": 0.33,
        "state_H,I": 0.33,
        "state_*": 0.33,
    },
    "PH2": {"usa_*": 1.0, "state_*": 1.0},
    "PH3": {
        "usa_A-G": 0.5,
        "usa_H,I": 0.5,
        "usa_*": 0.5,
        "state_A-G": 0.5,
        "state_H,I": 0.5,
        "state_*": 0.5,
    },
    "PH4": {
        "usa_A-G": 0.5,
        "usa_H,I": 0.5,
        "usa_*": 0.5,
        "state_A-G": 0.5,
        "state_H,I": 0.5,
        "state_*": 0.5,
    },
    "PH5_denom": {
        "usa_A-G": 0.5,
        "usa_H,I": 0.5,
        "usa_*": 0.5,
        "state_A-G": 0.5,
        "state_H,I": 0.5,
        "state_*": 0.5,
    },
    "PH6": {"usa_*": 1.0, "state_*": 1.0},
    "PH7": {
        "usa_A-G": 0.5,
        "usa_H,I": 0.5,
        "usa_*": 0.5,
        "state_A-G": 0.5,
        "state_H,I": 0.5,
        "state_*": 0.5,
    },
    "PH8_denom": {
        "usa_A-G": 0.5,
        "usa_H,I": 0.5,
        "usa_*": 0.5,
        "state_A-G": 0.5,
        "state_H,I": 0.5,
        "state_*": 0.5,
    },
}

TAUS = {"PH1_num": 5, "PH2": 10, "PH3": 5, "PH4": 5, "PH6": 5, "PH7": 5}

CONFIG = {
    STATE_FILTER_FLAG: ["01", "02", "44", "51"],
    READER_FLAG: "csv",
    PRIVACY_BUDGET_KEY: PRIVACY_BUDGETS,
    PRIVACY_DEFN_FLAG: "puredp",
    TAU_KEY: TAUS,
}

VALIDATION_PUREDP_CONFIG = {
    "PH1_num": {"budget": "PH1_num", "*": "dp_standard"},
    "PH1_denom": {"budget": "PH1_denom", "*": "dp_denom"},
    "PH2": {"budget": "PH2", "*": "dp_standard"},
    "PH3": {"budget": "PH3", "*": "dp_standard"},
    "PH4": {"budget": "PH4", "*": "dp_standard"},
    "PH5_num": {"budget": "PH4", "*": "dp_standard"},
    "PH5_denom": {"budget": "PH5_denom", "*": "dp_denom"},
    "PH6": {"budget": "PH6", "*": "dp_standard"},
    "PH7": {"budget": "PH7", "*": "dp_standard"},
    "PH8_num": {"budget": "PH7", "2": "dp_double", "3": "dp_standard"},
    "PH8_denom": {"budget": "PH8_denom", "*": "dp_denom"},
}

VALIDATION_ZCDP_CONFIG = {
    "PH1_num": {"budget": "PH1_num", "*": "zc_standard"},
    "PH1_denom": {"budget": "PH1_denom", "*": "zc_denom"},
    "PH2": {"budget": "PH2", "*": "zc_standard"},
    "PH3": {"budget": "PH3", "*": "zc_standard"},
    "PH4": {"budget": "PH4", "*": "zc_standard"},
    "PH5_num": {"budget": "PH4", "*": "zc_standard"},
    "PH5_denom": {"budget": "PH5_denom", "*": "zc_denom"},
    "PH6": {"budget": "PH6", "*": "zc_standard"},
    "PH7": {"budget": "PH7", "*": "zc_standard"},
    "PH8_num": {"budget": "PH7", "2": "zc_double", "3": "zc_standard"},
    "PH8_denom": {"budget": "PH8_denom", "*": "zc_denom"},
}


@pytest.fixture(scope="class")
def setup_tabulation_variance(spark, request):
    """Class fixture for tabulation variance tests"""
    # This dataset came from ph3
    puredf = """
    REGION_ID|REGION_TYPE|ITERATION_CODE|PH3_DATA_CELL|COUNT|NOISE_DISTRIBUTION|VARIANCE
    01|STATE|E|2|26|Two-Sided Geometric|1151.83335
    01|STATE|E|4|-26|Two-Sided Geometric|1151.83335
    01|STATE|G|7|7|Two-Sided Geometric|1151.83335
    """
    zcdf = """
    REGION_ID|REGION_TYPE|ITERATION_CODE|PH3_DATA_CELL|COUNT|NOISE_DISTRIBUTION|VARIANCE
    01|STATE|A|5|6|Discrete Gaussian|144
    01|STATE|B|6|10|Discrete Gaussian|144
    01|STATE|G|2|-25|Discrete Gaussian|144
    """
    # Errors thrown from tabulation causes a cascade of massive spark warn logs.
    # We temporarily turn them off for these sets of tests.
    request.cls.spark = SparkSession.builder.getOrCreate()
    request.cls.spark.sparkContext.setLogLevel("OFF")
    request.cls.puredf = spark.createDataFrame(pd.read_csv(StringIO(puredf), sep="|"))
    request.cls.zcdf = spark.createDataFrame(pd.read_csv(StringIO(zcdf), sep="|"))
    request.cls.tabulation = "PH3"
    request.cls.formulas = {
        "dp_denom": _dp_denom,
        "dp_double": _dp_double,
        "dp_standard": _dp_standard,
        "zc_denom": _zc_denom,
        "zc_double": _zc_double,
        "zc_standard": _zc_standard,
    }
    yield
    request.cls.spark.sparkContext.setLogLevel("WARN")
    request.cls.spark.stop()


@pytest.mark.usefixtures("spark")
@pytest.mark.usefixtures("setup_tabulation_variance")
class TestTabulationVariance:
    """Small tabulation tests to test tabulation variance correctness."""

    puredf: DataFrame
    zcdf: DataFrame
    tabulation: str
    formulas: Mapping[str, Callable[..., float]]

    def test_tabulation_variance(self):
        """Tests the success case for a regular tabulation variance."""
        okay = validate_tabulation_variance(
            tabulation_data=self.puredf,
            privacy_budget=PRIVACY_BUDGETS,
            budget_source=self.tabulation,
            tabulation=self.tabulation,
            tabulation_config=VALIDATION_PUREDP_CONFIG[self.tabulation],
            formulas=self.formulas,
            tau=CONFIG[TAU_KEY][self.tabulation],
        )
        assert okay, "tabulation unexpectedly failed"

    @pytest.mark.parametrize("tau", [1, 2, 3, 4, 6, 7, 8, 9])
    def test_tabulation_variance_wrong_tau(self, tau: int):
        """Tests the cases where the variance does not match with the tau."""
        okay = validate_tabulation_variance(
            tabulation_data=self.puredf,
            privacy_budget=PRIVACY_BUDGETS,
            budget_source=self.tabulation,
            tabulation=self.tabulation,
            tabulation_config=VALIDATION_PUREDP_CONFIG[self.tabulation],
            formulas=self.formulas,
            tau=tau,
        )
        assert not okay, f"tabulation unexpectedly passed on tau {tau}"

    @pytest.mark.parametrize("budget", [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0])
    def test_tabulation_variance_wrong_budget(self, budget: float):
        """Tests the cases where the variance does not match with the budget."""
        privacy_budget = copy.deepcopy(PRIVACY_BUDGETS)
        for regions in privacy_budget[self.tabulation].keys():
            privacy_budget[self.tabulation][regions] = budget
        okay = validate_tabulation_variance(
            tabulation_data=self.puredf,
            privacy_budget=privacy_budget,
            budget_source=self.tabulation,
            tabulation=self.tabulation,
            tabulation_config=VALIDATION_PUREDP_CONFIG[self.tabulation],
            formulas=self.formulas,
            tau=TAUS[self.tabulation],
        )
        assert not okay, f"tabulation unexpectedly passed on budget {budget}"
