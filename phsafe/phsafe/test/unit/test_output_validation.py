"""Tests input validation on spark dataframes."""

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

# pylint: disable=no-name-in-module

import io
import logging
import pkgutil
from typing import Dict, List

import pandas as pd
import pytest
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit
from pyspark.sql.types import DoubleType, LongType, StringType, StructField, StructType

from tmlt.common.pyspark_test_tools import pyspark  # pylint: disable=unused-import
from tmlt.phsafe import PHSafeOutput
from tmlt.phsafe.output_validation import validate_output
from tmlt.phsafe.paths import RESOURCES_PACKAGE_NAME
from tmlt.phsafe.utils import get_config_privacy_budget_dict


@pytest.mark.usefixtures("spark")
class TestOutputValidation:
    """Parameterized unit tests for output validation."""

    @pytest.fixture
    def output_sdfs(self, spark) -> PHSafeOutput:
        """Return output dataframes for testing."""
        ph2_sdf = spark.createDataFrame(
            pd.read_csv(
                io.BytesIO(
                    pkgutil.get_data(  # type: ignore
                        "tmlt.phsafe", RESOURCES_PACKAGE_NAME + "/test/test1/PH2.csv"
                    )
                ),
                encoding="utf8",
                dtype={
                    "REGION_ID": object,
                    "REGION_TYPE": object,
                    "PH2_DATA_CELL": int,
                    "COUNT": object,
                },
            ),
            StructType(
                [
                    StructField("REGION_ID", StringType(), True),
                    StructField("REGION_TYPE", StringType(), True),
                    StructField("PH2_DATA_CELL", LongType(), True),
                    StructField("COUNT", StringType(), True),
                ]
            ),
        )
        ph2_sdf = ph2_sdf.withColumn(
            "NOISE_DISTRIBUTION", lit("Discrete Gaussian")
        ).withColumn("VARIANCE", lit(967.83335))

        ph3_sdf = spark.createDataFrame(
            pd.read_csv(
                io.BytesIO(
                    pkgutil.get_data(  # type: ignore
                        "tmlt.phsafe",
                        RESOURCES_PACKAGE_NAME
                        + "/test/test_outputs/PH3_with_variance_sampled.csv",
                    )
                ),
                encoding="utf8",
                dtype={
                    "REGION_ID": object,
                    "REGION_TYPE": object,
                    "ITERATION_CODE:": object,
                    "PH3_DATA_CELL": int,
                    "COUNT": object,
                    "NOISE_DISTRIBUTION": object,
                    "VARIANCE": float,
                },
            ),
            StructType(
                [
                    StructField("REGION_ID", StringType(), True),
                    StructField("REGION_TYPE", StringType(), True),
                    StructField("ITERATION_CODE", StringType(), True),
                    StructField("PH3_DATA_CELL", LongType(), True),
                    StructField("COUNT", StringType(), True),
                    StructField("NOISE_DISTRIBUTION", StringType(), True),
                    StructField("VARIANCE", DoubleType(), True),
                ]
            ),
        )

        return PHSafeOutput(PH2=ph2_sdf, PH3=ph3_sdf)

    @pytest.fixture
    def privacy_budgets(self) -> Dict[str, Dict[str, float]]:
        """Return privacy budgets for testing."""
        privacy_budget_dict = get_config_privacy_budget_dict(float(0))
        privacy_budget_dict["PH2"] = {
            geo_iteration: float(1)
            for geo_iteration in privacy_budget_dict["PH2"].keys()
        }

        # Non-uniform privacy budget: usa_H,I has higher budget (and
        # therefore a different variance)
        privacy_budget_dict["PH3"] = {
            geo_iteration: float(0.5)
            for geo_iteration in privacy_budget_dict["PH3"].keys()
        }
        privacy_budget_dict["PH3"]["usa_H,I"] = 0.6
        return privacy_budget_dict

    @pytest.fixture
    def filter_states(self) -> List[str]:
        """Return states to filter for testing."""
        return ["01", "44"]

    @pytest.fixture
    def tau(self) -> Dict[str, int]:
        """Return tau for testing."""
        return {"PH2": 10, "PH3": 5}

    def test_valid_output(
        self,
        output_sdfs: PHSafeOutput,
        privacy_budgets: Dict[str, Dict[str, float]],
        filter_states: List[str],
        tau: Dict[str, int],
        caplog,
    ):
        """Output is correct and states are filtered."""
        caplog.set_level(logging.INFO)
        okay = validate_output(
            output_sdfs=output_sdfs,
            privacy_budgets=privacy_budgets,
            filter_states=filter_states,
            tau=tau,
            privacy_defn="puredp",
        )
        assert okay, "\n".join(caplog.messages)
        for msg in [
            "tmlt.phsafe.output_validation",
            "Output validation successful.",
            "All output files are as expected.",
        ]:
            assert msg in caplog.text

    @pytest.mark.parametrize(
        "state_filter, df, error_message",
        [
            (
                ["72"],
                pd.DataFrame(
                    [
                        ["1", "USA", 3, 5, "Two-Sided Geometric", 23],
                        ["1", "USA", 4, 3, "Two-Sided Geometric", 23],
                        ["720000000", "STATE", 13, 8, "Two-Sided Geometric", 23],
                    ],
                    columns=[
                        "REGION_ID",
                        "REGION_TYPE",
                        "PH2_DATA_CELL",
                        "COUNT",
                        "NOISE_DISTRIBUTION",
                        "VARIANCE",
                    ],
                ),
                "Invalid values found in REGION_ID: ['720000000']",
            ),
            (
                ["72"],
                pd.DataFrame(
                    [
                        ["1", "USA", 7, -5, "Two-Sided Geometric", 23],
                        ["1", "USA", 14, -3, "Two-Sided Geometric", 23],
                        ["72", "STATE", 1, -8, "Two-Sided Geometric", 23],
                    ],
                    columns=[
                        "REGION_ID",
                        "REGION_TYPE",
                        "PH2_DATA_CELL",
                        "COUNT",
                        "NOISE_DISTRIBUTION",
                        "VARIANCE",
                    ],
                ),
                "Invalid values found in PH2_DATA_CELL: [14]",
            ),
            (
                ["72"],
                pd.DataFrame(
                    [
                        ["1", "USA", 7, -5, "Two-Sided Geometric", 23],
                        ["11", "STATE", 1, -8, "Two-Sided Geometric", 23],
                    ],
                    columns=[
                        "REGION_ID",
                        "REGION_TYPE",
                        "PH2_DATA_CELL",
                        "COUNT",
                        "NOISE_DISTRIBUTION",
                        "VARIANCE",
                    ],
                ),
                "Invalid output: Output contains states not in the config state_filter "
                "flag.",
            ),
        ],
    )
    def test_invalid_output(
        self,
        state_filter: List[str],
        df: pd.DataFrame,
        error_message: str,
        spark: SparkSession,
        caplog,
    ):
        """Output validation fails on invalid output and logs appropriate error."""
        sdf = spark.createDataFrame(df)
        privacy_budget_dict = get_config_privacy_budget_dict(float(0))
        privacy_budget_dict["PH2"] = {
            geo_iteration: float(1)
            for geo_iteration in privacy_budget_dict["PH2"].keys()
        }
        caplog.set_level(logging.ERROR)
        okay = validate_output(
            output_sdfs=PHSafeOutput(PH2=sdf),
            privacy_budgets=privacy_budget_dict,
            filter_states=state_filter,
            tau=None,
            privacy_defn="puredp",
        )
        assert not okay
        assert error_message in caplog.text
