"""Tests for :mod:`accuracy_report`."""

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


import tempfile
from typing import Dict, Union
from unittest.mock import patch

import pandas as pd
import pytest
from typing_extensions import Literal

from tmlt.common.pyspark_test_tools import pyspark  # pylint: disable=unused-import
from tmlt.common.pyspark_test_tools import assert_frame_equal_with_sort
from tmlt.phsafe import PHSafeInput, PHSafeOutput
from tmlt.phsafe.accuracy_report import (
    aggregate_error_report,
    create_get_expected_error,
    create_phsafe_error_report,
    create_unaggregated_error_report,
)
from tmlt.phsafe.input_processing import PrivatePHSafeParameters


@pytest.mark.usefixtures("spark")
class TestAccuracyReport:
    """Tests for :mod:`~.accuracy_report` that use spark."""

    def test_create_unaggregated_error_report(self, spark):
        """Test for :func:`~.create_unaggregated_error_report`."""
        private_answer = PHSafeOutput(
            PH2=spark.createDataFrame(
                pd.DataFrame(
                    [
                        ["06", "STATE", 1, 1],
                        ["06", "STATE", 8, 1],
                        ["06", "STATE", 9, 1],
                        ["1", "USA", 1, 1],
                        ["1", "USA", 8, 1],
                        ["1", "USA", 9, 1],
                    ],
                    columns=["REGION_ID", "REGION_TYPE", "PH2_DATA_CELL", "COUNT"],
                )
            ),
            PH4=spark.createDataFrame(
                pd.DataFrame(
                    [
                        ["06", "STATE", "*", 1, 1],
                        ["06", "STATE", "*", 3, 1],
                        ["06", "STATE", "A", 1, 1],
                        ["06", "STATE", "A", 3, 1],
                        ["06", "STATE", "I", 1, 1],
                        ["06", "STATE", "I", 3, 1],
                        ["1", "USA", "*", 1, 1],
                        ["1", "USA", "*", 3, 1],
                        ["1", "USA", "A", 1, 1],
                        ["1", "USA", "A", 3, 1],
                        ["1", "USA", "I", 1, 1],
                        ["1", "USA", "I", 3, 1],
                    ],
                    columns=[
                        "REGION_ID",
                        "REGION_TYPE",
                        "ITERATION_CODE",
                        "PH4_DATA_CELL",
                        "COUNT",
                    ],
                )
            ),
        )
        nonprivate_answer = PHSafeOutput(
            PH2=spark.createDataFrame(
                pd.DataFrame(
                    [
                        ["06", "STATE", 1, 3],
                        ["06", "STATE", 8, 2],
                        ["06", "STATE", 9, 2],
                        ["1", "USA", 1, 2],
                        ["1", "USA", 8, 2],
                        ["1", "USA", 9, 2],
                    ],
                    columns=["REGION_ID", "REGION_TYPE", "PH2_DATA_CELL", "COUNT"],
                )
            ),
            PH4=spark.createDataFrame(
                pd.DataFrame(
                    [
                        ["06", "STATE", "*", 1, 4],
                        ["06", "STATE", "*", 3, 2],
                        ["06", "STATE", "A", 1, 2],
                        ["06", "STATE", "A", 3, 2],
                        ["06", "STATE", "I", 1, 2],
                        ["06", "STATE", "I", 3, 2],
                        ["1", "USA", "*", 1, 2],
                        ["1", "USA", "*", 3, 2],
                        ["1", "USA", "A", 1, 2],
                        ["1", "USA", "A", 3, 2],
                        ["1", "USA", "I", 1, 2],
                        ["1", "USA", "I", 3, 2],
                    ],
                    columns=[
                        "REGION_ID",
                        "REGION_TYPE",
                        "ITERATION_CODE",
                        "PH4_DATA_CELL",
                        "COUNT",
                    ],
                )
            ),
        )
        actual = create_unaggregated_error_report(private_answer, nonprivate_answer)
        expected = pd.DataFrame(
            [
                ["PH2", "06", "STATE", "*", 1, 2],
                ["PH2", "06", "STATE", "*", 8, 1],
                ["PH2", "06", "STATE", "*", 9, 1],
                ["PH2", "1", "USA", "*", 1, 1],
                ["PH2", "1", "USA", "*", 8, 1],
                ["PH2", "1", "USA", "*", 9, 1],
                ["PH4", "06", "STATE", "*", 1, 3],
                ["PH4", "06", "STATE", "*", 3, 1],
                ["PH4", "06", "STATE", "A", 1, 1],
                ["PH4", "06", "STATE", "A", 3, 1],
                ["PH4", "06", "STATE", "I", 1, 1],
                ["PH4", "06", "STATE", "I", 3, 1],
                ["PH4", "1", "USA", "*", 1, 1],
                ["PH4", "1", "USA", "*", 3, 1],
                ["PH4", "1", "USA", "A", 1, 1],
                ["PH4", "1", "USA", "A", 3, 1],
                ["PH4", "1", "USA", "I", 1, 1],
                ["PH4", "1", "USA", "I", 3, 1],
            ],
            columns=[
                "TABLE",
                "REGION_ID",
                "REGION_TYPE",
                "ITERATION_CODE",
                "DATA_CELL",
                "ERROR",
            ],
        )
        assert_frame_equal_with_sort(actual.toPandas(), expected)

    def test_aggregate_error_report(self, spark):
        """Test for :func:`~.aggregate_error_report`."""
        error_report = pd.DataFrame(
            [
                ["PH2", "01", "STATE", "*", 1, 0],
                ["PH2", "01", "STATE", "*", 1, 1],
                ["PH2", "02", "STATE", "*", 1, 2],
                ["PH2", "03", "STATE", "*", 1, 3],
                ["PH2", "04", "STATE", "*", 1, 4],
                ["PH2", "05", "STATE", "*", 1, 5],
                ["PH2", "01", "STATE", "*", 1, 6],
                ["PH2", "01", "STATE", "*", 1, 7],
                ["PH2", "01", "STATE", "*", 1, 8],
                ["PH2", "01", "STATE", "*", 1, 9],
                ["PH2", "01", "STATE", "*", 1, 10],
                ["PH2", "01", "STATE", "*", 8, 1],
                ["PH2", "01", "STATE", "A", 1, 2],
                ["PH2", "1", "USA", "*", 1, 3],
                ["PH4", "01", "STATE", "*", 1, 4],
            ],
            columns=[
                "TABLE",
                "REGION_ID",
                "REGION_TYPE",
                "ITERATION_CODE",
                "DATA_CELL",
                "ERROR",
            ],
        )
        actual = aggregate_error_report(spark.createDataFrame(error_report))
        expected = pd.DataFrame(
            [
                ["PH2", "STATE", "*", 1, 9.0, 11],
                ["PH2", "STATE", "*", 8, 1.0, 1],
                ["PH2", "STATE", "A", 1, 2.0, 1],
                ["PH2", "USA", "*", 1, 3.0, 1],
                ["PH4", "STATE", "*", 1, 4.0, 1],
            ],
            columns=[
                "TABLE",
                "REGION_TYPE",
                "ITERATION_CODE",
                "DATA_CELL",
                "EXPERIMENTAL_MOE",
                "COUNT",
            ],
        )
        assert_frame_equal_with_sort(actual.toPandas(), expected)

    def test_create_phsafe_error_report_mock_inputs(self, spark):
        """Test :func:`~.create_phsafe_error_report` with mock inputs."""
        private_answer = PHSafeOutput(
            PH2=spark.createDataFrame(
                pd.DataFrame(
                    [["06", "STATE", 1, 1], ["1", "USA", 1, 1]],
                    columns=["REGION_ID", "REGION_TYPE", "PH2_DATA_CELL", "COUNT"],
                )
            ),
            PH4=spark.createDataFrame(
                pd.DataFrame(
                    [["06", "STATE", "*", 1, 1], ["1", "USA", "*", 1, 1]],
                    columns=[
                        "REGION_ID",
                        "REGION_TYPE",
                        "ITERATION_CODE",
                        "PH4_DATA_CELL",
                        "COUNT",
                    ],
                )
            ),
        )
        nonprivate_answer = PHSafeOutput(
            PH2=spark.createDataFrame(
                pd.DataFrame(
                    [["06", "STATE", 1, 3], ["1", "USA", 1, 2]],
                    columns=["REGION_ID", "REGION_TYPE", "PH2_DATA_CELL", "COUNT"],
                )
            ),
            PH4=spark.createDataFrame(
                pd.DataFrame(
                    [["06", "STATE", "*", 1, 4], ["1", "USA", "*", 1, 2]],
                    columns=[
                        "REGION_ID",
                        "REGION_TYPE",
                        "ITERATION_CODE",
                        "PH4_DATA_CELL",
                        "COUNT",
                    ],
                )
            ),
        )
        expected = pd.DataFrame(
            [
                ["PH2", "STATE", "*", "1", "2.0", "1", "44.0"],
                ["PH2", "USA", "*", "1", "1.0", "1", "31.1"],
                ["PH4", "STATE", "*", "1", "3.0", "1", "38.1"],
                ["PH4", "USA", "*", "1", "1.0", "1", "33.0"],
            ],
            columns=[
                "TABLE",
                "REGION_TYPE",
                "ITERATION_CODE",
                "DATA_CELL",
                "EXPERIMENTAL_MOE",
                "COUNT",
                "EXPECTED_MOE",
            ],
        )

        with patch(
            "tmlt.phsafe.accuracy_report.process_private_input_parameters",
            return_value=PrivatePHSafeParameters(
                state_filter=[""],
                reader="",
                data_path="",
                output_path="",
                privacy_defn="zcdp",
                privacy_budget={
                    "PH2": {"state_*": 0.1, "usa_*": 0.2},
                    "PH4": {"state_*": 0.3, "usa_*": 0.4},
                },
                tau={"PH2": 5, "PH4": 8},
            ),
        ), patch(
            "tmlt.phsafe.accuracy_report._get_phsafe_input",
            return_value=PHSafeInput(persons=None, units=None, geo=None),
        ), patch(
            "tmlt.phsafe.accuracy_report.preprocess_geo", return_value=None
        ), patch(
            "tmlt.phsafe.accuracy_report.NonPrivateTabulations",
            return_value=lambda _: nonprivate_answer,
        ), patch(
            "tmlt.phsafe.accuracy_report.PrivateTabulations",
            return_value=lambda _: private_answer,
        ), tempfile.TemporaryDirectory() as output_path:
            create_phsafe_error_report("", "", output_path)
            actual = spark.read.csv(output_path, header=True, sep="|")
            assert_frame_equal_with_sort(actual.toPandas(), expected)


class TestExpectedError:
    """Tests for :func:`~.create_get_expected_error`."""

    @pytest.mark.parametrize(
        "tau, privacy_defn, privacy_budget, table, "
        "iteration_code, region_type, data_cell, expected_error",
        [
            # tests for tables with sensitivity 2tau+2
            (
                {table: 5},
                "zcdp",
                {table: {"state_*": 0.1}},
                table,
                "*",
                "STATE",
                1,
                44.0,
            )
            for table in ["PH1_num", "PH2", "PH4", "PH6", "PH7"]
        ]
        + [
            # tests for tables with sensivity 2
            ({}, "zcdp", {table: {"state_*": 0.2}}, table, "*", "STATE", 1, 5.19)
            for table in ["PH1_denom", "PH5_denom", "PH8_denom"]
        ]
        + [
            # PH5_num is a copy of PH4
            (
                {"PH4": 5},
                "zcdp",
                {"PH4": {"state_*": 0.1}},
                "PH5_num",
                "*",
                "STATE",
                1,
                44.0,
            ),
            # PH8_num cell 3 is a copy of PH7
            (
                {"PH7": 5},
                "zcdp",
                {"PH7": {"state_*": 0.1}},
                "PH8_num",
                "*",
                "STATE",
                3,
                44.0,
            ),
            # Error for PH8_num cell 2 is estimated from PH7 values.
            (
                {"PH7": 5},
                "zcdp",
                {"PH7": {"state_*": 0.1}},
                "PH8_num",
                "*",
                "STATE",
                2,
                62.2,
            ),
            # test for puredp
            (
                {"PH2": 5},
                "puredp",
                {"PH2": {"state_*": 1.0}},
                "PH2",
                "*",
                "STATE",
                1,
                27.1,
            ),
            # tests for other pop groups and tau
            ({"PH2": 10}, "zcdp", {"PH2": {"usa_*": 0.1}}, "PH2", "*", "USA", 1, 80.7),
            (
                {"PH3": 15},
                "zcdp",
                {"PH3": {"state_H,I": 0.1}},
                "PH3",
                "I",
                "STATE",
                1,
                117.3,
            ),
            (
                {"PH3": 15},
                "zcdp",
                {"PH3": {"state_A-G": 0.1}},
                "PH3",
                "B",
                "STATE",
                1,
                117.3,
            ),
        ],
    )
    def test_create_get_expected_error(
        self,
        tau: Dict[str, int],
        privacy_defn: Union[Literal["puredp"], Literal["zcdp"]],
        privacy_budget: Dict[str, Dict[str, float]],
        table: str,
        iteration_code: str,
        region_type: str,
        data_cell: int,
        expected_error: float,
    ):
        """Test for :func:`~.create_get_expected_error`."""
        params = PrivatePHSafeParameters(
            state_filter=[""],
            reader="",
            data_path="",
            output_path="",
            privacy_defn=privacy_defn,
            privacy_budget=privacy_budget,
            tau=tau,
        )
        error_function = create_get_expected_error(params)
        actual = error_function(table, iteration_code, region_type, data_cell)
        assert actual == pytest.approx(expected_error, abs=1e-1)
