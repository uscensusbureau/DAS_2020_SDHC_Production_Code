"""Tests `config.json` is processed correctly."""

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
import json
import os
import shutil
import tempfile
from typing import Collection, ContextManager, Dict, Optional

import pytest
from pyspark.sql.session import SparkSession

from tmlt.common.pyspark_test_tools import pyspark  # pylint: disable=unused-import
from tmlt.common.pyspark_test_tools import assert_frame_equal_with_sort
from tmlt.phsafe.input_processing import (
    PRIVACY_BUDGET_KEY,
    PRIVACY_DEFN_FLAG,
    READER_FLAG,
    STATE_FILTER_FLAG,
    TAU_KEY,
    NonprivatePHSafeParameters,
    PrivatePHSafeParameters,
    _parse_config_json,
    process_nonprivate_input_parameters,
    process_private_input_parameters,
)
from tmlt.phsafe.runners import preprocess_geo

PRIVACY_BUDGETS = {
    "PH1_num": {
        "usa_A-G": 0.055,
        "usa_H,I": 0.055,
        "usa_*": 0.055,
        "state_A-G": 0.055,
        "state_H,I": 0.055,
        "state_*": 0.055,
    },
    "PH1_denom": {
        "usa_A-G": 0.055,
        "usa_H,I": 0.055,
        "usa_*": 0.055,
        "state_A-G": 0.055,
        "state_H,I": 0.055,
        "state_*": 0.055,
    },
    "PH2": {"usa_*": 0.166, "state_*": 0.166},
    "PH3": {
        "usa_A-G": 0,
        "usa_H,I": 0,
        "usa_*": 0,
        "state_A-G": 0,
        "state_H,I": 0,
        "state_*": 0,
    },
    "PH4": {
        "usa_A-G": 0.055,
        "usa_H,I": 0.055,
        "usa_*": 0.055,
        "state_A-G": 0.055,
        "state_H,I": 0.055,
        "state_*": 0.055,
    },
    "PH5_denom": {
        "usa_A-G": 0.055,
        "usa_H,I": 0.055,
        "usa_*": 0.055,
        "state_A-G": 0.055,
        "state_H,I": 0.055,
        "state_*": 0.055,
    },
    "PH6": {"usa_*": 0.166, "state_*": 0.166},
    "PH7": {
        "usa_A-G": 0.055,
        "usa_H,I": 0.055,
        "usa_*": 0.055,
        "state_A-G": 0.055,
        "state_H,I": 0.055,
        "state_*": 0.055,
    },
    "PH8_denom": {
        "usa_A-G": 0.055,
        "usa_H,I": 0.055,
        "usa_*": 0.055,
        "state_A-G": 0.055,
        "state_H,I": 0.055,
        "state_*": 0.055,
    },
}

PRIVACY_BUDGETS_INVALID_TYPE = copy.deepcopy(PRIVACY_BUDGETS)
PRIVACY_BUDGETS_INVALID_TYPE["PH6"] = {"usa_*": "0.166", "state_*": 0.166}

PRIVACY_BUDGETS_INVALID_VALUE = copy.deepcopy(PRIVACY_BUDGETS)
PRIVACY_BUDGETS_INVALID_VALUE["PH6"] = {"usa_*": -0.166, "state_*": 0.166}

PRIVACY_BUDGETS_MISSING_POP_GROUP = copy.deepcopy(PRIVACY_BUDGETS)
PRIVACY_BUDGETS_MISSING_POP_GROUP["PH6"] = {"usa_*": -0.166}

PRIVACY_BUDGETS_WITH_EXTRA_POP_GROUP = copy.deepcopy(PRIVACY_BUDGETS)
PRIVACY_BUDGETS_WITH_EXTRA_POP_GROUP["PH6"] = {
    "usa_*": 0.166,
    "state_*": 0.166,
    "foo": 1,
}


@pytest.mark.usefixtures("spark")
class TestPHSafeParameters:
    """Parameterized unit tests for csv reader."""

    def setUp(self):
        """Set up test."""
        self.tmp_dir = tempfile.mkdtemp()

        config = {STATE_FILTER_FLAG: ["44", "29"], READER_FLAG: "csv"}
        self.config_path = os.path.join(self.tmp_dir, "config.json")
        with open(self.config_path, "w") as f:
            json.dump(config, f, indent=4)

    @pytest.mark.parametrize(
        "config_data, config_keys, expected_config",
        [
            (dict(), set(), None),
            ({"key": "value"}, set(), dict()),
            ({STATE_FILTER_FLAG: ["72"]}, {STATE_FILTER_FLAG}, None),
            ({STATE_FILTER_FLAG: ["01", "56"]}, {STATE_FILTER_FLAG}, None),
            ({READER_FLAG: "csv"}, {READER_FLAG}, None),
            ({READER_FLAG: "cef"}, {READER_FLAG}, None),
            ({PRIVACY_BUDGET_KEY: PRIVACY_BUDGETS}, {PRIVACY_BUDGET_KEY}, None),
            ({PRIVACY_DEFN_FLAG: "puredp"}, {PRIVACY_DEFN_FLAG}, None),
            (
                {
                    TAU_KEY: {
                        "PH1_num": 5,
                        "PH2": 10,
                        "PH3": 5,
                        "PH4": 10,
                        "PH6": 5,
                        "PH7": 5,
                    }
                },
                {TAU_KEY},
                None,
            ),
            (
                {
                    TAU_KEY: {
                        "PH1_num": 5,
                        "PH2": 5,
                        "PH3": 5,
                        "PH4": 5,
                        "PH6": 5,
                        "PH7": 5,
                    }
                },
                {TAU_KEY},
                None,
            ),
        ],
    )
    def test_parse_config_json_valid(
        self,
        config_data: Dict,
        config_keys: Collection,
        expected_config: Optional[Dict],
    ):
        """_parse_config_json correctly validates and returns valid configurations.

        Args:
            config_data: JSON data to be validated, as a Python dict
            config_keys: The expected keys in the configuration
            expected_config: The expected configuration for _parse_config_json
                to return, or None, in which case the returned config should
                match config_data.
        """
        if expected_config is None:
            expected_config = config_data

        with tempfile.NamedTemporaryFile("w") as fp:
            json.dump(config_data, fp)
            fp.flush()

            actual_config = _parse_config_json(fp.name, config_keys)
            assert actual_config == expected_config

    # Remember to escape opening braces when they contain a number (e.g. \{03})
    # so that they are not interpreted as regex repetition markers.
    @pytest.mark.parametrize(
        "config_data, config_keys, expectation",
        [
            ({}, {"key"}, pytest.raises(RuntimeError, match="missing keys {key}")),
            (
                {STATE_FILTER_FLAG: []},
                {STATE_FILTER_FLAG},
                pytest.raises(
                    RuntimeError, match=f"expected {STATE_FILTER_FLAG} to not be empty"
                ),
            ),
            (
                {STATE_FILTER_FLAG: [10]},
                {STATE_FILTER_FLAG},
                pytest.raises(
                    RuntimeError,
                    match=(
                        f"expected {STATE_FILTER_FLAG} elements to have type str, not"
                        " {int}"
                    ),
                ),
            ),
            (
                {STATE_FILTER_FLAG: ["1"]},
                {STATE_FILTER_FLAG},
                pytest.raises(
                    RuntimeError,
                    match=rf"{STATE_FILTER_FLAG} contains invalid codes \{{1}}",
                ),
            ),
            (
                {STATE_FILTER_FLAG: ["011", "57", "14", "60", "64", "78"]},
                {STATE_FILTER_FLAG},
                pytest.raises(
                    RuntimeError,
                    match=rf"{STATE_FILTER_FLAG} contains invalid codes \{{011, 14, 57,"
                    " 60, 64, 78}",
                ),
            ),
            (
                {STATE_FILTER_FLAG: ["03"]},
                {STATE_FILTER_FLAG},
                pytest.raises(
                    RuntimeError,
                    match=rf"{STATE_FILTER_FLAG} contains invalid codes \{{03}}",
                ),
            ),
            (
                {STATE_FILTER_FLAG: ["01", "02", "01"]},
                {STATE_FILTER_FLAG},
                pytest.raises(
                    RuntimeError,
                    match=f"{STATE_FILTER_FLAG} list contains duplicate values",
                ),
            ),
            (
                {STATE_FILTER_FLAG: ["01", "72"]},
                {STATE_FILTER_FLAG},
                pytest.raises(
                    RuntimeError,
                    match="Running PR with the rest of the US is not supported",
                ),
            ),
            (
                {READER_FLAG: "cff"},
                {READER_FLAG},
                pytest.raises(
                    RuntimeError, match=f"{READER_FLAG} must be one of: csv, cef"
                ),
            ),
            (
                {PRIVACY_DEFN_FLAG: "pure"},
                {PRIVACY_DEFN_FLAG},
                pytest.raises(
                    RuntimeError,
                    match=f"{PRIVACY_DEFN_FLAG} must be one of: puredp, zcdp",
                ),
            ),
            (
                {
                    PRIVACY_BUDGET_KEY: {
                        "PH1_num": 1,
                        "PH1_denom": 1,
                        "PH2": 1,
                        "PH3": 2,
                        "PH4": 2,
                        "PH5_denom": 1,
                        "PH6": 1,
                        "PH7": 1,
                        "PH8_denom": 1,
                    }
                },
                {PRIVACY_BUDGET_KEY},
                pytest.raises(
                    RuntimeError, match="expected PH1_num to have type dict, not int"
                ),
            ),
            (
                {PRIVACY_BUDGET_KEY: PRIVACY_BUDGETS_MISSING_POP_GROUP},
                {PRIVACY_BUDGET_KEY},
                pytest.raises(
                    RuntimeError,
                    match="PH6 is missing required budget values:" r" \['state_\*'\]",
                ),
            ),
            (
                {PRIVACY_BUDGET_KEY: PRIVACY_BUDGETS_WITH_EXTRA_POP_GROUP},
                {PRIVACY_BUDGET_KEY},
                pytest.raises(
                    RuntimeError,
                    match="PH6 has unexpected budget values: " r"\['foo'\]",
                ),
            ),
            (
                {PRIVACY_BUDGET_KEY: PRIVACY_BUDGETS_INVALID_TYPE},
                {PRIVACY_BUDGET_KEY},
                pytest.raises(
                    RuntimeError,
                    match=r"expected PH6's 'usa_\*' key to have float value, not str",
                ),
            ),
            (
                {PRIVACY_BUDGET_KEY: PRIVACY_BUDGETS_INVALID_VALUE},
                {PRIVACY_BUDGET_KEY},
                pytest.raises(
                    RuntimeError,
                    match=r"value of PH6's 'usa_\*' key must be non-negative",
                ),
            ),
            (
                {
                    PRIVACY_BUDGET_KEY: {
                        k: v
                        for k, v in PRIVACY_BUDGETS_INVALID_TYPE.items()
                        if k != "PH6"
                    }
                },
                {PRIVACY_BUDGET_KEY},
                pytest.raises(
                    RuntimeError,
                    match="Missing privacy_budget value for these tabulations: "
                    r"\['PH6'\]",
                ),
            ),
            (
                {
                    PRIVACY_BUDGET_KEY: {
                        k: v
                        for k, v in PRIVACY_BUDGETS_INVALID_TYPE.items()
                        if k not in ["PH3", "PH6"]
                    }
                },
                {PRIVACY_BUDGET_KEY},
                pytest.raises(
                    RuntimeError,
                    match="Missing privacy_budget value for these tabulations: "
                    r"\['PH3', 'PH6'\]",
                ),
            ),
            (
                {
                    TAU_KEY: {
                        "PH1_num": 5,
                        "PH2": 1.1,
                        "PH3": 5,
                        "PH4": 5,
                        "PH6": 5,
                        "PH7": 5,
                    }
                },
                {TAU_KEY},
                pytest.raises(
                    RuntimeError,
                    match="expected tau value for PH2 to have type int, not float",
                ),
            ),
            (
                {
                    TAU_KEY: {
                        "PH1_num": 5,
                        "PH2": 0,
                        "PH3": 5,
                        "PH4": 5,
                        "PH6": 5,
                        "PH7": 5,
                    }
                },
                {TAU_KEY},
                pytest.raises(
                    RuntimeError, match="value of tau for PH2 must be greater than zero"
                ),
            ),
            (
                {
                    TAU_KEY: {
                        "PH1_num": -1,
                        "PH2": 1,
                        "PH3": 1,
                        "PH4": 1,
                        "PH6": 1,
                        "PH7": 1,
                    }
                },
                {TAU_KEY},
                pytest.raises(
                    RuntimeError,
                    match="value of tau for PH1_num must be greater than zero",
                ),
            ),
            (
                {TAU_KEY: {"PH1_num": 1, "PH6": 1, "PH7": 1}},
                {TAU_KEY},
                pytest.raises(
                    RuntimeError,
                    match="Missing tau value for these tabulations: "
                    r"\['PH2', 'PH3', 'PH4'\]",
                ),
            ),
        ],
    )
    def test_parse_config_json_invalid_config(
        self,
        config_data: Dict,
        config_keys: Collection,
        expectation: ContextManager[None],
    ):
        """_parse_config_json raises appropriate errors for invalid configurations.

        Args:
            config_data: JSON data to be validated, as a Python dict
            config_keys: The expected keys in the configuration
            expectation: The expectation of whether the function will throw an error or
                not. Should contain a regex to match the error message against.
        """
        with tempfile.NamedTemporaryFile("w") as fp:
            json.dump(config_data, fp)
            fp.flush()

            with expectation:
                _ = _parse_config_json(fp.name, config_keys)

    @pytest.mark.parametrize(
        "config",
        [
            {STATE_FILTER_FLAG: ["44", "29"], READER_FLAG: "csv"},
            {STATE_FILTER_FLAG: ["44", "29"], READER_FLAG: "cef"},
        ],
    )
    def test_process_nonprivate_input_parameters(self, config: Dict):
        """Test that process_nonprivate_input_parameters works correctly.

        Args:
            config: Config dictionary.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = os.path.join(tmp_dir, "config.json")
            with open(config_path, "w") as f:
                json.dump(config, f)
                f.flush()

                params = process_nonprivate_input_parameters(
                    config_path, tmp_dir, tmp_dir
                )
                expected_params = NonprivatePHSafeParameters(
                    config[STATE_FILTER_FLAG], config[READER_FLAG], tmp_dir, tmp_dir
                )
                assert params == expected_params

    @pytest.mark.parametrize(
        "config",
        [
            {
                STATE_FILTER_FLAG: ["44", "29"],
                READER_FLAG: "csv",
                PRIVACY_DEFN_FLAG: "puredp",
                PRIVACY_BUDGET_KEY: PRIVACY_BUDGETS,
                TAU_KEY: {
                    "PH1_num": 1,
                    "PH2": 1,
                    "PH3": 5,
                    "PH4": 1,
                    "PH6": 1,
                    "PH7": 1,
                },
            },
            {
                STATE_FILTER_FLAG: ["44", "29"],
                READER_FLAG: "cef",
                PRIVACY_DEFN_FLAG: "puredp",
                PRIVACY_BUDGET_KEY: PRIVACY_BUDGETS,
                TAU_KEY: {
                    "PH1_num": 1,
                    "PH2": 1,
                    "PH3": 5,
                    "PH4": 1,
                    "PH6": 1,
                    "PH7": 1,
                },
            },
        ],
    )
    def test_process_private_input_parameters(self, config: Dict):
        """Test that process_private_input_parameters works correctly.

        Args:
            config: Config dictionary.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = os.path.join(tmp_dir, "config.json")
            with open(config_path, "w") as f:
                json.dump(config, f)
                f.flush()

                params = process_private_input_parameters(config_path, tmp_dir, tmp_dir)
                expected_params = PrivatePHSafeParameters(
                    config[STATE_FILTER_FLAG],
                    config[READER_FLAG],
                    tmp_dir,
                    tmp_dir,
                    config[PRIVACY_DEFN_FLAG],
                    config[PRIVACY_BUDGET_KEY],
                    config[TAU_KEY],
                )
                assert params == expected_params

    # Test that RTYPE 4 and 5 are filtered out in the preprocessed geo df.
    @pytest.mark.parametrize(
        "input_frame_data, expected_frame_data",
        [
            # This should return the values with only RTYPE = 2
            (
                (
                    [
                        (
                            "2",
                            "100000005",
                            "29",
                            "161",
                            "890400",
                            "1073",
                            "9",
                            "1",
                            "2",
                            "99999",
                            "0001",
                        ),
                        (
                            "4",
                            "100000004",
                            "29",
                            "065",
                            "960100",
                            "1123",
                            "1",
                            "1",
                            "0",
                            "99999",
                            "9999",
                        ),
                        (
                            "2",
                            "100000002",
                            "72",
                            "139",
                            "060102",
                            "1005",
                            "1",
                            "9",
                            "0",
                            "83606",
                            "9999",
                        ),
                    ],
                    [
                        "RTYPE",
                        "MAFID",
                        "TABBLKST",
                        "TABBLKCOU",
                        "TABTRACTCE",
                        "TABBLK",
                        "TABBLKGRPCE",
                        "REGIONCE",
                        "DIVISIONCE",
                        "PLACEFP",
                        "AIANNHCE",
                    ],
                ),
                (
                    [("100000005", "1", "29"), ("100000002", "1", "72")],
                    ["MAFID", "USA", "STATE"],
                ),
            )
        ],
    )
    def testingFiltering(
        self, input_frame_data: tuple, expected_frame_data: tuple, spark: SparkSession
    ):
        """Tests an input to preprocess_geo, which filters and modifies geo data."""
        input_df = spark.createDataFrame(input_frame_data[0], input_frame_data[1])
        output = preprocess_geo(input_df)
        expected = spark.createDataFrame(expected_frame_data[0], expected_frame_data[1])
        assert_frame_equal_with_sort(output.toPandas(), expected.toPandas())

    def tearDown(self) -> None:
        """Cleans up temp directory."""
        shutil.rmtree(self.tmp_dir)
