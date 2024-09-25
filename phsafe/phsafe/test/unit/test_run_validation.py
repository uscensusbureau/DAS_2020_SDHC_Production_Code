"""Tests end-to-end PHSafe validation."""

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

import json
import os
from tempfile import TemporaryDirectory
from typing import Dict

import pytest

from tmlt.common.pyspark_test_tools import pyspark  # pylint: disable=unused-import
from tmlt.phsafe.input_processing import (
    PRIVACY_BUDGET_KEY,
    PRIVACY_DEFN_FLAG,
    READER_FLAG,
    STATE_FILTER_FLAG,
    TAU_KEY,
)
from tmlt.phsafe.runners import run_input_validation

VALID_PERSONS_DATA = (
    "RTYPE,MAFID,QAGE,CENHISP,CENRACE,RELSHIP\n"
    "3,100000001,19,1,01,20\n"
    "3,100000001,18,1,01,21\n"
    "3,100000002,19,1,01,20\n"
    "5,100000004,94,1,01,38\n"
    "5,100000004,19,2,01,38"
)

VALID_UNITS_DATA = (
    "RTYPE,MAFID,FINAL_POP,NPF,HHSPAN,HHRACE,TEN,HHT,HHT2,CPLT\n"
    "2,100000001,2,2,1,01,2,1,02,1\n"
    "2,100000002,1,0,1,01,2,4,09,5\n"
    "2,100000003,0,0,0,00,0,0,00,0\n"
    "2,100000005,0,0,0,00,0,0,00,0\n"
    "4,100000004,2,0,0,00,0,0,00,0"
)

VALID_GEO_DATA = (
    "RTYPE,MAFID,TABBLKST,TABBLKCOU,TABTRACTCE,TABBLK,TABBLKGRPCE,REGIONCE,DIVISIONCE,"
    "PLACEFP,AIANNHCE\n"
    "2,100000001,44,055,450200,4069,1,4,2,99999,9999\n"
    "2,100000002,44,161,890500,2078,0,2,2,19828,9999\n"
    "2,100000003,29,161,890400,1073,9,1,2,62912,9999\n"
    "2,100000005,29,161,890400,1073,9,1,2,99999,0001\n"
    "4,100000004,29,065,960100,1123,1,1,0,99999,9999"
)

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

TAUS = {"PH1_num": 1, "PH2": 1, "PH3": 5, "PH4": 1, "PH6": 5, "PH7": 1}

# pylint: disable=consider-using-with


@pytest.fixture(scope="class")
def setup_run_validation(request):
    """Set up."""
    request.cls.valid_inputs = TemporaryDirectory()
    with open(os.path.join(request.cls.valid_inputs.name, "persons.csv"), "w") as f:
        f.write(VALID_PERSONS_DATA)

    with open(os.path.join(request.cls.valid_inputs.name, "units.csv"), "w") as f:
        f.write(VALID_UNITS_DATA)

    with open(os.path.join(request.cls.valid_inputs.name, "geo.csv"), "w") as f:
        f.write(VALID_GEO_DATA)

    request.cls.valid_configs = TemporaryDirectory()
    request.cls.valid_configs_path = os.path.join(
        request.cls.valid_configs.name, "config.json"
    )
    with open(request.cls.valid_configs_path, "w") as f:
        json.dump(
            {
                PRIVACY_BUDGET_KEY: PRIVACY_BUDGETS,
                PRIVACY_DEFN_FLAG: "puredp",
                TAU_KEY: TAUS,
                STATE_FILTER_FLAG: ["44", "29"],
                READER_FLAG: "csv",
            },
            f,
        )
        f.flush()
    yield
    request.cls.valid_inputs.cleanup()
    request.cls.valid_configs.cleanup()


@pytest.mark.usefixtures("spark")
@pytest.mark.usefixtures("setup_run_validation")
class TestValidation:
    """Parameterized unit tests for PHSafe end-to-end validation."""

    valid_inputs: TemporaryDirectory
    valid_configs: TemporaryDirectory
    valid_configs_path: str

    @pytest.mark.parametrize(
        "config, msg",
        [
            (  # Missing TAU_KEY
                {
                    PRIVACY_BUDGET_KEY: PRIVACY_BUDGETS,
                    PRIVACY_DEFN_FLAG: "puredp",
                    READER_FLAG: "cef",
                    STATE_FILTER_FLAG: ["44"],
                },
                "missing keys {tau}",
            ),
            (  # Invalid READER_FLAG
                {
                    PRIVACY_BUDGET_KEY: PRIVACY_BUDGETS,
                    PRIVACY_DEFN_FLAG: "puredp",
                    READER_FLAG: "def",
                    STATE_FILTER_FLAG: ["44"],
                    TAU_KEY: TAUS,
                },
                f"{READER_FLAG} must be one of: csv, cef",
            ),
            (  # Invalid STATE_FILTER_FLAG
                {
                    PRIVACY_BUDGET_KEY: PRIVACY_BUDGETS,
                    PRIVACY_DEFN_FLAG: "puredp",
                    READER_FLAG: "cef",
                    STATE_FILTER_FLAG: ["444"],
                    TAU_KEY: TAUS,
                },
                rf"{STATE_FILTER_FLAG} contains invalid codes \{{444}}",
            ),
            (  # Invalid PRIVACY_DEFN_FLAG
                {
                    PRIVACY_BUDGET_KEY: PRIVACY_BUDGETS,
                    PRIVACY_DEFN_FLAG: "tau",
                    READER_FLAG: "cef",
                    STATE_FILTER_FLAG: ["44"],
                    TAU_KEY: TAUS,
                },
                f"{PRIVACY_DEFN_FLAG} must be one of: puredp, zcdp",
            ),
        ],
    )
    def test_validation_checks_config_parameters(self, config: Dict, msg: str):
        """Test that end-to-end validation checks config parameters correctly.

        Args:
            config: Dictionary with config parameters.
            msg: Expected error message.
        """
        config_path = os.path.join(self.valid_inputs.name, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f)
            f.flush()
            with pytest.raises(RuntimeError, match=msg):
                run_input_validation(
                    config_path=config_path, data_path=self.valid_inputs.name
                )

    @pytest.mark.parametrize(
        "persons_raw, units_raw, geo_raw",
        [
            (  # Invalid RTYPE in persons table
                "RTYPE,MAFID,QAGE,CENHISP,CENRACE,RELSHIP\n"
                "1,100000001,19,1,01,20\n"
                "3,100000001,18,1,01,21\n"
                "3,100000002,19,1,01,20\n"
                "5,100000004,94,1,01,38\n"
                "5,100000004,19,2,01,38",
                VALID_UNITS_DATA,
                VALID_GEO_DATA,
            ),
            (  # HHRACE column missing in units table
                VALID_PERSONS_DATA,
                "RTYPE,MAFID,FINAL_POP,NPF,HHSPAN,TEN,HHT,HHT2,CPLT\n"
                "2,100000001,2,2,1,2,1,02,1\n"
                "2,100000002,1,0,1,2,4,09,5\n"
                "2,100000003,0,0,0,0,0,00,0\n"
                "2,100000005,0,0,0,0,0,00,0\n"
                "4,100000004,2,0,0,0,0,00,0",
                VALID_GEO_DATA,
            ),
            (  # Invalid MAFID in geo table
                VALID_PERSONS_DATA,
                VALID_UNITS_DATA,
                "RTYPE,MAFID,TABBLKST,TABBLKCOU,TABTRACTCE,TABBLK,TABBLKGRPCE,"
                "REGIONCE,DIVISIONCE,PLACEFP,AIANNHCE\n"
                "2,1000000001,44,055,450200,4069,0,4,2,99999,9999\n"
                "2,100000002,44,161,890500,2078,1,4,2,99999,9999\n"
                "2,100000003,29,161,890400,1073,1,4,2,99999,9999\n"
                "2,100000005,29,161,890400,1073,9,1,2,62912,0001\n"
                "4,100000004,29,065,960100,1123,9,1,2,99999,9999",
            ),
            (  # Invalid TABBLKCOU=55 in geo table
                VALID_PERSONS_DATA,
                VALID_UNITS_DATA,
                "RTYPE,MAFID,TABBLKST,TABBLKCOU,TABTRACTCE,TABBLK,TABBLKGRPCE,"
                "REGIONCE,DIVISIONCE,PLACEFP,AIANNHCE\n"
                "2,100000001,44,55,450200,4069,0,4,2,99999,9999\n"
                "2,100000002,44,161,890500,2078,1,4,2,99999,9999\n"
                "2,100000003,29,161,890400,1073,1,4,2,99999,9999\n"
                "2,100000005,29,161,890400,1073,9,1,2,62912,0001\n"
                "4,100000004,29,065,960100,1123,9,1,2,99999,9999",
            ),
        ],
    )
    def test_validation_checks_input_files(
        self, persons_raw: str, units_raw: str, geo_raw: str
    ):
        """Test that end-to-end validation checks input files.

        Args:
            persons_raw: CSV-formatted string containing persons records.
            units_raw: CSV-formatted string containing units records.
            geo_raw: CSV-formatted string containing geo records.
        """
        with open(os.path.join(self.valid_configs.name, "persons.csv"), "w") as f:
            f.write(persons_raw)
        with open(os.path.join(self.valid_configs.name, "units.csv"), "w") as f:
            f.write(units_raw)
        with open(os.path.join(self.valid_configs.name, "geo.csv"), "w") as f:
            f.write(geo_raw)
        with pytest.raises(RuntimeError, match="Input validation Failed."):
            run_input_validation(
                config_path=self.valid_configs_path, data_path=self.valid_configs.name
            )
