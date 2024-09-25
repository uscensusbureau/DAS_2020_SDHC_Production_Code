"""Tests end-to-end PHSafe tabulations on toy data."""

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
import shutil
import tempfile
from typing import Dict

import pytest
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import LongType, StringType, StructField, StructType

from tmlt.common.pyspark_test_tools import pyspark  # pylint: disable=unused-import
from tmlt.common.pyspark_test_tools import assert_frame_equal_with_sort
from tmlt.phsafe.csv_reader import (
    GEO_FILENAME,
    PERSON_FILENAME,
    UNIT_FILENAME,
    CSVReader,
)
from tmlt.phsafe.input_processing import (
    PRIVACY_BUDGET_KEY,
    PRIVACY_DEFN_FLAG,
    READER_FLAG,
    STATE_FILTER_FLAG,
    TAU_KEY,
)
from tmlt.phsafe.runners import preprocess_geo, run_tabulation

OUTPUT_DIRS = [
    "PH1_num",
    "PH1_denom",
    "PH2",
    "PH3",
    "PH4",
    "PH5_num",
    "PH5_denom",
    "PH6",
    "PH7",
    "PH8_num",
    "PH8_denom",
]

VALID_GEO_DATA = (
    "RTYPE,MAFID,TABBLKST,TABBLKCOU,TABTRACTCE,TABBLK,TABBLKGRPCE,REGIONCE,DIVISIONCE,"
    "PLACEFP,AIANNHCE\n"
    "2,100000001,44,055,450200,4069,1,4,2,99999,9999\n"
    "2,100000002,44,161,890500,2078,0,2,2,19828,9999\n"
    "2,100000003,29,161,890400,1073,9,1,2,62912,9999\n"
    "2,100000005,29,161,890400,1073,9,1,2,99999,0001\n"
    "4,100000004,29,065,960100,1123,1,1,0,99999,9999"
)

VALID_PERSONS_DATA = """RTYPE,MAFID,QAGE,CENHISP,CENRACE,RELSHIP\n
3,100000001,19,1,01,20\n
3,100000001,18,1,01,21\n
3,100000002,19,1,01,20\n
5,100000004,94,1,01,38\n
5,100000004,19,2,01,38"""

VALID_UNITS_DATA = """RTYPE,MAFID,FINAL_POP,NPF,HHSPAN,HHRACE,TEN,HHT,HHT2,CPLT\n
2,100000001,2,2,1,01,2,1,02,1\n
2,100000002,1,0,1,01,2,4,09,5\n
2,100000003,0,0,0,00,0,0,00,0\n
2,100000005,0,0,0,00,0,0,00,0\n
4,100000004,2,0,0,00,0,0,00,0"""

PRIVACY_BUDGETS = {
    "PH1_num": {
        "usa_A-G": 0.055,
        "usa_H,I": 0.055,
        "usa_*": 0.055,
        "state_A-G": 0,
        "state_H,I": 0,
        "state_*": 0,
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

TAUS = {"PH1_num": 5, "PH2": 5, "PH3": 5, "PH4": 5, "PH6": 5, "PH7": 5}

CONFIG = {
    STATE_FILTER_FLAG: ["44", "29"],
    READER_FLAG: "csv",
    PRIVACY_BUDGET_KEY: PRIVACY_BUDGETS,
    PRIVACY_DEFN_FLAG: "puredp",
    TAU_KEY: TAUS,
}


@pytest.fixture(scope="class")
def setup_run_tabulations(request):
    """Set up test."""
    request.cls.tmp_dir = tempfile.mkdtemp()

    request.cls.config_path = os.path.join(request.cls.tmp_dir, "config.json")
    with open(request.cls.config_path, "w") as f:
        json.dump(CONFIG, f, indent=4)

    geo_filename = os.path.join(request.cls.tmp_dir, GEO_FILENAME)
    unit_filename = os.path.join(request.cls.tmp_dir, UNIT_FILENAME)
    person_filename = os.path.join(request.cls.tmp_dir, PERSON_FILENAME)
    with open(geo_filename, "w") as f:
        f.write(VALID_GEO_DATA)
    with open(unit_filename, "w") as f:
        f.write(VALID_UNITS_DATA)
    with open(person_filename, "w") as f:
        f.write(VALID_PERSONS_DATA)

    reader = CSVReader(request.cls.tmp_dir, ["44", "29"])
    request.cls.geo_df = reader.get_geo_df()
    request.cls.unit_df = reader.get_unit_df()
    request.cls.person_df = reader.get_person_df()
    yield
    shutil.rmtree(request.cls.tmp_dir)


@pytest.mark.usefixtures("spark")
@pytest.mark.usefixtures("setup_run_tabulations")
class TestRunner:
    """Parameterized unit tests for csv reader."""

    tmp_dir: str
    config_path: str
    geo_df: DataFrame
    unit_df: DataFrame
    person_df: DataFrame

    def test_preprocess_geo(self, spark: SparkSession):
        """Test geo_df preprocessing."""
        actual = preprocess_geo(self.geo_df)
        expected = spark.createDataFrame(
            spark.sparkContext.parallelize(
                [
                    (100000001, "1", "44"),
                    (100000002, "1", "44"),
                    (100000003, "1", "29"),
                    (100000005, "1", "29"),
                ]
            ),
            StructType(
                [
                    StructField("MAFID", LongType(), True),
                    StructField("USA", StringType(), True),
                    StructField("STATE", StringType(), True),
                ]
            ),
        )

        assert_frame_equal_with_sort(
            actual.toPandas(), expected.toPandas(), ["MAFID", "USA", "STATE"]
        )

    # This test is not run frequently as it is covered by E2E system test
    @pytest.mark.slow
    @pytest.mark.parametrize(
        "validate, private",
        [(False, False), (False, True), (True, False), (True, True)],
    )
    def test_run_smoke(self, validate: bool, private: bool):
        """End to end without crashing

        Args:
            validate: If true, perform validation before tabulation.
            private: If true, run DP algorithm.
        """
        run_tabulation(
            self.config_path,
            self.tmp_dir,
            self.tmp_dir,
            should_validate_input=validate,
            should_validate_private_output=False,
            private=private,
        )

    @pytest.mark.parametrize(
        "person_raw, unit_raw, geo_raw, config, private",
        [
            (person, unit, geo, config, private)
            for person, unit, geo, config in [
                (  # Invalid State filter
                    VALID_PERSONS_DATA,
                    VALID_UNITS_DATA,
                    VALID_GEO_DATA,
                    {
                        STATE_FILTER_FLAG: ["444", "29"],
                        READER_FLAG: "csv",
                        PRIVACY_DEFN_FLAG: "puredp",
                        PRIVACY_BUDGET_KEY: PRIVACY_BUDGETS,
                        TAU_KEY: TAUS,
                    },
                ),
                (  # Invalid RTYPE in persons table
                    "RTYPE,MAFID,QAGE,CENHISP,CENRACE,RELSHIP\n"
                    "1,100000001,19,1,01,20\n"
                    "3,100000001,18,1,01,21\n"
                    "3,100000002,19,1,01,20\n"
                    "5,100000004,94,1,01,38\n"
                    "5,100000004,19,2,01,38",
                    VALID_PERSONS_DATA,
                    VALID_GEO_DATA,
                    CONFIG,
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
                    CONFIG,
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
                    CONFIG,
                ),
            ]
            for private in [True, False]
        ],
    )
    def test_run_tabulation_with_validation_fails(
        self, person_raw: str, unit_raw: str, geo_raw: str, config: Dict, private: bool
    ):
        """Tests run with validation flag validates before tabulation."""
        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, PERSON_FILENAME), "w") as f:
                f.write(person_raw)
            with open(os.path.join(td, UNIT_FILENAME), "w") as f:
                f.write(unit_raw)
            with open(os.path.join(td, GEO_FILENAME), "w") as f:
                f.write(geo_raw)
            config_path = os.path.join(td, "config.json")
            with open(config_path, "w") as f:
                json.dump(config, f)
                f.flush()
            with pytest.raises(RuntimeError, match="[Vv]alidation failed"):
                run_tabulation(
                    config_path,
                    td,
                    td,
                    should_validate_input=True,
                    should_validate_private_output=False,
                    private=private,
                )
