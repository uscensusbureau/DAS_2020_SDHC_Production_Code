"""System tests for PHSafe, making sure that the algorithm has the correct output."""

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

# pylint: disable=protected-access, no-member
import json
import os
import shutil
import tempfile
from typing import List

import pandas as pd
import pytest

from tmlt.common.io_helpers import multi_read_csv
from tmlt.common.pyspark_test_tools import pyspark  # pylint: disable=unused-import
from tmlt.phsafe import TABULATIONS_KEY
from tmlt.phsafe.paths import RESOURCES_DIR
from tmlt.phsafe.runners import run_tabulation
from tmlt.phsafe.utils import (
    get_augmented_df,
    get_config_privacy_budget_dict,
    update_config_file,
    validate_directory_single_config,
)

ITERATION_CODES = {
    "A-G": ["A", "B", "C", "D", "E", "F", "G"],
    "H,I": ["H", "I"],
    "*": ["*"],
}
"""All possible values for ITERATION_CODE by iteration level."""

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
        "usa_A-G": 0.055,
        "usa_H,I": 0.055,
        "usa_*": 0.055,
        "state_A-G": 0.055,
        "state_H,I": 0.055,
        "state_*": 0.055,
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

CONFIG_PUREDP = {
    "privacy_budget": PRIVACY_BUDGETS,
    "tau": {"PH1_num": 3, "PH2": 3, "PH3": 3, "PH4": 3, "PH6": 3, "PH7": 3},
    "state_filter": ["44", "29"],
    "reader": "csv",
    "privacy_defn": "puredp",
}
CONFIG_ZCDP = {
    "privacy_budget": PRIVACY_BUDGETS,
    "tau": {"PH1_num": 3, "PH2": 3, "PH3": 3, "PH4": 3, "PH6": 3, "PH7": 3},
    "state_filter": ["44", "29"],
    "reader": "csv",
    "privacy_defn": "zcdp",
}
PERSONS = (
    "RTYPE,MAFID,QAGE,CENHISP,CENRACE,RELSHIP\n"
    "3,100000001,49,1,01,20\n"
    "3,100000001,18,1,01,21\n"
    "3,100000002,17,1,01,20\n"
    "5,100000004,94,1,01,38\n"
    "5,100000004,19,2,01,38"
)
UNITS = (
    "RTYPE,MAFID,FINAL_POP,NPF,HHSPAN,HHRACE,TEN,HHT,HHT2,CPLT\n"
    "2,100000001,2,2,1,01,2,1,02,1\n"
    "2,100000002,1,0,1,01,2,4,09,5\n"
    "2,100000003,0,0,0,00,0,0,00,0\n"
    "2,100000005,0,0,0,00,0,0,00,0\n"
    "4,100000004,2,0,0,00,0,0,00,0"
)
GEO = (
    "RTYPE,MAFID,TABBLKST,TABBLKCOU,TABTRACTCE,TABBLK,TABBLKGRPCE,REGIONCE,DIVISIONCE,"
    "PLACEFP,AIANNHCE\n"
    "2,100000001,44,055,450200,4069,1,4,2,99999,9999\n"
    "2,100000002,44,161,890500,2078,0,2,2,19828,9999\n"
    "2,100000003,29,161,890400,1073,9,1,2,62912,9999\n"
    "2,100000005,29,161,890400,1073,9,1,2,99999,0001\n"
    "4,100000004,29,065,960100,1123,1,1,0,99999,9999"
)


@pytest.mark.parametrize("privacy_defn", ["puredp", "zcdp"], scope="class")
@pytest.mark.usefixtures("spark")
class TestPHSafeAlgorithms:
    """Test PHSafe algorithms"""

    privacy_defn: str
    input_path: str
    output_dp: str
    output_nondp: str
    config_file: str
    output_files: List[str]

    @pytest.fixture(autouse=True)
    def setup_test_correctness(self, privacy_defn: str):
        """Create temporary directories and create input files."""
        self.privacy_defn = privacy_defn
        self.input_path = tempfile.mkdtemp()
        self.output_dp = tempfile.mkdtemp()
        self.output_nondp = tempfile.mkdtemp()
        with open(os.path.join(self.input_path, "persons.csv"), "w") as f:
            f.write(PERSONS)
        with open(os.path.join(self.input_path, "units.csv"), "w") as f:
            f.write(UNITS)
        with open(os.path.join(self.input_path, "geo.csv"), "w") as f:
            f.write(GEO)
        if self.privacy_defn == "puredp":
            self.config_file = os.path.join(self.input_path, "config_puredp.json")
            with open(self.config_file, "w") as f:
                json.dump(CONFIG_PUREDP, f, indent=4)
                f.flush()
        else:
            assert self.privacy_defn == "zcdp"
            self.config_file = os.path.join(self.input_path, "config_zcdp.json")
            with open(self.config_file, "w") as f:
                json.dump(CONFIG_ZCDP, f, indent=4)
                f.flush()
        update_config_file(
            self.config_file,
            {"privacy_budget": get_config_privacy_budget_dict(float("inf"))},
        )
        self.output_files = [
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
        yield
        shutil.rmtree(self.input_path)
        if os.path.isdir(self.output_dp):
            shutil.rmtree(self.output_dp)
        if os.path.isdir(self.output_nondp):
            shutil.rmtree(self.output_nondp)

    @pytest.mark.slow
    def test_infinite_privacy_budget(self):
        """Test case for dp algorithm with infinite privacy_budget"""
        run_tabulation(
            config_path=self.config_file,
            data_path=self.input_path,
            output_path=self.output_nondp,
            private=False,
            should_validate_input=False,
            should_validate_private_output=False,
        )
        run_tabulation(
            config_path=self.config_file,
            data_path=self.input_path,
            output_path=self.output_dp,
            private=True,
            should_validate_input=False,
            should_validate_private_output=False,
        )
        for output_file in self.output_files:
            print(self.privacy_defn, output_file)
            df_merged = get_augmented_df(output_file, self.output_dp, self.output_nondp)
            print(df_merged)
            pd.testing.assert_series_equal(
                df_merged["NOISY"], df_merged["GROUND_TRUTH"], check_names=False
            )

    @pytest.mark.slow
    # This test is not run frequently based on the criticality of the test and runtime
    def test_exclude_states(self):
        """PHSafe can exclude specific states from tabulation."""
        include_states = {"29"}
        update_config_file(self.config_file, {"state_filter": ["29"]})
        run_tabulation(
            config_path=self.config_file,
            data_path=self.input_path,
            output_path=self.output_dp,
            private=True,
            should_validate_input=False,
            should_validate_private_output=True,
        )
        for output_file in self.output_files:
            print(self.privacy_defn, output_file)
            df = multi_read_csv(
                os.path.join(self.output_dp, output_file),
                dtype=str,
                sep="|",
                usecols=["REGION_TYPE", "REGION_ID"],
            )
            df = df[df["REGION_TYPE"] == "STATE"]
            actual = set(df["REGION_ID"].unique())
            assert actual == include_states

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "skip_tabulations, expected_subdir",
        [
            (TABULATIONS_KEY, []),  # All tabulations skipped
            (
                # PH5_num and PH8_num is computed since PH4 and PH7 is not skipped
                ["PH1_denom", "PH5_denom", "PH1_num", "PH6", "PH3", "PH2", "PH8_denom"],
                ["PH5_num", "PH7", "PH8_num", "PH4"],
            ),
        ],
    )
    def test_private_tabulation_zero_budget(
        self, skip_tabulations: List[str], expected_subdir: List[str]
    ):
        """Private run tabulates only tables with non-zero total budget."""
        privacy_budget_object = get_config_privacy_budget_dict(1)
        for tabulation in skip_tabulations:
            privacy_budget_object[tabulation] = {
                geo_iteration: float(0)
                for geo_iteration in privacy_budget_object[tabulation].keys()
            }
        update_config_file(self.config_file, {"privacy_budget": privacy_budget_object})
        run_tabulation(
            config_path=self.config_file,
            data_path=self.input_path,
            output_path=self.output_dp,
            private=True,
            should_validate_input=False,
            should_validate_private_output=True,
        )
        actual_subdir = os.listdir(self.output_dp)
        assert sorted(expected_subdir) == sorted(actual_subdir)

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "unskipped_tabulation, unskipped_geo_iteration",
        [("PH1_num", ["usa_A-G", "usa_*"]), ("PH2", ["state_*"])],
    )
    def test_private_tabulation_non_zero_budget_pop_group(
        self, unskipped_tabulation: str, unskipped_geo_iteration: List[str]
    ):
        """Per tabulation, population groups with zero budget are skipped."""
        privacy_budget_object = get_config_privacy_budget_dict(0)
        unskipped_geos = []
        unskipped_iterations = []
        for geo_iteration in unskipped_geo_iteration:
            privacy_budget_object[unskipped_tabulation][geo_iteration] = float("inf")
            region_type, level = geo_iteration.split("_")
            unskipped_geos.append(region_type.upper())
            unskipped_iterations.extend(ITERATION_CODES[level])

        update_config_file(self.config_file, {"privacy_budget": privacy_budget_object})
        run_tabulation(
            config_path=self.config_file,
            data_path=self.input_path,
            output_path=self.output_dp,
            private=True,
            should_validate_input=False,
            should_validate_private_output=True,
        )
        assert [unskipped_tabulation] == os.listdir(self.output_dp)
        df = multi_read_csv(
            os.path.join(self.output_dp, unskipped_tabulation), sep="|", dtype=str
        )
        assert all(x in list(set(unskipped_geos)) for x in df["REGION_TYPE"])
        if unskipped_tabulation not in ["PH2", "PH6"]:
            assert all(
                x in list(set(unskipped_iterations)) for x in df["ITERATION_CODE"]
            )

    @pytest.mark.slow
    # This test is not run frequently based on the criticality of the test and runtime
    def test_truncation_threshold(self):
        """Test case for truncation threshold."""
        tau_1, tau_2 = 4, 5
        persons = (
            "RTYPE,MAFID,QAGE,CENHISP,CENRACE,RELSHIP\n"
            + "3,100000001,17,1,01,25\n" * tau_1
            + "3,100000002,17,1,01,25\n" * tau_2
        )
        units = (
            "RTYPE,MAFID,FINAL_POP,NPF,HHSPAN,HHRACE,TEN,HHT,HHT2,CPLT\n"
            f"2,100000001,{tau_1},2,1,01,2,1,01,1\n"
            f"2,100000002,{tau_2},2,1,01,2,1,01,1"
        )
        geo = (
            "RTYPE,MAFID,TABBLKST,TABBLKCOU,TABTRACTCE,TABBLK,TABBLKGRPCE,REGIONCE,"
            "DIVISIONCE,PLACEFP,AIANNHCE\n"
            "2,100000001,44,055,450200,4069,1,4,2,99999,0001\n"
            "2,100000002,44,055,450200,4069,1,4,2,99999,0001"
        )
        with open(os.path.join(self.input_path, "persons.csv"), "w") as f:
            f.write(persons)
        with open(os.path.join(self.input_path, "units.csv"), "w") as f:
            f.write(units)
        with open(os.path.join(self.input_path, "geo.csv"), "w") as f:
            f.write(geo)
        output_path1 = tempfile.mkdtemp()
        output_path2 = tempfile.mkdtemp()
        update_config_file(
            self.config_file,
            {
                "tau": {
                    "PH1_num": tau_1,
                    "PH2": tau_1,
                    "PH3": tau_1,
                    "PH4": tau_1,
                    "PH6": tau_1,
                    "PH7": tau_1,
                }
            },
        )
        run_tabulation(
            config_path=self.config_file,
            data_path=self.input_path,
            output_path=output_path1,
            private=True,
            should_validate_input=False,
            should_validate_private_output=False,
        )
        update_config_file(
            self.config_file,
            {
                "tau": {
                    "PH1_num": tau_2,
                    "PH2": tau_2,
                    "PH3": tau_2,
                    "PH4": tau_2,
                    "PH6": tau_2,
                    "PH7": tau_2,
                }
            },
        )
        run_tabulation(
            config_path=self.config_file,
            data_path=self.input_path,
            output_path=output_path2,
            private=True,
            should_validate_input=False,
            should_validate_private_output=False,
        )
        for name in ["PH2", "PH3", "PH1_num", "PH4", "PH5_num", "PH6", "PH7"]:
            print(self.privacy_defn, name, "private join truncation")
            df1 = multi_read_csv(os.path.join(output_path1, name), sep="|", dtype=str)
            df2 = multi_read_csv(os.path.join(output_path2, name), sep="|", dtype=str)
            count1 = df1["COUNT"].astype(int)
            count2 = df2["COUNT"].astype(int)
            count1, count2 = count1[count1 > 0], count2[count2 > 0]
            assert any((count1.empty, count2.empty)) is False
            assert all(x == 2 * tau_1 for x in count1)
            assert all(x == tau_1 + tau_2 for x in count2)
        for name in ["PH8_num"]:
            print(self.privacy_defn, name, "sum truncation")
            df1 = multi_read_csv(os.path.join(output_path1, name), sep="|", dtype=str)
            df2 = multi_read_csv(os.path.join(output_path2, name), sep="|", dtype=str)
            sum1 = df1["COUNT"].astype(int)
            sum2 = df2["COUNT"].astype(int)
            sum1, sum2 = sum1[sum1 > 0], sum2[sum2 > 0]
            assert any((sum1.empty, sum2.empty)) is False
            assert all(x == tau_1 * 2 for x in sum1)
            assert all(x == tau_1 + tau_2 for x in sum2)
        shutil.rmtree(output_path1)
        shutil.rmtree(output_path2)

    # mode test. Add (False,) to pytest.mark.parametrize
    # This test is not run frequently as it takes longer than 10 minutes
    @pytest.mark.slow
    @pytest.mark.parametrize("private", [True])
    def test_output_format(self, private: bool):
        """PHSafe output files pass validation.

        See resources/config/output for details about the expected output formats.
        """
        output_path = self.output_dp if private else self.output_nondp
        run_tabulation(
            config_path=self.config_file,
            data_path=self.input_path,
            output_path=output_path,
            private=private,
            should_validate_input=False,
            should_validate_private_output=private,
            # validate ouputs as part of algorithm for private run
        )
        for output_file in self.output_files:
            print(self.privacy_defn, output_file)
            assert validate_directory_single_config(
                os.path.join(output_path, output_file),
                str(os.path.join(RESOURCES_DIR, f"config/output/{output_file}.json")),
                delimiter="|",
            )
