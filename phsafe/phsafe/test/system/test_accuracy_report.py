"""System test for :mod:`~.accuracy_report`."""

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
import tempfile

import pytest

from tmlt.common.pyspark_test_tools import pyspark  # pylint: disable=unused-import
from tmlt.phsafe.accuracy_report import create_phsafe_error_report
from tmlt.phsafe.paths import RESOURCES_DIR


@pytest.fixture(name="input_dir")
def fixture_input_dir():
    """Returns a temporary directory to use for input files."""
    with tempfile.TemporaryDirectory() as to_return:
        yield to_return


@pytest.fixture(name="output_dir")
def fixture_output_dir():
    """Returns a temporary directory to use for output files."""
    with tempfile.TemporaryDirectory() as to_return:
        yield to_return


ZERO_BUDGETS = {
    "PH1_num": {
        "usa_A-G": 0,
        "usa_H,I": 0,
        "usa_*": 0,
        "state_A-G": 0,
        "state_H,I": 0,
        "state_*": 0,
    },
    "PH1_denom": {
        "usa_A-G": 0,
        "usa_H,I": 0,
        "usa_*": 0,
        "state_A-G": 0,
        "state_H,I": 0,
        "state_*": 0,
    },
    "PH2": {"usa_*": 0, "state_*": 0},
    "PH3": {
        "usa_A-G": 0,
        "usa_H,I": 0,
        "usa_*": 0,
        "state_A-G": 0,
        "state_H,I": 0,
        "state_*": 0,
    },
    "PH4": {
        "usa_A-G": 0,
        "usa_H,I": 0,
        "usa_*": 0,
        "state_A-G": 0,
        "state_H,I": 0,
        "state_*": 0,
    },
    "PH5_denom": {
        "usa_A-G": 0,
        "usa_H,I": 0,
        "usa_*": 0,
        "state_A-G": 0,
        "state_H,I": 0,
        "state_*": 0,
    },
    "PH6": {"usa_*": 0, "state_*": 0},
    "PH7": {
        "usa_A-G": 0,
        "usa_H,I": 0,
        "usa_*": 0,
        "state_A-G": 0,
        "state_H,I": 0,
        "state_*": 0,
    },
    "PH8_denom": {
        "usa_A-G": 0,
        "usa_H,I": 0,
        "usa_*": 0,
        "state_A-G": 0,
        "state_H,I": 0,
        "state_*": 0,
    },
}


@pytest.mark.usefixtures("spark")
class TestAccuracyReport:
    """Tests for :mod:`~.accuracy_report`."""

    @pytest.mark.slow
    @pytest.mark.parametrize("config_file", ["config_zcdp.json", "config_puredp.json"])
    def test_create_phsafe_accuracy_report(self, config_file: str) -> None:
        """Smoke test :func:`~.accuracy_report.create_phsafe_accuracy_report`.

        Args:
            config_file: Name of the config file to use.
        """
        # <placeholder: test code>
        toy_dataset_path = os.path.join(RESOURCES_DIR, "toy_dataset")

        with tempfile.TemporaryDirectory() as output_path:
            create_phsafe_error_report(
                config_path=os.path.join(toy_dataset_path, config_file),
                data_path=toy_dataset_path,
                output_path=output_path,
            )

    def test_trials(self, spark, input_dir, output_dir):
        """Test to make sure multiple trials end up in the output."""
        toy_dataset_path = os.path.join(RESOURCES_DIR, "toy_dataset")
        config_path = os.path.join(input_dir, "config.json")
        num_trials = 2

        config = {
            "privacy_budget": ZERO_BUDGETS,
            "tau": {"PH1_num": 3, "PH2": 3, "PH3": 3, "PH4": 3, "PH6": 3, "PH7": 3},
            "state_filter": ["44", "29"],
            "reader": "csv",
            "privacy_defn": "zcdp",
        }
        config["privacy_budget"]["PH2"]["usa_*"] = 1
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)
            f.flush()

        create_phsafe_error_report(
            config_path, toy_dataset_path, output_dir, trials=num_trials
        )
        report = spark.read.csv(output_dir, header=True, sep="|").toPandas()
        assert all(report["COUNT"] == str(num_trials))
