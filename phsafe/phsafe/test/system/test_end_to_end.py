"""Checks the output from test_end_to_end.sh,
which runs PHSafe as expected in prodution.
"""

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

import glob
import subprocess
import unittest
from pathlib import Path

import pandas as pd
import pytest

from tmlt.common.pyspark_test_tools import pyspark  # pylint: disable=unused-import
from tmlt.common.pyspark_test_tools import assert_frame_equal_with_sort

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
# Note: the expected output file names match the OUTPUT_DIRS list.


@pytest.mark.usefixtures("spark")
class TestSessionInit:
    """Tests that PHSafe can run from a shell script with true input files."""

    @pytest.mark.slow
    @pytest.mark.parametrize("test", ["test1", "test2"])
    def test_output(self, test: str):
        """This test checks that output of a PHSafe
        run exactly equals the expected output."""

        test_dir = Path(__file__).parent

        subprocess.run(
            ["bash", f"{test_dir}/end_to_end_inputs/{test}/test_end_to_end.sh"],
            check=True,
        )

        for directory in OUTPUT_DIRS:
            outputs_dir = f"{test_dir}/end_to_end_inputs/{test}/outputs/{directory}"

            # Gets a list of the csv output files.
            results = glob.glob(f"{outputs_dir}/*.csv")

            # A dictionary to help Pandas import/export the DataFrame outputs exactly.
            column_types = {
                "REGION_ID": "string",
                "REGION_TYPE": "string",
                "ITERATION_CODE": "string",
                "COUNT": "Int32",
            }
            data_cell_col = directory.upper() + "_DATA_CELL"
            column_types.update({data_cell_col: "string"})

            # Multiple parts files can be published,
            # this combines these into one output.
            results_df = pd.DataFrame()
            for file in results:
                file_df = pd.read_csv(file, delimiter="|", dtype=column_types)
                results_df = pd.concat([results_df, file_df], axis=0)

            expected_df = pd.read_csv(
                f"{test_dir}/end_to_end_inputs/{test}/expected_outputs/{directory}.csv",
                delimiter="|",
                dtype=column_types,
            )

            assert_frame_equal_with_sort(results_df, expected_df)

            # Below the output is sorted so that file diff's with a Golden Test work.
            # The outputs are git ignored so this is helpful for creating and
            # modifying tests in the future.
            results_df = sort_ph_output(results_df)
            results_df.to_csv(
                f"{test_dir}/end_to_end_inputs/{test}"
                f"/outputs/{directory}/{directory}.csv",
                sep="|",
                index=False,
            )


def sort_ph_output(df: pd.DataFrame):
    """Takes in a PHSafe output DF and sorts it."""
    columns = list(df.columns)
    columns.remove("COUNT")
    end_df = df.set_index(columns).sort_index().reset_index()
    return end_df


if __name__ == "__main__":
    unittest.main()
