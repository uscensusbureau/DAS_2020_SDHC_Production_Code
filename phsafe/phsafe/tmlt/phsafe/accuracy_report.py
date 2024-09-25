"""Utilities for comparing the results from tmlt.phsafe against the ground truth.

A full error report can be run from the command line using the following command:

> python accuracy_report.py <config_path> <data_path> <output_path>

As this report uses the ground truth counts, it violates differential privacy,
and should not be created using sensitive data. Rather its purpose is to test
PHSafe on non-sensitive or synthetic datasets to help tune the algorithms and
to predict the performance on the private data.
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
# pylint: disable=no-name-in-module, redefined-builtin

import argparse
from typing import Callable

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql.functions import abs, col, count, lit, udf
from pyspark.sql.pandas.functions import pandas_udf
from pyspark.sql.types import FloatType

from tmlt.phsafe import PHSafeInput, PHSafeOutput
from tmlt.phsafe.input_processing import (
    PrivatePHSafeParameters,
    process_private_input_parameters,
)
from tmlt.phsafe.nonprivate_tabulations import NonPrivateTabulations
from tmlt.phsafe.private_tabulations import PrivateTabulations
from tmlt.phsafe.runners import _get_phsafe_input, import_cef_reader, preprocess_geo
from tmlt.phsafe.utils import get_augmented_df_from_spark

TABULATION_CELL_NAMES = {
    "PH1_num": "PH1_NUM_DATA_CELL",
    "PH1_denom": "PH1_DENOM_DATA_CELL",
    "PH2": "PH2_DATA_CELL",
    "PH3": "PH3_DATA_CELL",
    "PH4": "PH4_DATA_CELL",
    "PH5_num": "PH5_NUM_DATA_CELL",
    "PH5_denom": "PH5_DENOM_DATA_CELL",
    "PH6": "PH6_DATA_CELL",
    "PH7": "PH7_DATA_CELL",
    "PH8_num": "PH8_NUM_DATA_CELL",
    "PH8_denom": "PH8_DENOM_DATA_CELL",
}
TABULATION_TABLE_NAMES = {
    "PH1_num": "PH1",
    "PH1_denom": "PH1",
    "PH2": "PH2",
    "PH3": "PH3",
    "PH4": "PH4",
    "PH5_num": "PH5",
    "PH5_denom": "PH5",
    "PH6": "PH6",
    "PH7": "PH7",
    "PH8_num": "PH8",
    "PH8_denom": "PH8",
}

ITERATION_CODE_TO_LEVEL = {
    "A": "A-G",
    "B": "A-G",
    "C": "A-G",
    "D": "A-G",
    "E": "A-G",
    "F": "A-G",
    "G": "A-G",
    "H": "H,I",
    "I": "H,I",
    "*": "*",
}


@pandas_udf(FloatType())
def quantile(s: pd.Series) -> float:
    """Quantile aggregator.

    Args:
        s: Pandas series to aggregate with quantile.
    """
    return s.quantile(0.90, interpolation="linear")


def create_phsafe_error_report(
    config_path: str, data_path: str, output_path: str, trials: int = 1
):
    """Create an aggregated error report by comparing two runs of PHSafe.

    Runs an error report for PHSafe using and aggregates the results. The error report
    is saved to the provided output path.

    Args:
        config_path: the path containing the PHSafe config.json.
        data_path: If csv reader, the location of input files.
            If cef reader, the file path to the reader config.
        output_path: the path where the error reports should be saved.
        trials: the number of times to run the private tabulations.
    """
    # Process inputs.
    params = process_private_input_parameters(config_path, data_path, output_path)
    reader = import_cef_reader(params.reader)
    input_sdfs = _get_phsafe_input(params, reader)

    # Run the private and non-private algorithms.
    nonprivate_answer = NonPrivateTabulations()(
        PHSafeInput(
            persons=input_sdfs.persons,
            units=input_sdfs.units,
            geo=preprocess_geo(input_sdfs.geo),
        )
    )
    runs_with_error = None
    for _ in range(trials):
        private_answer = PrivateTabulations(
            tau=params.tau,
            privacy_budget=params.privacy_budget,
            privacy_defn=params.privacy_defn,
        )(
            PHSafeInput(
                persons=input_sdfs.persons,
                units=input_sdfs.units,
                geo=preprocess_geo(input_sdfs.geo),
            )
        )

        # Create the unaggregated error report. This joins the private and non-private
        # results and stacks all the tables into one.
        run_with_error = create_unaggregated_error_report(
            private_answer, nonprivate_answer
        )
        # We want the results with error all sombined into a single dataframe so we
        # can aggregate them.
        if not runs_with_error:
            runs_with_error = run_with_error
        else:
            runs_with_error = runs_with_error.unionAll(run_with_error)

    # Aggregate the error report.
    aggregated_error_report = aggregate_error_report(runs_with_error)

    # Add expected MOE
    get_expected_moe = udf(create_get_expected_error(params), FloatType())
    aggregated_error_report = aggregated_error_report.withColumn(
        "EXPECTED_MOE",
        # pylint: disable=too-many-function-args,redundant-keyword-arg
        get_expected_moe(
            col("TABLE"), col("ITERATION_CODE"), col("REGION_TYPE"), col("DATA_CELL")
        ),
    )

    aggregated_error_report.repartition(1).write.csv(
        output_path, sep="|", header=True, mode="overwrite"
    )


def create_unaggregated_error_report(
    private_answer: PHSafeOutput, nonprivate_answer: PHSafeOutput
) -> DataFrame:
    """Create an an unaggregated error report from private and non-private runs.

    Results from all tables are stacked into one dataframe.

    Args:
        private_answer: the results from the private run.
        nonprivate_answer: the results from the non-private run.
    """
    error_report = None
    private_answer_dict = private_answer._asdict()
    nonprivate_answer_dict = nonprivate_answer._asdict()
    for tabulation in (
        x for x in private_answer_dict if private_answer_dict[x] is not None
    ):
        augmented_tabulation = (
            get_augmented_df_from_spark(
                private_answer_dict[tabulation], nonprivate_answer_dict[tabulation]
            )
            .withColumn("TABLE", lit(tabulation))
            .withColumn("ERROR", abs(col("NOISY") - col("GROUND_TRUTH")))
            .withColumnRenamed(TABULATION_CELL_NAMES[tabulation], "DATA_CELL")
        )
        if "ITERATION_CODE" not in augmented_tabulation.columns:
            augmented_tabulation = augmented_tabulation.withColumn(
                "ITERATION_CODE", lit("*")
            )
        augmented_tabulation = augmented_tabulation.select(
            "TABLE", "REGION_ID", "REGION_TYPE", "ITERATION_CODE", "DATA_CELL", "ERROR"
        )

        if error_report is not None:
            error_report = error_report.unionByName(augmented_tabulation)
        else:
            error_report = augmented_tabulation

    if error_report is None:
        raise ValueError("No tables were provided.")
    return error_report


def aggregate_error_report(error_report: DataFrame) -> DataFrame:
    """Aggregates the error report to get MOEs.

    Args:
        error_report: An error report that contains absolute error.
    """
    # We don't aggregate over the data cell because PH8_num has different error for
    # different data cells. We need to separate aggregations because pandas UDF
    # aggregations can't be mixed with built-in aggregations.
    grouping_columns = ["TABLE", "REGION_TYPE", "ITERATION_CODE", "DATA_CELL"]
    grouped_error_report = error_report.groupBy(grouping_columns)
    df1 = grouped_error_report.agg(quantile("ERROR").alias("EXPERIMENTAL_MOE"))
    df2 = grouped_error_report.agg(count("ERROR").alias("COUNT"))
    return df1.join(df2, on=grouping_columns)


def create_get_expected_error(
    params: PrivatePHSafeParameters,
) -> Callable[[str, str, str, int], float]:
    """Create a udf that returns the expected MOE.

    Args:
        params: the parameters used to run PHSafe.
    """

    def get_expected_error(
        table: str, iteration_code: str, region_type: str, data_cell: int
    ) -> float:
        """Returns the expected MOE.

        Args:
            table: the table name.
            iteration_code: the iteration code.
            region_type: the region type.
            data_cell: the data cell.
        """
        if table == "PH5_num":
            # PH5_num is a copy of PH4.
            return get_expected_error("PH4", iteration_code, region_type, data_cell)
        if table == "PH8_num":
            # For PH8_num cell 3 (renter occupied) the cell is a direct copy of PH7.
            if data_cell == 3:
                return get_expected_error("PH7", iteration_code, region_type, data_cell)
            # For PH8_num cell 2 (owner ocuppied), the cell is a sum of a pair of PH7
            # values. We estimate the MOE using MOE_PH8_num = sqrt(2*MOE_PH7^2)) =
            # sqrt(2) * MOE_PH7.
            assert data_cell == 2
            return round(
                float(
                    np.sqrt(2)
                    * get_expected_error("PH7", iteration_code, region_type, data_cell)
                ),
                1,
            )

        def get_gaussian_moe(sensitivity: int, rho: float) -> float:
            """Returns the Gaussian MOE for a given sensitivity and rho.

            Args:
                sensitivity: The sensitivity of a query in the table.
                rho: The privacy budget.
            """
            return 1.64 * sensitivity / np.sqrt(2 * rho)

        def get_geometric_moe(sensitivity: int, epsilon: float) -> float:
            """Returns the Geometric MOE for a given sensitivity and epsilon.

            Args:
                sensitivity: The sensitivity of a query in the table.
                epsilon: The privacy budget.
            """
            b = sensitivity / epsilon
            return b * (np.log(20 / (1 + np.exp(1 / b))))

        moe_function = (
            get_geometric_moe if params.privacy_defn == "puredp" else get_gaussian_moe
        )
        sensitivity = (
            2 * params.tau[table] + 2
            if table in ("PH1_num", "PH2", "PH3", "PH4", "PH6", "PH7")
            else 2
        )
        cell_budget = params.privacy_budget[table][
            f"{region_type.lower()}_{ITERATION_CODE_TO_LEVEL[iteration_code]}"
        ]
        return round(float(moe_function(sensitivity, cell_budget)), 1)

    return get_expected_error


def main():
    """Parse arguments and run the error report."""
    parser = argparse.ArgumentParser(prog="ph-safe error report")
    parser.add_argument(dest="config_file", help="path to PHSafe config file", type=str)
    parser.add_argument(
        dest="data_path", help="input csv files directory path", type=str
    )
    parser.add_argument(
        dest="output_path",
        help="name of directory that contains all output files",
        type=str,
    )
    parser.add_argument(
        "--trials",
        dest="trials",
        help=(
            "The number of times to run the private computations in the accuracy"
            " report."
        ),
        type=int,
        default=1,
    )
    args = parser.parse_args()
    create_phsafe_error_report(
        args.config_file, args.data_path, args.output_path, args.trials
    )


if __name__ == "__main__":
    main()
