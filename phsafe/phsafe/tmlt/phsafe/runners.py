"""Module for end-to-end phsafe algorithms."""

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

# pylint: disable=ungrouped-imports,import-outside-toplevel

import logging
import os
from typing import Any, Union

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, lit  # pylint: disable=no-name-in-module

from tmlt.phsafe import PHSafeInput, PHSafeOutput
from tmlt.phsafe.input_processing import (
    NonprivatePHSafeParameters,
    PrivatePHSafeParameters,
    process_nonprivate_input_parameters,
    process_private_input_parameters,
)
from tmlt.phsafe.input_validation import validate_input
from tmlt.phsafe.nonprivate_tabulations import NonPrivateTabulations
from tmlt.phsafe.output_validation import validate_output
from tmlt.phsafe.private_tabulations import PrivateTabulations
from tmlt.phsafe.utils import setup_input_config_dir


def import_cef_reader(reader: str) -> Any:
    """This function imports the CEFReader if specified, otherwise the CSV reader.

    Args:
        reader: The reader type to use with PHSafe.
    """
    if reader == "cef":
        try:
            from phsafe_safetab_reader.cef_reader import CEFReader  # type: ignore
        except ImportError as e:
            logging.info(
                "Failed to import CEFReader from module "
                f"phsafe_safetab_reader.cef_reader: {e}. Please verify the "
                "phsafe_safetab_reader.cef_reader module has been placed in your "
                "python path."
            )
        return CEFReader
    else:
        from tmlt.phsafe.csv_reader import CSVReader

        return CSVReader


def preprocess_geo(geo_df: DataFrame) -> DataFrame:
    """Returns processed geography dataframe.

    Args:
        geo_df: input geography dataframe
    """
    # The filter removes group quarters only geographies.
    return geo_df.filter(~col("RTYPE").isin("4", "5")).select(
        col("MAFID"), lit("1").alias("USA"), col("TABBLKST").alias("STATE")
    )


def _get_phsafe_input(
    params: Union[PrivatePHSafeParameters, NonprivatePHSafeParameters], file_reader: Any
) -> PHSafeInput:
    """Reads input dataframes with PHSafe parameters.

    Args:
        params: PHSafe parameters for reading input dataframes.
        file_reader: The reader type to use with PHSafe.
    """
    reader = file_reader(params.data_path, params.state_filter)
    return PHSafeInput(
        persons=reader.get_person_df(),
        units=reader.get_unit_df(),
        geo=reader.get_geo_df(),
    )


def _write_answers(answers: PHSafeOutput, output_path: str):
    """Compute and write out PHSafe tabulations.

    Args:
        answers: Answers to be written out.
        output_path: Path to output directory.
    """
    for tabulation, answer in answers._asdict().items():
        if answer is not None:
            answer.repartition(1).write.csv(
                os.path.join(output_path, tabulation),
                sep="|",
                header=True,
                mode="overwrite",
            )


def run_tabulation(
    config_path: str,
    data_path: str,
    output_path: str,
    should_validate_input: bool,
    should_validate_private_output: bool,
    private: bool,
):
    """Runs dp algorithm for person in household queries.

    Args:
        config_path: Path to json config.
        data_path: Path to input directory.
        output_path: Path to output directory.
        should_validate_input: If True, validate inputs before tabulations.
        should_validate_private_output: If True, validate
        private output after tabulations.
        private: If True, compute DP tabulations.
    """
    setup_input_config_dir()

    params: Union[PrivatePHSafeParameters, NonprivatePHSafeParameters]
    tabulator: Union[PrivateTabulations, NonPrivateTabulations]
    if private:
        params = process_private_input_parameters(config_path, data_path, output_path)
        tabulator = PrivateTabulations(
            tau=params.tau,
            privacy_budget=params.privacy_budget,
            privacy_defn=params.privacy_defn,
        )

    else:
        params = process_nonprivate_input_parameters(
            config_path, data_path, output_path
        )
        tabulator = NonPrivateTabulations()

    reader = import_cef_reader(params.reader)

    input_sdfs = _get_phsafe_input(params, reader)  # type: ignore
    if should_validate_input:
        if not validate_input(input_sdfs, params.state_filter):
            raise RuntimeError("Validation failed.")

    output_sdfs = tabulator(
        PHSafeInput(
            persons=input_sdfs.persons,
            units=input_sdfs.units,
            geo=preprocess_geo(input_sdfs.geo),
        )
    )
    _write_answers(output_sdfs, params.output_path)
    if private and should_validate_private_output:
        # help out mypy
        assert isinstance(params, PrivatePHSafeParameters)
        if not validate_output(
            output_sdfs=output_sdfs,
            privacy_budgets=params.privacy_budget,
            tau=params.tau,
            filter_states=params.state_filter,
            privacy_defn=params.privacy_defn,
        ):
            raise RuntimeError("Output validation Failed.")
    logging.info("Execution completed successfully.")


def run_input_validation(config_path: str, data_path: str):
    """Validate config and inputs.

    Note: This function requires `config.json` to be valid for running DP alogrithm. In
        particular, `privacy_budget` and `tau` must be present and valid in the config.

    Args:
        config_path: Path to `config.json`.
        data_path: Path used to configure the reader.
    """
    params = process_private_input_parameters(config_path, data_path, "")
    reader = import_cef_reader(params.reader)
    input_sdfs = _get_phsafe_input(params, reader)
    if not validate_input(input_sdfs, params.state_filter):
        raise RuntimeError("Input validation Failed.")
    logging.info("Input validation completed successfully.")
