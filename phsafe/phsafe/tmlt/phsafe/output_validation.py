"""Validates output files of PHSafe DP run.

Also creates updated versions of output config files which are used for further
validation.

See `Appendix A` for a description of each file.
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

# pylint: disable=no-name-in-module

import json
import logging
import os
from typing import Mapping, Optional, Sequence, Union

import smart_open
from pyspark.sql.functions import col
from typing_extensions import Literal

from tmlt.common.configuration import Config
from tmlt.common.validation import validate_spark_df
from tmlt.phsafe import TABULATION_OUTPUT_COLUMNS, PHSafeOutput
from tmlt.phsafe.paths import ALT_OUTPUT_CONFIG_DIR
from tmlt.phsafe.utils import setup_output_config_dir, validate_output_variance


def validate_output(
    output_sdfs: PHSafeOutput,
    privacy_budgets: Mapping[str, Mapping[str, float]],
    tau: Optional[Mapping[str, int]],
    privacy_defn: Optional[Union[Literal["puredp"], Literal["zcdp"]]],
    filter_states: Optional[Sequence[str]] = None,
) -> bool:
    """Return whether all outputs from algorithm execution are as expected.

    Note: This function requires `config.json` to be valid for running DP alogrithm. In
            particular, `privacy_budget` and `state_filter` must be present and valid
            in the config. And valid output_path from DP run is provided.

    Args:
        output_sdfs: NamedTuple that contains the private output spark dataframes
        privacy_budgets: Budget assigned per tabulation in config
        tau: Tau values assigned per tabulation in config
        privacy_defn: Type of privacy used
        filter_states: Optional. List of states included.
    """
    logger = logging.getLogger(__name__)
    logger.info("Validating PHSafe outputs ...")
    setup_output_config_dir()
    logger.info("Starting output validation...")
    logger.info("Checking that required output folders are present in output path...")

    # Read privacy budget allocation. If any table has no budget assigned, PHSafe
    # will not produce output for that table.
    total_privacy_budget_per_tabulation = {}
    for tabulation, geo_iteration_budget_dict in privacy_budgets.items():
        if tabulation not in ["PH5_num", "PH8_num"]:
            total_privacy_budget_per_tabulation[tabulation] = sum(
                geo_iteration_budget_dict.values()
            )

    # PH5_num and PH8_num will be present only if PH4 and PH7 have non-zero
    # budget respectively.
    total_privacy_budget_per_tabulation[
        "PH5_num"
    ] = total_privacy_budget_per_tabulation["PH4"]
    total_privacy_budget_per_tabulation[
        "PH8_num"
    ] = total_privacy_budget_per_tabulation["PH7"]

    output_sdfs_dict = output_sdfs._asdict()

    actual_output_subdirs = list(
        {tabulation for tabulation in output_sdfs_dict if output_sdfs_dict[tabulation]}
    )
    expected_output_subdirs = []
    for tabulation in TABULATION_OUTPUT_COLUMNS:
        if total_privacy_budget_per_tabulation[tabulation] != 0:
            expected_output_subdirs.append(tabulation)

    if len(actual_output_subdirs) > len(set(expected_output_subdirs)):
        extra_subdirs = sorted(
            set(actual_output_subdirs) - set(expected_output_subdirs)
        )
        extra_subdirs_str = "{" + ", ".join(extra_subdirs) + "}"
        logger.error(
            f"Invalid output: Additional output folders present {extra_subdirs_str}."
        )
        return False

    missing_subdirs = sorted(set(expected_output_subdirs) - set(actual_output_subdirs))

    if missing_subdirs:
        missing_subdirs_str = "{" + ", ".join(missing_subdirs) + "}"
        logger.error(
            f"Invalid output: missing required output folders {missing_subdirs_str}."
        )
        return False
    logger.info("All required output folders present.")

    logger.info(
        "Outputs are checked for expected formats as per Appendix A output spec..."
    )
    okay = True
    for output_file in actual_output_subdirs:
        okay &= validate_spark_df(
            output_file,
            output_sdfs_dict[output_file],
            Config.load_json(
                os.path.join(ALT_OUTPUT_CONFIG_DIR, f"{output_file}.json")
            ),
            unexpected_column_strategy="error",
            check_column_order=True,
        )
    if not okay:
        logger.error("Invalid output: Not as per expected format. See above.")
        return False
    logger.info("All generated outputs are as per prior expectation.")

    if filter_states is not None:
        logger.info(
            "Outputs are checked to ensure appropriate states are tabulated as per the "
            "config state_filter flag..."
        )
        okay = True
        for output_file in actual_output_subdirs:
            if output_sdfs_dict[output_file] is not None:
                okay &= (
                    output_sdfs_dict[output_file]
                    .filter(
                        (
                            col("REGION_TYPE").isin(["STATE"])
                            & ~col("REGION_ID").isin(filter_states)
                        )
                    )
                    .rdd.isEmpty()
                )

        if not okay:
            logger.error(
                "Invalid output: Output contains states not in the config state_filter"
                " flag."
            )
            return False
    if tau is not None:
        logger.info("Checking that output records have the correct variance...")
        with smart_open.open(
            os.path.join(ALT_OUTPUT_CONFIG_DIR, f"validate_{privacy_defn}.json")
        ) as f:
            validation_config: Mapping[str, Mapping[str, str]] = json.load(f)
        okay = validate_output_variance(
            privacy_budget=privacy_budgets,
            taus=tau,
            data=output_sdfs_dict,
            validation_config=validation_config,
        )
        if not okay:
            logger.error("Variance validation failed.")
            return False
        logger.info("Variance is as expected.")
    logger.info("Output validation successful. All output files are as expected.")
    return True
