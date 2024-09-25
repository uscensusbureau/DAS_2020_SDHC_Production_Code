"""Validates input files to PHSafe."""

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

import logging
import os
from typing import List, Optional, Sequence

from pyspark.sql.dataframe import DataFrame
from pyspark.sql.functions import col

from tmlt.common.configuration import Config, Unrestricted
from tmlt.common.schema import Schema
from tmlt.common.validation import validate_spark_df
from tmlt.phsafe import PHSafeInput
from tmlt.phsafe.paths import ALT_INPUT_CONFIG_DIR
from tmlt.phsafe.utils import setup_input_config_dir

CONFIG_PERSON = "persons"
"""Constant used as the dictionary key for the person table's config"""

CONFIG_UNIT = "units"
"""Constant used as the dictionary key for the unit table's config"""

CONFIG_GEO = "geo"
"""Constant used as the dictionary key for the geo table's config"""


def _get_mafid_domain(input_df: DataFrame) -> List[int]:
    """Return the domain of mafid.

    Args:
        input_df: The spark dataframe.
    """
    return list(set(input_df.select("MAFID").toPandas()["MAFID"]))


def validate_schema(df: DataFrame, config: Config, name: str) -> bool:
    """Validates the schema of the dataframe against the config.

    Args:
        df: The spark dataframe.
        config: The config object.
        name: The name of the df and config.
    """
    logger = logging.getLogger(__name__)

    all_column_config = Config(
        [
            config[column] if column in config.columns else Unrestricted(column)
            for column in df.columns
        ]
    )
    all_column_schema = Schema.from_config_object(all_column_config)
    equal_schema = all_column_schema.spark_schema == df.schema
    if not equal_schema:
        message = (
            f"Schema for {name} is not equal to schema of Config provided.\n"
            f"sdf schema: {df.schema}\n"
            f"config schema: {all_column_schema.spark_schema}"
        )
        logger.error(message)

    return equal_schema


def validate_input(
    input_sdfs: PHSafeInput, filter_states: Optional[Sequence[str]] = None
) -> bool:
    """Return whether all input spark dataframes (sdfs) are consistent and as expected.

    Validates using a three-step process

    1. Checks that the type of the input sdfs matches the schema.
    2. Uses our prior knowledge to validate the input sdfs using prebuilt
       schemas. For some columns we know the full domain (for instance QAGE),
       for other columns we only know the expected format.
    3. Validate again using the input sdfs. The following properties are checked:
       - Every MAFID in the persons dataframe exists in the units dataframe.
       - Every MAFID in the geo dataframe exists in the units dataframe.
       - Every MAFID in the units dataframe exists in the geo dataframe.
       - Every state in the geo dataframe exists in the state filter.
       - No MAFID appears twice in either the units or geo dataframe.

    Args:
        input_sdfs: A PHSafeInput NamedTuple that contains the input spark dataframes,
            persons, units, and geo.
        filter_states: Optional. List of states included.
    """
    logger = logging.getLogger(__name__)
    setup_input_config_dir()
    config = {}
    config[CONFIG_PERSON] = Config.load_json(
        os.path.join(ALT_INPUT_CONFIG_DIR, "persons.json")
    )
    config[CONFIG_UNIT] = Config.load_json(
        os.path.join(ALT_INPUT_CONFIG_DIR, "units.json")
    )
    config[CONFIG_GEO] = Config.load_json(
        os.path.join(ALT_INPUT_CONFIG_DIR, "geo.json")
    )

    logger.info(
        "Starting phase 1 of input validation. Checking that types of sdf columns "
        "are correct based on prior expectations..."
    )
    okay = True
    okay &= validate_schema(input_sdfs.persons, config[CONFIG_PERSON], CONFIG_PERSON)
    okay &= validate_schema(input_sdfs.units, config[CONFIG_UNIT], CONFIG_UNIT)
    okay &= validate_schema(input_sdfs.geo, config[CONFIG_GEO], CONFIG_GEO)
    if not okay:
        logger.error("Errors found in phase 1. See above.")
        return False
    logger.info("Phase 1 successful.")

    logger.info(
        "Starting phase 2 of input validation. Checking spark dataframes "
        "against prior expectations..."
    )
    okay = True
    okay &= validate_spark_df(
        CONFIG_PERSON,
        input_sdfs.persons,
        config[CONFIG_PERSON],
        unexpected_column_strategy="error",
        check_column_order=True,
    )
    okay &= validate_spark_df(
        CONFIG_UNIT,
        input_sdfs.units,
        config[CONFIG_UNIT],
        unexpected_column_strategy="error",
        check_column_order=True,
    )
    okay &= validate_spark_df(
        CONFIG_GEO,
        input_sdfs.geo,
        config[CONFIG_GEO],
        unexpected_column_strategy="error",
        check_column_order=True,
    )
    if not okay:
        logger.error("Errors found in phase 2. See above.")
        return False
    logger.info("Phase 2 successful.")
    logger.info(
        "Starting phase 3 of input validation. Checking that input files are"
        "consistent with each other..."
    )
    person_mafids = input_sdfs.persons.select("MAFID")
    unit_mafids = input_sdfs.units.select("MAFID")
    geo_mafids = input_sdfs.geo.select("MAFID")

    # 0. Check that all MAFIDs in the units and geos dataframes are unique.
    duplicate_unit_mafids = (
        unit_mafids.groupBy("MAFID").count().where(col("count") > 1).collect()
    )
    if duplicate_unit_mafids:
        okay = False
        logger.error(
            f"Found two rows with the same MAFID in the {CONFIG_UNIT} file: "
            f"{sorted([row.MAFID for row in duplicate_unit_mafids])}"
        )

    duplicate_geo_mafids = (
        geo_mafids.groupBy("MAFID").count().where(col("count") > 1).collect()
    )
    if duplicate_geo_mafids:
        okay = False
        logger.error(
            f"Found two rows with the same MAFID in the {CONFIG_GEO} file: "
            f"{sorted([row.MAFID for row in duplicate_geo_mafids])}"
        )

    # 1. Check that every MAFID in the persons dataframe exists in the units dataframe.
    invalid_mafids = (
        person_mafids.join(unit_mafids, on="MAFID", how="left_anti")
        .distinct()
        .collect()
    )
    if invalid_mafids:
        okay = False
        logger.error(
            f"Found MAFIDs in the {CONFIG_PERSON} file that "
            f"are not in the {CONFIG_UNIT} file: "
            f"{sorted([row.MAFID for row in invalid_mafids])}"
        )

    # Note that the next two checks could be combined into one check as an outer join,
    # but
    # 1. we want to know which input is missing the MAFIDs (this makes that easier)
    # 2. the left anti joins are simpler than an outer join with the checks for nulls
    # 3. It isn't clear that an outer join would be faster than two left anti joins
    #    (but it might be)

    # 2. Check that every MAFID in the geo dataframe exists in the units dataframe.
    invalid_mafids = (
        geo_mafids.join(unit_mafids, on="MAFID", how="left_anti").distinct().collect()
    )
    if invalid_mafids:
        okay = False
        logger.error(
            f"Found MAFIDs in the {CONFIG_GEO} file that "
            f"are not in the {CONFIG_UNIT} file: "
            f"{sorted([row.MAFID for row in invalid_mafids])}"
        )

    # 3. Check that every MAFID in the units dataframe exists in the geo dataframe.
    invalid_mafids = (
        unit_mafids.join(geo_mafids, on="MAFID", how="left_anti").distinct().collect()
    )
    if invalid_mafids:
        okay = False
        logger.error(
            f"Found MAFIDs in the {CONFIG_UNIT} file that "
            f"are not in the {CONFIG_GEO} file: "
            f"{sorted([row.MAFID for row in invalid_mafids])}"
        )

    # 4. Check that every state in the geo dataframe exists in the state filter
    if filter_states is not None:
        # There are only 50 states + PR, so doing the check locally in memory is fine.
        states = input_sdfs.geo.select("TABBLKST").distinct().collect()
        invalid_states = [
            state.TABBLKST for state in states if state.TABBLKST not in filter_states
        ]
        if invalid_states:
            okay = False
            logger.error(
                f"Found TABBLKSTs in the {CONFIG_GEO} file that "
                f"are not in the state filter: {sorted(invalid_states)}"
            )
    if not okay:
        logger.error("Errors found in phase 3. See above.")
        return False
    logger.info("Phase 3 successful. All files are as expected.")
    return True
