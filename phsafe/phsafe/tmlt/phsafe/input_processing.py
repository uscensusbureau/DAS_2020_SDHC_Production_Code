"""Module for processing algorithm input parameters."""

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
import logging
from dataclasses import dataclass
from typing import Any, Collection, Dict, List, Mapping, Union

from pyspark.sql import SparkSession
from smart_open import open as smart_open
from typing_extensions import Literal

from tmlt.common.io_helpers import is_s3_path
from tmlt.phsafe import TABULATIONS_KEY
from tmlt.phsafe.utils import get_config_privacy_budget_dict

STATE_FILTER_FLAG = "state_filter"
"""Key in the config.json that indicates state filtering."""

READER_FLAG = "reader"
"""Key in the config.json that indicates the reader being used."""

PRIVACY_DEFN_FLAG = "privacy_defn"
"""Key in the config.json that indicate the privacy definition being used."""

PRIVACY_BUDGET_KEY = "privacy_budget"
"""Key in the config.json for privacy budget."""

TAU_KEY = "tau"
"""Key in the config.json for max persons per household."""

TRUNCATIONS_KEY = {"PH1_num", "PH2", "PH3", "PH4", "PH6", "PH7"}
"""Nested keys in config.json that indicate tabulations with truncation thresholds."""

GEO_SCHEMA = {
    "RTYPE": "VARCHAR",
    "MAFID": "INTEGER",
    "TABBLKST": "VARCHAR",
    "TABBLKCOU": "VARCHAR",
    "TABTRACTCE": "VARCHAR",
    "TABBLK": "VARCHAR",
    "TABBLKGRPCE": "VARCHAR",
    "REGIONCE": "VARCHAR",
    "DIVISIONCE": "VARCHAR",
    "PLACEFP": "VARCHAR",
    "AIANNHCE": "VARCHAR",
}
"""Analytics schema for the geo df."""

UNIT_SCHEMA = {
    "RTYPE": "VARCHAR",
    "MAFID": "INTEGER",
    "FINAL_POP": "INTEGER",
    "NPF": "INTEGER",
    "HHSPAN": "INTEGER",
    "HHRACE": "VARCHAR",
    "TEN": "VARCHAR",
    "HHT": "VARCHAR",
    "HHT2": "VARCHAR",
    "CPLT": "VARCHAR",
}
"""Analytics schema for the unit df."""

PERSON_SCHEMA = {
    "RTYPE": "VARCHAR",
    "MAFID": "INTEGER",
    "QAGE": "INTEGER",
    "CENHISP": "INTEGER",
    "CENRACE": "VARCHAR",
    "RELSHIP": "VARCHAR",
}
"""Analytics schema for the person df."""


def _parse_config_json(
    config_path: str, config_keys: Collection[str]
) -> Dict[str, Any]:
    """Parses, validates, and returns configuration from the given file.

    Args:
        config_path: The relative or absolute path to the PHSafe config file.
        config_keys: The list of keys that should be present in the configuration;
            all configuration values not on this list are dropped from the
            configuration dictionary, and it is an error if any values in this list
            are not specified in the config file.

    Raises:
        OSError: When the given config file does not exist or could not be read.
        JSONDecodeError: When the given config file is not a valid JSON file.
        RuntimeError: When the configuration in the given file is not valid.
    """
    logger = logging.getLogger(__name__)
    logger.info("Validating PHSafe config file.")

    with smart_open(config_path, "r") as fp:
        config_json = json.load(fp)

    missing_keys = set(config_keys) - set(config_json)
    if missing_keys:
        missing_keys_str = "{" + ", ".join(missing_keys) + "}"
        raise RuntimeError(
            f"Config validation failed on {config_path}: "
            f"missing keys {missing_keys_str}"
        )

    unused_keys = set(config_json) - set(config_keys)
    if unused_keys:
        unused_keys_str = "{" + ", ".join(unused_keys) + "}"
        logger.info(
            f"Ignoring unneeded configuration keys {unused_keys_str} from {config_path}"
        )

    config = {k: config_json[k] for k in config_keys}
    config = _validate_config_values(config, config_path)
    logger.info("PHSafe config file validation successful.")
    return config


def _validate_config_values(config: Dict[str, Any], config_path: str) -> Dict[str, Any]:
    """Validate settings in a config dict, performing safe type conversions.

    Validates the config values in the given config dictionary, checking that
    they are both the appropriate types and that they have allowed values. The
    returned configuration is identical to the provided one, but some config
    values may have undergone type conversion to their expected types.

    Currently, config values for the following keys are validated:
     * STATE_FILTER_FLAG
     * READER_FLAG
     * PRIVACY_DEFN_FLAG
     * PRIVACY_BUDGET_KEY
     * TAU_KEY
    Any other config values are returned without validation.

    Args:
        config: A configuration dictionary to validate. It is assumed that the keys
            in this dictionary have been validated elsewhere.
        config_path: The relative or absolute path to the PHSafe config file.

    Returns: The given config dictionary, possibly with some values' types changed.

    Raises:
        RuntimeError: When any of the config values are invalid.
    """
    if STATE_FILTER_FLAG in config:
        _validate_state_filter(config[STATE_FILTER_FLAG], config_path)

    if READER_FLAG in config:
        if config[READER_FLAG] not in ["csv", "cef"]:
            raise RuntimeError(
                f"Config validation failed on {config_path}: "
                f"{READER_FLAG} must be one of: csv, cef"
                f" Value received: {config[READER_FLAG]}"
            )

    if PRIVACY_DEFN_FLAG in config:
        if config[PRIVACY_DEFN_FLAG] not in ["puredp", "zcdp"]:
            raise RuntimeError(
                f"Config validation failed on {config_path}: "
                f"{PRIVACY_DEFN_FLAG} must be one of: puredp, zcdp"
                f" Value received: {config[PRIVACY_DEFN_FLAG]}"
            )

    if PRIVACY_BUDGET_KEY in config:
        if set(config[PRIVACY_BUDGET_KEY]) != set(TABULATIONS_KEY):
            raise RuntimeError(
                "Missing privacy_budget value for these tabulations: "
                f"{sorted(set(TABULATIONS_KEY) - set(config[PRIVACY_BUDGET_KEY]))}"
            )

        for tabulation, geo_iteration_budget_dict in config[PRIVACY_BUDGET_KEY].items():
            # If privacy_budget happens to be integer and is entered with no decimal
            # part, the JSON loader will return it as an int, so fix its type.
            if not isinstance(geo_iteration_budget_dict, dict):
                raise RuntimeError(
                    f"Config validation failed on {config_path}: "
                    f"expected {tabulation} "
                    f"to have type dict, not {type(geo_iteration_budget_dict).__name__}"
                )

            expected_geo_iteration_keys = get_config_privacy_budget_dict(0)[
                tabulation
            ].keys()
            if len(expected_geo_iteration_keys - geo_iteration_budget_dict.keys()) > 0:
                missing_keys = (
                    expected_geo_iteration_keys - geo_iteration_budget_dict.keys()
                )
                raise RuntimeError(
                    f"Config validation failed on {config_path}: "
                    f"{tabulation} is missing required budget values: "
                    f"{sorted(missing_keys)}"
                )
            if len(geo_iteration_budget_dict.keys() - expected_geo_iteration_keys) > 0:
                extra_keys = (
                    geo_iteration_budget_dict.keys() - expected_geo_iteration_keys
                )
                raise RuntimeError(
                    f"Config validation failed on {config_path}: "
                    f"{tabulation} has unexpected budget values: "
                    f"{sorted(extra_keys)}"
                )

            for geo_iteration, budget in geo_iteration_budget_dict.items():
                if isinstance(budget, int):
                    budget = float(budget)
                if not isinstance(budget, float):
                    raise RuntimeError(
                        f"Config validation failed on {config_path}: "
                        f"expected {tabulation}'s '{geo_iteration}' key "
                        f"to have float value, not {type(budget).__name__}"
                    )
                if budget < 0:
                    raise RuntimeError(
                        f"Config validation failed on {config_path}: "
                        f"value of {tabulation}'s '{geo_iteration}' key "
                        f"must be non-negative. Value received: {budget}"
                    )

    if TAU_KEY in config:
        if set(config[TAU_KEY]) != set(TRUNCATIONS_KEY):
            raise RuntimeError(
                "Missing tau value for these tabulations: "
                f"{sorted(set(TRUNCATIONS_KEY) - set(config[TAU_KEY]))}"
            )

        for tabulation, threshold in config[TAU_KEY].items():
            if not isinstance(threshold, int):
                raise RuntimeError(
                    f"Config validation failed on {config_path}: expected tau value for"
                    f" {tabulation} to have type int, not {type(threshold).__name__}"
                )
            if threshold <= 0:
                raise RuntimeError(
                    f"Config validation failed on {config_path}: "
                    f"value of tau for {tabulation} must be greater than zero."
                    f" Value received: {threshold}"
                )
    return config


def _validate_state_filter(state_filter: Any, config_path: str):
    """Validate a list of state codes for the state_filter configuration option.

    Validates that the state_filter value has the following properties:
     * Is a list of strings
     * Contains no duplicate values
     * All elements are valid FIPS codes for the States and the
     District of Columbia OR Puerto Rico
     * FIPS Codes for Outlying Areas of the United States and the
     Freely Associated States are considered invalid

    See
    https://www.census.gov/library/reference/code-lists/ansi/ansi-codes-for-states.html.

    Args:
        state_filter: The state_filter value to be validated, as read in
            from the config file.
        config_path: The relative or absolute path to the PHSafe config file.

    Raises:
        RuntimeError: When the given state_filter is not valid.
    """
    VALID_STATE_CODE_NUMS = set(range(1, 57))
    VALID_STATE_CODE_NUMS -= {3, 7, 14, 43, 52}
    VALID_STATE_CODE_NUMS |= {72}
    VALID_STATE_CODES = {f"{i:02d}" for i in VALID_STATE_CODE_NUMS}

    if not state_filter:
        raise RuntimeError(
            f"Config validation failed on {config_path}: expected "
            f"{STATE_FILTER_FLAG} to not be empty"
        )

    if not isinstance(state_filter, list):
        raise RuntimeError(
            f"Config validation failed on {config_path}: expected "
            f"{STATE_FILTER_FLAG} to have type list, not {type(state_filter).__name__}"
        )

    bad_types = {type(e) for e in state_filter if not isinstance(e, str)}
    if bad_types:
        bad_types_str = "{" + ", ".join(t.__name__ for t in bad_types) + "}"
        raise RuntimeError(
            f"Config validation failed on {config_path}: expected "
            f"{STATE_FILTER_FLAG} elements to have type str, not {bad_types_str}"
        )

    if len(state_filter) != len(set(state_filter)):
        duplicates = [e for e in state_filter if state_filter.count(e) > 1]
        duplicates_str = "{" + ", ".join(duplicates) + "}"
        raise RuntimeError(
            f"Config validation failed on {config_path}: "
            f"{STATE_FILTER_FLAG} list contains duplicate values:"
            f" {duplicates_str}"
        )

    if "72" in state_filter and len(state_filter) > 1:
        raise RuntimeError(
            f"Config validation failed on {config_path}: expected"
            f" {STATE_FILTER_FLAG} to have either US FIPS codes or 72. Running PR with"
            " the rest of the US is not supported."
        )

    bad_state_codes = sorted(set(state_filter) - VALID_STATE_CODES)
    if bad_state_codes:
        bad_state_codes_str = "{" + ", ".join(bad_state_codes) + "}"
        raise RuntimeError(
            f"Config validation failed on {config_path}: "
            f"{STATE_FILTER_FLAG} contains invalid codes {bad_state_codes_str}"
        )


def check_s3_path(data_path: str, output_path: str) -> None:
    """Check if the file_path is a S3 path.

    If true and spark is in Local mode, raise RuntimeError.

    Args:
        data_path: path to the input directory
        output_path: path to the output directory
    """
    if SparkSession.builder.getOrCreate().conf.get("spark.master").startswith("local"):
        if any((is_s3_path(data_path), is_s3_path(output_path))):
            raise RuntimeError(
                "Reading and writing to and from s3 is not supported when running "
                "Spark in local mode."
            )


@dataclass(eq=True)
class NonprivatePHSafeParameters:
    """Class for maintaining nonprivate phsafe parameters."""

    state_filter: List[str]
    reader: str
    data_path: str
    output_path: str


@dataclass(eq=True)
class PrivatePHSafeParameters:
    """Class for maintaining private phsafe parameters."""

    state_filter: List[str]
    reader: str
    data_path: str
    output_path: str
    privacy_defn: Union[Literal["puredp"], Literal["zcdp"]]
    privacy_budget: Mapping[str, Mapping[str, float]]
    tau: Mapping[str, int]


def process_nonprivate_input_parameters(
    config_path: str, data_path: str, output_path: str
) -> NonprivatePHSafeParameters:
    """Returns dataclass with parameters.

    Args:
        config_path: Path to json config for PHSafe application.
        data_path: Path used to configure the reader.
        output_path: Path to output directory.
    """
    logger = logging.getLogger(__name__)
    check_s3_path(data_path, output_path)
    CONFIG_KEYS = {STATE_FILTER_FLAG, READER_FLAG}
    config = _parse_config_json(config_path, CONFIG_KEYS)

    params = NonprivatePHSafeParameters(
        config[STATE_FILTER_FLAG], config[READER_FLAG], data_path, output_path
    )
    logger.info(f"Non-private PHSafe parameters: {params}")
    return params


def process_private_input_parameters(
    config_path: str, data_path: str, output_path: str
) -> PrivatePHSafeParameters:
    """Returns dataclass with parameters.

    Args:
        config_path: Path to json config for PHSafe application.
        data_path: Path used to configure the reader.
        output_path: Path to output directory.
    """
    logger = logging.getLogger(__name__)
    check_s3_path(data_path, output_path)
    CONFIG_KEYS = {
        STATE_FILTER_FLAG,
        READER_FLAG,
        PRIVACY_DEFN_FLAG,
        PRIVACY_BUDGET_KEY,
        TAU_KEY,
    }
    config = _parse_config_json(config_path, CONFIG_KEYS)

    params = PrivatePHSafeParameters(
        config[STATE_FILTER_FLAG],
        config[READER_FLAG],
        data_path,
        output_path,
        config[PRIVACY_DEFN_FLAG],
        config[PRIVACY_BUDGET_KEY],
        config[TAU_KEY],
    )
    logger.info(f"Private PHSafe parameters: {params}")
    return params
