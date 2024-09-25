"""PHSafe module."""

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
from typing import NamedTuple, Optional

from pyspark.sql.dataframe import DataFrame

TABULATIONS_KEY = {
    "PH1_denom",
    "PH1_num",
    "PH2",
    "PH3",
    "PH4",
    "PH5_denom",
    "PH6",
    "PH7",
    "PH8_denom",
}
"""Nested keys in config.json that indicated the tabulations to be performed."""

TABULATION_OUTPUT_COLUMNS = {
    "PH2": {
        "REGION_ID": str,
        "REGION_TYPE": str,
        "PH2_DATA_CELL": str,
        "COUNT": int,
        "NOISE_DISTRIBUTION": str,
        "VARIANCE": int,
    },
    "PH3": {
        "REGION_ID": str,
        "REGION_TYPE": str,
        "ITERATION_CODE": str,
        "PH3_DATA_CELL": int,
        "COUNT": int,
        "NOISE_DISTRIBUTION": str,
        "VARIANCE": int,
    },
    "PH1_num": {
        "REGION_ID": str,
        "REGION_TYPE": str,
        "ITERATION_CODE": str,
        "PH1_NUM_DATA_CELL": int,
        "COUNT": int,
        "NOISE_DISTRIBUTION": str,
        "VARIANCE": int,
    },
    "PH1_denom": {
        "REGION_ID": str,
        "REGION_TYPE": str,
        "ITERATION_CODE": str,
        "PH1_DENOM_DATA_CELL": int,
        "COUNT": int,
        "NOISE_DISTRIBUTION": str,
        "VARIANCE": int,
    },
    "PH4": {
        "REGION_ID": str,
        "REGION_TYPE": str,
        "ITERATION_CODE": str,
        "PH4_DATA_CELL": int,
        "COUNT": int,
        "NOISE_DISTRIBUTION": str,
        "VARIANCE": int,
    },
    "PH5_num": {
        "REGION_ID": str,
        "REGION_TYPE": str,
        "ITERATION_CODE": str,
        "PH5_NUM_DATA_CELL": int,
        "COUNT": int,
        "NOISE_DISTRIBUTION": str,
        "VARIANCE": int,
    },
    "PH5_denom": {
        "REGION_ID": str,
        "REGION_TYPE": str,
        "ITERATION_CODE": str,
        "PH5_DENOM_DATA_CELL": int,
        "COUNT": int,
        "NOISE_DISTRIBUTION": str,
        "VARIANCE": int,
    },
    "PH6": {
        "REGION_ID": str,
        "REGION_TYPE": str,
        "PH6_DATA_CELL": int,
        "COUNT": int,
        "NOISE_DISTRIBUTION": str,
        "VARIANCE": int,
    },
    "PH7": {
        "REGION_ID": str,
        "REGION_TYPE": str,
        "ITERATION_CODE": str,
        "PH7_DATA_CELL": int,
        "COUNT": int,
        "NOISE_DISTRIBUTION": str,
        "VARIANCE": int,
    },
    "PH8_num": {
        "REGION_ID": str,
        "REGION_TYPE": str,
        "ITERATION_CODE": str,
        "PH8_NUM_DATA_CELL": int,
        "COUNT": int,
        "NOISE_DISTRIBUTION": str,
        "VARIANCE": int,
    },
    "PH8_denom": {
        "REGION_ID": str,
        "REGION_TYPE": str,
        "ITERATION_CODE": str,
        "PH8_DENOM_DATA_CELL": int,
        "COUNT": int,
        "NOISE_DISTRIBUTION": str,
        "VARIANCE": int,
    },
}
"""The mapping of tabulation and expected output columns with dtype, in order,
    present in each private output file."""


class PHSafeInput(NamedTuple):
    """A container to pass around PHSafeInput for input validation."""

    persons: DataFrame
    units: DataFrame
    geo: DataFrame


class PHSafeOutput(NamedTuple):
    """A container to ouput all PHSafe output tables."""

    PH1_num: Optional[DataFrame] = None
    PH1_denom: Optional[DataFrame] = None
    PH2: Optional[DataFrame] = None
    PH3: Optional[DataFrame] = None
    PH4: Optional[DataFrame] = None
    PH5_num: Optional[DataFrame] = None
    PH5_denom: Optional[DataFrame] = None
    PH6: Optional[DataFrame] = None
    PH7: Optional[DataFrame] = None
    PH8_num: Optional[DataFrame] = None
    PH8_denom: Optional[DataFrame] = None
