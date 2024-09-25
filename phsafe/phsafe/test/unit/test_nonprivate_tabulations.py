"""Tests nonprivate tabulations on person household data."""

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

import io
import pkgutil
from test.conftest import parametrize

import pandas as pd
import pytest
from pyspark.sql.types import LongType, StringType, StructField, StructType

from tmlt.common.pyspark_test_tools import pyspark  # pylint: disable=unused-import
from tmlt.common.pyspark_test_tools import assert_frame_equal_with_sort
from tmlt.phsafe import TABULATION_OUTPUT_COLUMNS, PHSafeInput, PHSafeOutput
from tmlt.phsafe.nonprivate_tabulations import NonPrivateTabulations
from tmlt.phsafe.paths import RESOURCES_PACKAGE_NAME

TABULATION_OUTPUT_COLUMNS = {
    "PH2": {"REGION_ID": str, "REGION_TYPE": str, "PH2_DATA_CELL": int, "COUNT": int},
    "PH3": {
        "REGION_ID": str,
        "REGION_TYPE": str,
        "ITERATION_CODE": str,
        "PH3_DATA_CELL": int,
        "COUNT": int,
    },
    "PH1_num": {
        "REGION_ID": str,
        "REGION_TYPE": str,
        "ITERATION_CODE": str,
        "PH1_NUM_DATA_CELL": int,
        "COUNT": int,
    },
    "PH1_denom": {
        "REGION_ID": str,
        "REGION_TYPE": str,
        "ITERATION_CODE": str,
        "PH1_DENOM_DATA_CELL": int,
        "COUNT": int,
    },
    "PH4": {
        "REGION_ID": str,
        "REGION_TYPE": str,
        "ITERATION_CODE": str,
        "PH4_DATA_CELL": int,
        "COUNT": int,
    },
    "PH5_num": {
        "REGION_ID": str,
        "REGION_TYPE": str,
        "ITERATION_CODE": str,
        "PH5_NUM_DATA_CELL": int,
        "COUNT": int,
    },
    "PH5_denom": {
        "REGION_ID": str,
        "REGION_TYPE": str,
        "ITERATION_CODE": str,
        "PH5_DENOM_DATA_CELL": int,
        "COUNT": int,
    },
    "PH6": {"REGION_ID": str, "REGION_TYPE": str, "PH6_DATA_CELL": int, "COUNT": int},
    "PH7": {
        "REGION_ID": str,
        "REGION_TYPE": str,
        "ITERATION_CODE": str,
        "PH7_DATA_CELL": int,
        "COUNT": int,
    },
    "PH8_num": {
        "REGION_ID": str,
        "REGION_TYPE": str,
        "ITERATION_CODE": str,
        "PH8_NUM_DATA_CELL": int,
        "COUNT": int,
    },
    "PH8_denom": {
        "REGION_ID": str,
        "REGION_TYPE": str,
        "ITERATION_CODE": str,
        "PH8_DENOM_DATA_CELL": int,
        "COUNT": int,
    },
}
"""The mapping of tabulation and expected output columns with dtype, in order,
    present in each non-private output file."""


@pytest.fixture(scope="class")
def setup_nonprivate_tabulations(
    spark,
    person_data,
    unit_data,
    geo_data,
    expected_PH2,
    expected_PH3,
    expected_PH1_num,
    expected_PH1_denom,
    expected_PH4,
    expected_PH5_num,
    expected_PH5_denom,
    expected_PH6,
    expected_PH7,
    expected_PH8_num,
    expected_PH8_denom,
    request,
):
    """Set up test."""

    # Set up test variables.
    person = spark.createDataFrame(
        spark.sparkContext.parallelize(person_data),
        StructType(
            [
                StructField("RTYPE", StringType(), True),
                StructField("MAFID", LongType(), True),
                StructField("QAGE", LongType(), True),
                StructField("CENHISP", LongType(), True),
                StructField("CENRACE", StringType(), True),
                StructField("RELSHIP", StringType(), True),
            ]
        ),
    )
    unit = spark.createDataFrame(
        spark.sparkContext.parallelize(unit_data),
        StructType(
            [
                StructField("RTYPE", StringType(), True),
                StructField("MAFID", LongType(), True),
                StructField("FINAL_POP", LongType(), True),
                StructField("NPF", LongType(), True),
                StructField("HHSPAN", LongType(), True),
                StructField("HHRACE", StringType(), True),
                StructField("TEN", StringType(), True),
                StructField("HHT", StringType(), True),
                StructField("HHT2", StringType(), True),
                StructField("CPLT", StringType(), True),
            ]
        ),
    )
    processed_geo = spark.createDataFrame(
        spark.sparkContext.parallelize(geo_data),
        StructType(
            [
                StructField("MAFID", LongType(), True),
                StructField("USA", StringType(), True),
                StructField("STATE", StringType(), True),
            ]
        ),
    )

    # Evaluate tables.
    tabulator = NonPrivateTabulations()
    request.cls.tables = tabulator(
        PHSafeInput(persons=person, units=unit, geo=processed_geo)
    )
    request.cls.expected_PH2 = expected_PH2
    request.cls.expected_PH3 = expected_PH3
    request.cls.expected_PH1_num = expected_PH1_num
    request.cls.expected_PH1_denom = expected_PH1_denom
    request.cls.expected_PH4 = expected_PH4
    request.cls.expected_PH5_num = expected_PH5_num
    request.cls.expected_PH5_denom = expected_PH5_denom
    request.cls.expected_PH6 = expected_PH6
    request.cls.expected_PH7 = expected_PH7
    request.cls.expected_PH8_num = expected_PH8_num
    request.cls.expected_PH8_denom = expected_PH8_denom


@pytest.mark.usefixtures("spark")
@pytest.mark.usefixtures("setup_nonprivate_tabulations")
@parametrize(
    "person_data, unit_data, geo_data, expected_PH2, expected_PH3, expected_PH1_num,"
    " expected_PH1_denom, expected_PH4, expected_PH5_num, expected_PH5_denom,"
    " expected_PH6, expected_PH7, expected_PH8_num, expected_PH8_denom",
    [
        {  # Test PH2 table for HHT2 01-02
            "person_data": [
                ("3", 100000001, 25, 1, "01", "20"),
                ("3", 100000002, 25, 1, "01", "20"),
                ("3", 100000002, 25, 1, "01", "21"),
                ("3", 100000003, 25, 1, "01", "20"),
            ],
            "unit_data": [
                ("2", 100000001, 2, 0, 1, "01", "4", "0", "01", "1"),
                ("2", 100000002, 2, 0, 1, "01", "4", "0", "02", "2"),
                ("2", 100000003, 1, 0, 1, "01", "4", "0", "02", "5"),
            ],
            "geo_data": [
                (100000001, "00", "01"),
                (100000002, "00", "01"),
                (100000003, "00", "01"),
            ],
            "expected_PH2": pd.read_csv(
                io.BytesIO(
                    pkgutil.get_data(  # type: ignore
                        "tmlt.phsafe", RESOURCES_PACKAGE_NAME + "/test/test1/PH2.csv"
                    )
                ),
                encoding="utf8",
                dtype={
                    "REGION_ID": object,
                    "REGION_TYPE": object,
                    "PH2_DATA_CELL": int,
                },
            ),
            "expected_PH3": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH3"].keys()
            ),
            "expected_PH1_num": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH1_num"].keys()
            ),
            "expected_PH1_denom": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH1_denom"].keys()
            ),
            "expected_PH4": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH4"].keys()
            ),
            "expected_PH5_num": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH5_num"].keys()
            ),
            "expected_PH5_denom": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH5_denom"].keys()
            ),
            "expected_PH6": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH6"].keys()
            ),
            "expected_PH7": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH7"].keys()
            ),
            "expected_PH8_num": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH8_num"].keys()
            ),
            "expected_PH8_denom": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH8_denom"].keys()
            ),
        },
        {  # Test PH2 table for HHT2 03-04, 09
            "person_data": [
                ("3", 100000004, 25, 1, "01", "20"),
                ("3", 100000004, 25, 1, "01", "22"),
                ("3", 100000005, 25, 1, "01", "20"),
                ("3", 100000005, 25, 1, "01", "24"),
                ("3", 100000006, 25, 1, "01", "20"),
                ("3", 100000006, 25, 1, "01", "24"),
                ("3", 100000010, 25, 1, "01", "20"),
            ],
            "unit_data": [
                ("2", 100000004, 2, 0, 1, "01", "4", "0", "03", "3"),
                ("2", 100000005, 2, 0, 1, "01", "4", "0", "03", "4"),
                ("2", 100000006, 2, 0, 1, "01", "4", "0", "04", "5"),
                ("2", 100000010, 1, 0, 1, "03", "4", "0", "09", "5"),
            ],
            "geo_data": [
                (100000004, "00", "01"),
                (100000005, "00", "01"),
                (100000006, "00", "01"),
                (100000010, "00", "01"),
            ],
            "expected_PH2": pd.read_csv(
                io.BytesIO(
                    pkgutil.get_data(  # type: ignore
                        "tmlt.phsafe", RESOURCES_PACKAGE_NAME + "/test/test2/PH2.csv"
                    )
                ),
                encoding="utf8",
                dtype={
                    "REGION_ID": object,
                    "REGION_TYPE": object,
                    "PH2_DATA_CELL": int,
                },
            ),
            "expected_PH3": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH3"].keys()
            ),
            "expected_PH1_num": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH1_num"].keys()
            ),
            "expected_PH1_denom": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH1_denom"].keys()
            ),
            "expected_PH4": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH4"].keys()
            ),
            "expected_PH5_num": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH5_num"].keys()
            ),
            "expected_PH5_denom": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH5_denom"].keys()
            ),
            "expected_PH6": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH6"].keys()
            ),
            "expected_PH7": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH7"].keys()
            ),
            "expected_PH8_num": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH8_num"].keys()
            ),
            "expected_PH8_denom": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH8_denom"].keys()
            ),
        },
        {  # Test PH2 table for HHT2 5-12
            "person_data": [
                ("3", 100000007, 25, 1, "01", "20"),
                ("3", 100000008, 25, 1, "01", "20"),
                ("3", 100000009, 25, 1, "01", "20"),
            ],
            "unit_data": [
                ("2", 100000007, 1, 0, 1, "01", "4", "0", "10", "5"),
                ("2", 100000008, 1, 0, 1, "01", "4", "0", "05", "5"),
                ("2", 100000009, 1, 0, 1, "01", "4", "0", "06", "5"),
            ],
            "geo_data": [
                (100000007, "00", "01"),
                (100000008, "00", "01"),
                (100000009, "00", "01"),
            ],
            "expected_PH2": pd.read_csv(
                io.BytesIO(
                    pkgutil.get_data(  # type: ignore
                        "tmlt.phsafe", RESOURCES_PACKAGE_NAME + "/test/test3/PH2.csv"
                    )
                ),
                encoding="utf8",
                dtype={
                    "REGION_ID": object,
                    "REGION_TYPE": object,
                    "PH2_DATA_CELL": int,
                },
            ),
            "expected_PH3": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH3"].keys()
            ),
            "expected_PH1_num": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH1_num"].keys()
            ),
            "expected_PH1_denom": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH1_denom"].keys()
            ),
            "expected_PH4": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH4"].keys()
            ),
            "expected_PH5_num": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH5_num"].keys()
            ),
            "expected_PH5_denom": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH5_denom"].keys()
            ),
            "expected_PH6": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH6"].keys()
            ),
            "expected_PH7": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH7"].keys()
            ),
            "expected_PH8_num": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH8_num"].keys()
            ),
            "expected_PH8_denom": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH8_denom"].keys()
            ),
        },
        {  # Test PH3 table for RELSHIP 25-27 and RTYPE_PERSON=5 (also different races)
            "person_data": [
                ("3", 100000001, 25, 1, "34", "20"),
                ("3", 100000001, 5, 1, "34", "25"),
                ("5", 100000002, 26, 1, "01", "37"),
                ("5", 100000002, 17, 1, "01", "37"),
                ("5", 100000002, 17, 1, "02", "38"),
                ("5", 100000002, 17, 2, "03", "20"),
                ("3", 100000003, 25, 1, "34", "20"),
                ("3", 100000003, 5, 1, "34", "25"),
                ("3", 100000004, 25, 1, "04", "20"),
                ("3", 100000004, 25, 1, "04", "20"),
                ("3", 100000004, 5, 1, "04", "25"),
                ("3", 100000005, 25, 1, "05", "20"),
                ("3", 100000005, 25, 1, "05", "20"),
                ("3", 100000005, 5, 1, "05", "25"),
                ("3", 100000006, 25, 1, "06", "20"),
                ("3", 100000006, 5, 1, "06", "25"),
            ],
            "unit_data": [
                ("2", 100000001, 2, 0, 1, "34", "4", "0", "00", "0"),
                ("4", 100000002, 400, 0, 0, "00", "3", "0", "00", "5"),
                ("2", 100000003, 2, 0, 1, "34", "4", "0", "06", "1"),
                ("2", 100000004, 3, 0, 1, "04", "2", "0", "01", "1"),
                ("2", 100000005, 3, 0, 1, "05", "2", "0", "03", "3"),
                ("2", 100000006, 2, 0, 1, "06", "2", "0", "10", "1"),
            ],
            "geo_data": [
                (100000001, "00", "01"),
                (100000002, "00", "01"),
                (100000003, "00", "01"),
                (100000004, "00", "01"),
                (100000005, "00", "01"),
                (100000006, "00", "01"),
            ],
            "expected_PH2": pd.read_csv(
                io.BytesIO(
                    pkgutil.get_data(  # type: ignore
                        "tmlt.phsafe", RESOURCES_PACKAGE_NAME + "/test/test4/PH2.csv"
                    )
                ),
                encoding="utf8",
                dtype={
                    "REGION_ID": object,
                    "REGION_TYPE": object,
                    "PH2_DATA_CELL": int,
                },
            ),
            "expected_PH3": pd.read_csv(
                io.BytesIO(
                    pkgutil.get_data(  # type: ignore
                        "tmlt.phsafe", RESOURCES_PACKAGE_NAME + "/test/test4/PH3.csv"
                    )
                ),
                encoding="utf8",
                dtype={
                    "REGION_ID": object,
                    "REGION_TYPE": object,
                    "ITERATION_CODE": object,
                    "PH3_DATA_CELL": int,
                },
            ),
            "expected_PH1_num": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH1_num"].keys()
            ),
            "expected_PH1_denom": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH1_denom"].keys()
            ),
            "expected_PH4": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH4"].keys()
            ),
            "expected_PH5_num": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH5_num"].keys()
            ),
            "expected_PH5_denom": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH5_denom"].keys()
            ),
            "expected_PH6": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH6"].keys()
            ),
            "expected_PH7": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH7"].keys()
            ),
            "expected_PH8_num": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH8_num"].keys()
            ),
            "expected_PH8_denom": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH8_denom"].keys()
            ),
        },
        {  # Test PH3 table for RELSHIP 20-24, 28-36
            "person_data": [
                ("3", 100000001, 17, 1, "29", "20"),
                ("3", 100000002, 65, 1, "29", "20"),
                ("3", 100000002, 5, 1, "29", "30"),
                ("3", 100000003, 23, 1, "29", "20"),
                ("3", 100000003, 16, 1, "29", "33"),
                ("3", 100000004, 19, 1, "29", "20"),
                ("3", 100000004, 17, 1, "05", "34"),
            ],
            "unit_data": [
                ("2", 100000001, 1, 0, 1, "29", "3", "0", "00", "0"),
                ("2", 100000002, 2, 0, 0, "29", "3", "0", "00", "0"),
                ("2", 100000003, 2, 0, 1, "29", "3", "0", "00", "0"),
                ("2", 100000004, 2, 0, 1, "29", "3", "0", "00", "0"),
            ],
            "geo_data": [
                (100000001, "00", "01"),
                (100000002, "00", "01"),
                (100000003, "00", "01"),
                (100000004, "00", "01"),
            ],
            "expected_PH2": pd.read_csv(
                io.BytesIO(
                    pkgutil.get_data(  # type: ignore
                        "tmlt.phsafe", RESOURCES_PACKAGE_NAME + "/test/test5/PH2.csv"
                    )
                ),
                encoding="utf8",
                dtype={
                    "REGION_ID": object,
                    "REGION_TYPE": object,
                    "PH2_DATA_CELL": int,
                },
            ),
            "expected_PH3": pd.read_csv(
                io.BytesIO(
                    pkgutil.get_data(  # type: ignore
                        "tmlt.phsafe", RESOURCES_PACKAGE_NAME + "/test/test5/PH3.csv"
                    )
                ),
                encoding="utf8",
                dtype={
                    "REGION_ID": object,
                    "REGION_TYPE": object,
                    "ITERATION_CODE": object,
                    "PH3_DATA_CELL": int,
                },
            ),
            "expected_PH1_num": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH1_num"].keys()
            ),
            "expected_PH1_denom": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH1_denom"].keys()
            ),
            "expected_PH4": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH4"].keys()
            ),
            "expected_PH5_num": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH5_num"].keys()
            ),
            "expected_PH5_denom": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH5_denom"].keys()
            ),
            "expected_PH6": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH6"].keys()
            ),
            "expected_PH7": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH7"].keys()
            ),
            "expected_PH8_num": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH8_num"].keys()
            ),
            "expected_PH8_denom": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH8_denom"].keys()
            ),
        },
        {  # Test PH1 table
            "person_data": [
                ("3", 100000001, 25, 1, "01", "20"),
                ("3", 100000002, 25, 1, "02", "20"),
                ("3", 100000002, 25, 1, "02", "21"),
                ("3", 100000003, 25, 1, "03", "20"),
                ("3", 100000003, 25, 1, "03", "21"),
                ("3", 100000004, 25, 1, "04", "20"),
                ("3", 100000005, 25, 1, "05", "20"),
                ("3", 100000006, 25, 1, "06", "20"),
                ("3", 100000007, 25, 1, "07", "20"),
                ("3", 100000008, 25, 2, "23", "20"),
                ("3", 100000008, 3, 2, "23", "25"),
            ],
            "unit_data": [
                ("2", 100000001, 1, 0, 1, "01", "2", "2", "00", "5"),
                ("2", 100000002, 2, 0, 1, "02", "2", "1", "00", "1"),
                ("2", 100000003, 2, 0, 1, "03", "2", "1", "00", "1"),
                ("2", 100000004, 1, 0, 1, "04", "2", "6", "00", "5"),
                ("2", 100000005, 1, 0, 1, "05", "2", "1", "00", "5"),
                ("2", 100000006, 1, 0, 1, "06", "2", "1", "00", "5"),
                ("2", 100000007, 1, 0, 1, "07", "2", "1", "00", "5"),
                ("2", 100000008, 2, 0, 2, "23", "2", "3", "00", "5"),
            ],
            "geo_data": [
                (100000001, "00", "01"),
                (100000002, "00", "01"),
                (100000003, "00", "01"),
                (100000004, "00", "01"),
                (100000005, "00", "01"),
                (100000006, "00", "01"),
                (100000007, "00", "01"),
                (100000008, "00", "05"),
            ],
            "expected_PH2": pd.read_csv(
                io.BytesIO(
                    pkgutil.get_data(  # type: ignore
                        "tmlt.phsafe", RESOURCES_PACKAGE_NAME + "/test/test6/PH2.csv"
                    )
                ),
                encoding="utf8",
                dtype={
                    "REGION_ID": object,
                    "REGION_TYPE": object,
                    "PH2_DATA_CELL": int,
                },
            ),
            "expected_PH3": pd.read_csv(
                io.BytesIO(
                    pkgutil.get_data(  # type: ignore
                        "tmlt.phsafe", RESOURCES_PACKAGE_NAME + "/test/test6/PH3.csv"
                    )
                ),
                encoding="utf8",
                dtype=TABULATION_OUTPUT_COLUMNS["PH3"],
            ),
            "expected_PH1_num": pd.read_csv(
                io.BytesIO(
                    pkgutil.get_data(  # type: ignore
                        "tmlt.phsafe",
                        RESOURCES_PACKAGE_NAME + "/test/test6/PH1_num.csv",
                    )
                ),
                encoding="utf8",
                dtype=TABULATION_OUTPUT_COLUMNS["PH1_num"],
            ),
            "expected_PH1_denom": pd.read_csv(
                io.BytesIO(
                    pkgutil.get_data(  # type: ignore
                        "tmlt.phsafe", "resources/test/test6/PH1_denom.csv"
                    )
                ),
                encoding="utf8",
                dtype=TABULATION_OUTPUT_COLUMNS["PH1_denom"],
            ),
            "expected_PH4": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH4"].keys()
            ),
            "expected_PH5_num": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH5_num"].keys()
            ),
            "expected_PH5_denom": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH5_denom"].keys()
            ),
            "expected_PH6": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH6"].keys()
            ),
            "expected_PH7": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH7"].keys()
            ),
            "expected_PH8_num": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH8_num"].keys()
            ),
            "expected_PH8_denom": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH8_denom"].keys()
            ),
        },
        {  # test for ph4
            "person_data": [
                ("3", 100000004, 25, 1, "01", "20"),
                ("3", 100000004, 25, 1, "01", "21"),
                ("3", 100000004, 6, 1, "01", "25"),
                ("3", 100000005, 25, 1, "01", "20"),
                ("3", 100000005, 25, 1, "01", "21"),
                ("3", 100000006, 25, 1, "01", "20"),
                ("3", 100000006, 25, 1, "01", "22"),
                ("3", 100000006, 6, 1, "01", "25"),
                ("3", 100000010, 25, 1, "03", "20"),
            ],
            "unit_data": [
                ("2", 100000004, 3, 3, 1, "01", "4", "1", "01", "1"),
                ("2", 100000005, 2, 2, 1, "01", "4", "1", "02", "1"),
                ("2", 100000006, 3, 2, 1, "01", "4", "3", "03", "3"),
                ("2", 100000010, 1, 0, 1, "03", "4", "4", "09", "5"),
            ],
            "geo_data": [
                (100000004, "00", "01"),
                (100000005, "00", "01"),
                (100000006, "00", "01"),
                (100000010, "00", "01"),
            ],
            "expected_PH2": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH2"].keys()
            ),
            "expected_PH3": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH3"].keys()
            ),
            "expected_PH1_num": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH1_num"].keys()
            ),
            "expected_PH1_denom": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH1_denom"].keys()
            ),
            "expected_PH4": pd.read_csv(
                io.BytesIO(
                    pkgutil.get_data(  # type: ignore
                        "tmlt.phsafe", RESOURCES_PACKAGE_NAME + "/test/test7/PH4.csv"
                    )
                ),
                encoding="utf8",
                dtype={
                    "REGION_ID": object,
                    "REGION_TYPE": object,
                    "ITERATION_CODE": object,
                    "PH4_DATA_CELL": int,
                },
            ),
            "expected_PH5_num": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH5_num"].keys()
            ),
            "expected_PH5_denom": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH5_denom"].keys()
            ),
            "expected_PH6": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH6"].keys()
            ),
            "expected_PH7": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH7"].keys()
            ),
            "expected_PH8_num": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH8_num"].keys()
            ),
            "expected_PH8_denom": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH8_denom"].keys()
            ),
        },
        {  # test for ph5
            "person_data": [
                ("3", 100000004, 25, 1, "01", "20"),
                ("3", 100000004, 25, 1, "01", "21"),
                ("3", 100000004, 6, 1, "01", "25"),
                ("3", 100000005, 25, 1, "01", "20"),
                ("3", 100000005, 25, 1, "01", "21"),
                ("3", 100000006, 25, 1, "01", "20"),
                ("3", 100000006, 25, 1, "01", "22"),
                ("3", 100000006, 6, 1, "01", "25"),
                ("3", 100000010, 25, 1, "03", "20"),
            ],
            "unit_data": [
                ("2", 100000004, 3, 3, 1, "01", "4", "1", "01", "1"),
                ("2", 100000005, 2, 2, 1, "01", "4", "1", "02", "1"),
                ("2", 100000006, 3, 2, 1, "01", "4", "3", "03", "3"),
                ("2", 100000010, 1, 0, 1, "03", "4", "4", "09", "5"),
            ],
            "geo_data": [
                (100000004, "00", "01"),
                (100000005, "00", "01"),
                (100000006, "00", "01"),
                (100000010, "00", "01"),
            ],
            "expected_PH2": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH2"].keys()
            ),
            "expected_PH3": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH3"].keys()
            ),
            "expected_PH1_num": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH1_num"].keys()
            ),
            "expected_PH1_denom": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH1_denom"].keys()
            ),
            "expected_PH4": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH4"].keys()
            ),
            "expected_PH5_num": pd.read_csv(
                io.BytesIO(
                    pkgutil.get_data(  # type: ignore
                        "tmlt.phsafe",
                        RESOURCES_PACKAGE_NAME + "/test/test8/PH5_num.csv",
                    )
                ),
                encoding="utf8",
                dtype={
                    "REGION_ID": object,
                    "REGION_TYPE": object,
                    "ITERATION_CODE": object,
                    "PH5_NUM_DATA_CELL": int,
                },
            ),
            "expected_PH5_denom": pd.read_csv(
                io.BytesIO(
                    pkgutil.get_data(  # type: ignore
                        "tmlt.phsafe",
                        RESOURCES_PACKAGE_NAME + "/test/test8/PH5_denom.csv",
                    )
                ),
                encoding="utf8",
                dtype={
                    "REGION_ID": object,
                    "REGION_TYPE": object,
                    "ITERATION_CODE": object,
                    "PH5_DENOM_DATA_CELL": int,
                },
            ),
            "expected_PH6": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH6"].keys()
            ),
            "expected_PH7": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH7"].keys()
            ),
            "expected_PH8_num": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH8_num"].keys()
            ),
            "expected_PH8_denom": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH8_denom"].keys()
            ),
        },
        {  # test for ph6
            "person_data": [
                ("3", 100000004, 25, 1, "01", "20"),
                ("3", 100000004, 25, 1, "01", "21"),
                ("3", 100000004, 3, 1, "01", "25"),
                ("3", 100000005, 25, 1, "01", "20"),
                ("3", 100000005, 25, 1, "01", "22"),
                ("3", 100000005, 4, 1, "01", "26"),
                ("3", 100000006, 25, 1, "01", "20"),
                ("3", 100000006, 7, 1, "01", "27"),
                ("3", 100000010, 25, 1, "01", "20"),
                ("3", 100000010, 14, 1, "01", "25"),
            ],
            "unit_data": [
                ("2", 100000004, 3, 3, 1, "01", "4", "1", "01", "1"),
                ("2", 100000005, 3, 2, 1, "01", "4", "3", "03", "3"),
                ("2", 100000006, 2, 2, 1, "01", "4", "2", "10", "5"),
                ("2", 100000010, 2, 2, 1, "01", "4", "3", "06", "5"),
            ],
            "geo_data": [
                (100000004, "00", "01"),
                (100000005, "00", "01"),
                (100000006, "00", "01"),
                (100000010, "00", "01"),
            ],
            "expected_PH2": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH2"].keys()
            ),
            "expected_PH3": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH3"].keys()
            ),
            "expected_PH1_num": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH1_num"].keys()
            ),
            "expected_PH1_denom": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH1_denom"].keys()
            ),
            "expected_PH4": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH4"].keys()
            ),
            "expected_PH5_num": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH5_num"].keys()
            ),
            "expected_PH5_denom": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH5_denom"].keys()
            ),
            "expected_PH6": pd.read_csv(
                io.BytesIO(
                    pkgutil.get_data(  # type: ignore
                        "tmlt.phsafe", RESOURCES_PACKAGE_NAME + "/test/test9/PH6.csv"
                    )
                ),
                encoding="utf8",
                dtype={
                    "REGION_ID": object,
                    "REGION_TYPE": object,
                    "ITERATION_CODE": object,
                    "PH6_DATA_CELL": int,
                },
            ),
            "expected_PH7": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH7"].keys()
            ),
            "expected_PH8_num": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH8_num"].keys()
            ),
            "expected_PH8_denom": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH8_denom"].keys()
            ),
        },
        {  # test for ph7
            "person_data": [
                ("3", 100000001, 25, 1, "02", "20"),
                ("3", 100000002, 25, 1, "02", "20"),
                ("3", 100000003, 25, 1, "02", "20"),
                ("3", 100000004, 25, 1, "02", "20"),
            ],
            "unit_data": [
                ("2", 100000001, 1, 0, 1, "02", "1", "4", "09", "5"),
                ("2", 100000002, 1, 0, 1, "02", "2", "4", "09", "5"),
                ("2", 100000003, 1, 0, 1, "02", "3", "4", "09", "5"),
                ("2", 100000004, 1, 0, 1, "02", "4", "4", "09", "5"),
            ],
            "geo_data": [
                (100000001, "00", "01"),
                (100000002, "00", "01"),
                (100000003, "00", "01"),
                (100000004, "00", "01"),
            ],
            "expected_PH2": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH2"].keys()
            ),
            "expected_PH3": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH3"].keys()
            ),
            "expected_PH1_num": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH1_num"].keys()
            ),
            "expected_PH1_denom": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH1_denom"].keys()
            ),
            "expected_PH4": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH4"].keys()
            ),
            "expected_PH5_num": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH5_num"].keys()
            ),
            "expected_PH5_denom": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH5_denom"].keys()
            ),
            "expected_PH6": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH6"].keys()
            ),
            "expected_PH7": pd.read_csv(
                io.BytesIO(
                    pkgutil.get_data(  # type: ignore
                        "tmlt.phsafe", RESOURCES_PACKAGE_NAME + "/test/test10/PH7.csv"
                    )
                ),
                encoding="utf8",
                dtype={
                    "REGION_ID": object,
                    "REGION_TYPE": object,
                    "ITERATION_CODE": object,
                    "PH7_DATA_CELL": int,
                },
            ),
            "expected_PH8_num": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH8_num"].keys()
            ),
            "expected_PH8_denom": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH8_denom"].keys()
            ),
        },
        {  # test for ph8
            "person_data": [
                ("3", 100000001, 25, 1, "02", "20"),
                ("3", 100000002, 25, 1, "02", "20"),
                ("3", 100000003, 25, 1, "02", "20"),
                ("3", 100000004, 25, 1, "02", "20"),
            ],
            "unit_data": [
                ("2", 100000001, 1, 0, 1, "02", "1", "4", "09", "5"),
                ("2", 100000002, 1, 0, 1, "02", "2", "4", "09", "5"),
                ("2", 100000003, 1, 0, 1, "02", "3", "4", "09", "5"),
                ("2", 100000004, 1, 0, 1, "02", "4", "4", "09", "5"),
            ],
            "geo_data": [
                (100000001, "00", "01"),
                (100000002, "00", "01"),
                (100000003, "00", "01"),
                (100000004, "00", "01"),
            ],
            "expected_PH2": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH2"].keys()
            ),
            "expected_PH3": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH3"].keys()
            ),
            "expected_PH1_num": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH1_num"].keys()
            ),
            "expected_PH1_denom": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH1_denom"].keys()
            ),
            "expected_PH4": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH4"].keys()
            ),
            "expected_PH5_num": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH5_num"].keys()
            ),
            "expected_PH5_denom": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH5_denom"].keys()
            ),
            "expected_PH6": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH6"].keys()
            ),
            "expected_PH7": pd.DataFrame(
                [], columns=TABULATION_OUTPUT_COLUMNS["PH7"].keys()
            ),
            "expected_PH8_num": pd.read_csv(
                io.BytesIO(
                    pkgutil.get_data(  # type: ignore
                        "tmlt.phsafe",
                        RESOURCES_PACKAGE_NAME + "/test/test11/PH8_num.csv",
                    )
                ),
                encoding="utf8",
                dtype={
                    "REGION_ID": object,
                    "REGION_TYPE": object,
                    "ITERATION_CODE": object,
                    "PH8_NUM_DATA_CELL": int,
                },
            ),
            "expected_PH8_denom": pd.read_csv(
                io.BytesIO(
                    pkgutil.get_data(  # type: ignore
                        "tmlt.phsafe",
                        RESOURCES_PACKAGE_NAME + "/test/test11/PH8_denom.csv",
                    )
                ),
                encoding="utf8",
                dtype={
                    "REGION_ID": object,
                    "REGION_TYPE": object,
                    "ITERATION_CODE": object,
                    "PH8_DENOM_DATA_CELL": int,
                },
            ),
        },
    ],
    scope="class",
)
class TestNonPrivateTabulation:
    """Parameterized unit tests for nonprivate tabulations."""

    tables: PHSafeOutput
    expected_PH2: pd.DataFrame
    expected_PH3: pd.DataFrame
    expected_PH1_num: pd.DataFrame
    expected_PH1_denom: pd.DataFrame
    expected_PH4: pd.DataFrame
    expected_PH5_num: pd.DataFrame
    expected_PH5_denom: pd.DataFrame
    expected_PH6: pd.DataFrame
    expected_PH7: pd.DataFrame
    expected_PH8_num: pd.DataFrame
    expected_PH8_denom: pd.DataFrame

    def test_correct_ph1_num(self):
        """Test that the PH1_num table is correct."""

        df1 = self.tables.PH1_num.toPandas()
        df2 = self.expected_PH1_num
        if df2.size > 0:
            assert_frame_equal_with_sort(
                df1, df2, list(TABULATION_OUTPUT_COLUMNS["PH1_num"].keys())
            )

    def test_correct_ph1_denom(self):
        """Test that the PH1_denom table is correct."""

        df1 = self.tables.PH1_denom.toPandas()
        df2 = self.expected_PH1_denom
        if df2.size > 0:
            assert_frame_equal_with_sort(
                df1, df2, list(TABULATION_OUTPUT_COLUMNS["PH1_denom"].keys())
            )

    def test_correct_ph2(self):
        """Test that the PH2 table is correct."""

        if self.expected_PH2.size > 0:
            assert_frame_equal_with_sort(
                self.tables.PH2.toPandas(),
                self.expected_PH2,
                ["REGION_ID", "REGION_TYPE", "PH2_DATA_CELL", "COUNT"],
            )

    def test_correct_ph3(self):
        """Test that the PH3 table is correct."""

        df1 = self.tables.PH3.toPandas()
        df2 = self.expected_PH3
        if df2.size > 0:
            assert_frame_equal_with_sort(
                df1, df2, list(TABULATION_OUTPUT_COLUMNS["PH3"].keys())
            )

    def test_correct_ph4(self):
        """Test that the PH4 table is correct."""

        df1 = self.tables.PH4.toPandas()
        df2 = self.expected_PH4
        if df2.size > 0:
            assert_frame_equal_with_sort(
                df1, df2, list(TABULATION_OUTPUT_COLUMNS["PH4"].keys())
            )

    def test_correct_ph5_num(self):
        """Test that the PH5_num table is correct."""

        df1 = self.tables.PH5_num.toPandas()
        df2 = self.expected_PH5_num
        if df2.size > 0:
            assert_frame_equal_with_sort(
                df1, df2, list(TABULATION_OUTPUT_COLUMNS["PH5_num"].keys())
            )

    def test_correct_ph5_denom(self):
        """Test that the PH5_denom table is correct."""

        df1 = self.tables.PH5_denom.toPandas()
        df2 = self.expected_PH5_denom
        if df2.size > 0:
            assert_frame_equal_with_sort(
                df1, df2, list(TABULATION_OUTPUT_COLUMNS["PH5_denom"].keys())
            )

    def test_correct_ph6(self):
        """Test that the PH6 table is correct."""

        df1 = self.tables.PH6.toPandas()
        df2 = self.expected_PH6
        if df2.size > 0:
            assert_frame_equal_with_sort(
                df1, df2, list(TABULATION_OUTPUT_COLUMNS["PH6"].keys())
            )

    def test_correct_ph7(self):
        """Test that the PH7 table is correct."""

        df1 = self.tables.PH7.toPandas()
        df2 = self.expected_PH7
        if df2.size > 0:
            assert_frame_equal_with_sort(
                df1, df2, list(TABULATION_OUTPUT_COLUMNS["PH7"].keys())
            )

    def test_correct_ph8_num(self):
        """Test that the PH8_num table is correct."""

        df1 = self.tables.PH8_num.toPandas()
        df2 = self.expected_PH8_num
        if df2.size > 0:
            assert_frame_equal_with_sort(
                df1, df2, list(TABULATION_OUTPUT_COLUMNS["PH8_num"].keys())
            )

    def test_correct_ph8_denom(self):
        """Test that the PH8_denom table is correct."""

        df1 = self.tables.PH8_denom.toPandas()
        df2 = self.expected_PH8_denom
        if df2.size > 0:
            assert_frame_equal_with_sort(
                df1, df2, list(TABULATION_OUTPUT_COLUMNS["PH8_denom"].keys())
            )
