"""Tests input validation on spark dataframes."""

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

import pytest
from pyspark.sql.functions import col, lit, when  # pylint: disable=no-name-in-module
from pyspark.sql.types import LongType, StringType, StructField, StructType

from tmlt.common.pyspark_test_tools import pyspark  # pylint: disable=unused-import
from tmlt.phsafe import PHSafeInput
from tmlt.phsafe.input_validation import validate_input


@pytest.fixture(scope="class")
def setup_input_validation(spark, request):
    """Set up test."""

    person = spark.createDataFrame(
        spark.sparkContext.parallelize(
            [
                ("3", 100000001, 25, 1, "34", "20"),
                ("3", 100000001, 25, 1, "34", "20"),
                ("3", 100000002, 25, 1, "34", "20"),
                ("3", 899999999, 25, 1, "34", "20"),
            ]
        ),
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
        spark.sparkContext.parallelize(
            [
                ("2", 100000001, 99999, 0, 1, "34", "4", "2", "09", "0"),
                ("2", 100000002, 1, 0, 1, "34", "4", "2", "09", "0"),
                ("2", 899999999, 0, 0, 1, "34", "4", "2", "09", "0"),
            ]
        ),
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
    geo = spark.createDataFrame(
        spark.sparkContext.parallelize(
            [
                (
                    "2",
                    100000001,
                    "01",
                    "001",
                    "000100",
                    "0001",
                    "0",
                    "1",
                    "0",
                    "99999",
                    "0001",
                ),
                (
                    "2",
                    100000002,
                    "02",
                    "001",
                    "000101",
                    "0001",
                    "1",
                    "4",
                    "4",
                    "00001",
                    "0001",
                ),
                (
                    "2",
                    899999999,
                    "02",
                    "001",
                    "000101",
                    "0001",
                    "9",
                    "9",
                    "9",
                    "89999",
                    "0001",
                ),
            ]
        ),
        StructType(
            [
                StructField("RTYPE", StringType(), True),
                StructField("MAFID", LongType(), True),
                StructField("TABBLKST", StringType(), True),
                StructField("TABBLKCOU", StringType(), True),
                StructField("TABTRACTCE", StringType(), True),
                StructField("TABBLK", StringType(), True),
                StructField("TABBLKGRPCE", StringType(), True),
                StructField("REGIONCE", StringType(), True),
                StructField("DIVISIONCE", StringType(), True),
                StructField("PLACEFP", StringType(), True),
                StructField("AIANNHCE", StringType(), True),
            ]
        ),
    )
    request.cls.input_sdfs = PHSafeInput(persons=person, units=unit, geo=geo)


@pytest.mark.usefixtures("spark")
@pytest.mark.usefixtures("setup_input_validation")
class TestInputValidation:
    """Parameterized unit tests for input validation."""

    input_sdfs: PHSafeInput

    @pytest.fixture(autouse=True)
    def error_caplog(self, caplog):
        """Capture log messages at the error level."""
        caplog.set_level(logging.ERROR)

    def test_valid_input(self, caplog):
        """Input is correct."""
        caplog.set_level(logging.INFO)
        assert validate_input(self.input_sdfs)
        assert "Phase 3 successful. All files are as expected." in caplog.text

    def test_valid_input_filter_states(self, caplog):
        """Input is correct and states are filtered."""
        caplog.set_level(logging.INFO)
        assert validate_input(self.input_sdfs, ["01", "02"])
        assert "Phase 3 successful. All files are as expected." in caplog.text

    def test_invalid_input(self):
        """Input spark dataframes are incorrect."""
        with pytest.raises(TypeError):
            PHSafeInput(persons=self.input_sdfs.persons, units=self.input_sdfs.units)

    def test_unexpected_columns_phase2_failure(self, spark, caplog):
        """Input validation fails if unexpected columns are found.

        And exception is the `GRF-C.txt`, see
        :func:`.test_unexpected_columns_grfc_okay`.
        """
        person = spark.createDataFrame(
            [("3", 100000001, 25, 1, "34", "20", "new")],
            StructType(
                [
                    StructField("RTYPE", StringType(), True),
                    StructField("MAFID", LongType(), True),
                    StructField("QAGE", LongType(), True),
                    StructField("CENHISP", LongType(), True),
                    StructField("CENRACE", StringType(), True),
                    StructField("RELSHIP", StringType(), True),
                    StructField("new", StringType(), True),
                ]
            ),
        )
        new_sdf_mapping = PHSafeInput(
            persons=person, units=self.input_sdfs.units, geo=self.input_sdfs.geo
        )
        okay = validate_input(new_sdf_mapping)
        assert not okay
        assert "Errors found in phase 2. See above." in caplog.text

    def test_invalid_type_int_phase1_failure(self, caplog):
        """Input validatation fails if the type is invalid (str->int)."""
        new_sdf_mapping = PHSafeInput(
            persons=self.input_sdfs.persons.withColumn(
                "CENRACE", self.input_sdfs.persons["CENRACE"].cast(LongType())
            ),
            units=self.input_sdfs.units,
            geo=self.input_sdfs.geo,
        )
        okay = validate_input(new_sdf_mapping)
        assert not okay
        assert "Errors found in phase 1. See above." in caplog.text

    def test_invalid_type_str_phase1_failure(self, caplog):
        """Input validatation fails if the type is invalid (int->str)."""
        new_sdf_mapping = PHSafeInput(
            persons=self.input_sdfs.persons.withColumn(
                "MAFID", self.input_sdfs.persons["MAFID"].cast(StringType())
            ),
            units=self.input_sdfs.units,
            geo=self.input_sdfs.geo,
        )
        okay = validate_input(new_sdf_mapping)
        assert not okay
        assert "Errors found in phase 1. See above." in caplog.text

    def test_invalid_format_phase2_failure(self, caplog):
        """Input validatation fails if the format is invalid."""
        new_sdf_mapping = PHSafeInput(
            persons=self.input_sdfs.persons.withColumn(
                "MAFID", (self.input_sdfs.persons["MAFID"] / 1000).cast(LongType())
            ),
            units=self.input_sdfs.units,
            geo=self.input_sdfs.geo,
        )
        okay = validate_input(new_sdf_mapping)
        assert not okay
        assert "Errors found in phase 2. See above." in caplog.text

    def test_invalid_mafid_domain_phase2_failure(self, caplog):
        """Input validatation fails if the mafid is above its range."""
        new_sdf_mapping = PHSafeInput(
            persons=self.input_sdfs.persons.withColumn(
                "MAFID",
                when(col("MAFID").isNotNull(), lit(900000000))
                .otherwise(lit(None))
                .cast(LongType()),
            ),
            units=self.input_sdfs.units,
            geo=self.input_sdfs.geo,
        )
        okay = validate_input(new_sdf_mapping)
        assert not okay
        assert "Errors found in phase 2. See above." in caplog.text

    def test_invalid_final_pop_domain_phase2_failure(self, caplog):
        """Input validatation fails if the mafid is above its range."""
        new_sdf_mapping = PHSafeInput(
            persons=self.input_sdfs.persons,
            units=self.input_sdfs.units.withColumn(
                "FINAL_POP",
                when(col("FINAL_POP").isNotNull(), lit(100000))
                .otherwise(lit(None))
                .cast(LongType()),
            ),
            geo=self.input_sdfs.geo,
        )
        okay = validate_input(new_sdf_mapping)
        assert not okay
        assert "Errors found in phase 2. See above." in caplog.text

    def test_column_ordering_phase2_failure(self, caplog):
        """Input validation fails if columns are not in the correct order."""
        new_sdf_mapping = PHSafeInput(
            persons=self.input_sdfs.persons.select(
                "RELSHIP", "RTYPE", "QAGE", "CENHISP", "CENRACE", "MAFID"
            ),
            units=self.input_sdfs.units,
            geo=self.input_sdfs.geo,
        )
        okay = validate_input(new_sdf_mapping)
        assert not okay
        assert "Errors found in phase 2. See above." in caplog.text

    def test_unit_mafids_as_domain_for_person_mafids_phase3_failure(self, caplog):
        """Input validation checks MAFIDs in the person file are in the units file."""
        new_sdf_mapping = PHSafeInput(
            persons=self.input_sdfs.persons.withColumn(
                "MAFID",
                when(col("MAFID").isNotNull(), lit(100000010))
                .otherwise(lit(None))
                .cast(LongType()),
            ),
            units=self.input_sdfs.units,
            geo=self.input_sdfs.geo,
        )
        okay = validate_input(new_sdf_mapping)
        assert not okay
        assert "Errors found in phase 3. See above." in caplog.text
        assert (
            "Found MAFIDs in the persons file that are not in the units file:"
            " [100000010]"
            in caplog.text
        )

    def test_unit_mafids_as_domain_for_geo_mafids_phase3_failure(self, caplog):
        """Input validation checks MAFIDs in the geo file are in the units file."""
        new_sdf_mapping = PHSafeInput(
            persons=self.input_sdfs.persons,
            units=self.input_sdfs.units,
            geo=self.input_sdfs.geo.withColumn(
                "MAFID",
                when(col("MAFID").isNotNull(), lit(100000010))
                .otherwise(lit(None))
                .cast(LongType()),
            ),
        )
        okay = validate_input(new_sdf_mapping)
        assert not okay
        assert "Errors found in phase 3. See above." in caplog.text
        assert (
            "Found MAFIDs in the geo file that are not in the units file: [100000010]"
            in caplog.text
        )

    def test_geo_mafids_as_domain_for_unit_mafids_phase3_failure(self, spark, caplog):
        """Input validation checks MAFIDs in the units file are in the geo file."""
        geo = spark.createDataFrame(
            spark.sparkContext.parallelize(
                [
                    (
                        "2",
                        100000001,
                        "01",
                        "001",
                        "000100",
                        "0001",
                        "1",
                        "1",
                        "0",
                        "99999",
                        "0001",
                    )
                ]
            ),
            StructType(
                [
                    StructField("RTYPE", StringType(), True),
                    StructField("MAFID", LongType(), True),
                    StructField("TABBLKST", StringType(), True),
                    StructField("TABBLKCOU", StringType(), True),
                    StructField("TABTRACTCE", StringType(), True),
                    StructField("TABBLK", StringType(), True),
                    StructField("TABBLKGRPCE", StringType(), True),
                    StructField("REGIONCE", StringType(), True),
                    StructField("DIVISIONCE", StringType(), True),
                    StructField("PLACEFP", StringType(), True),
                    StructField("AIANNHCE", StringType(), True),
                ]
            ),
        )
        new_sdf_mapping = PHSafeInput(
            persons=self.input_sdfs.persons, units=self.input_sdfs.units, geo=geo
        )
        okay = validate_input(new_sdf_mapping)
        assert not okay
        assert "Errors found in phase 3. See above." in caplog.text
        assert (
            "Found MAFIDs in the units file that are not in the geo file: [100000002,"
            " 899999999]"
            in caplog.text
        )

    def test_filter_states_phase3_failure(self, caplog):
        """Input validation fails if some states are outside of filter_states."""
        okay = validate_input(self.input_sdfs, ["01"])
        assert not okay
        assert "Errors found in phase 3. See above." in caplog.text
        assert (
            "Found TABBLKSTs in the geo file that are not in the state filter: ['02']"
            in caplog.text
        )

    def test_duplicate_unit_mafids_fail(self, spark, caplog):
        """Input validation fails if a MAFID appears twice in the units file."""
        units_with_duplicate_mafid = spark.createDataFrame(
            spark.sparkContext.parallelize(
                [
                    ("2", 100000001, 1, 0, 1, "34", "4", "2", "09", "0"),
                    ("2", 100000001, 99999, 0, 1, "34", "4", "2", "09", "0"),
                    ("2", 100000002, 1, 0, 1, "34", "4", "2", "09", "0"),
                    ("2", 899999999, 0, 0, 1, "34", "4", "2", "09", "0"),
                ]
            ),
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
        new_input = PHSafeInput(
            persons=self.input_sdfs.persons,
            units=units_with_duplicate_mafid,
            geo=self.input_sdfs.geo,
        )
        okay = validate_input(new_input)
        assert not okay
        assert "Errors found in phase 3. See above." in caplog.text
        assert (
            "Found two rows with the same MAFID in the units file: [100000001]"
            in caplog.text
        )

    def test_duplicate_geo_mafids_fail(self, spark, caplog):
        """Input validation fails if a MAFID appears twice in the geo file."""
        geo_with_duplicate_mafid = spark.createDataFrame(
            spark.sparkContext.parallelize(
                [
                    (
                        "2",
                        100000001,
                        "01",
                        "001",
                        "000100",
                        "0001",
                        "0",
                        "1",
                        "0",
                        "99999",
                        "0001",
                    ),
                    (
                        "2",
                        100000001,
                        "01",
                        "001",
                        "000100",
                        "0002",
                        "0",
                        "1",
                        "0",
                        "99999",
                        "0001",
                    ),
                    (
                        "2",
                        100000002,
                        "02",
                        "001",
                        "000101",
                        "0001",
                        "1",
                        "4",
                        "4",
                        "00001",
                        "0001",
                    ),
                    (
                        "2",
                        899999999,
                        "02",
                        "001",
                        "000101",
                        "0001",
                        "9",
                        "9",
                        "9",
                        "89999",
                        "0001",
                    ),
                ]
            ),
            StructType(
                [
                    StructField("RTYPE", StringType(), True),
                    StructField("MAFID", LongType(), True),
                    StructField("TABBLKST", StringType(), True),
                    StructField("TABBLKCOU", StringType(), True),
                    StructField("TABTRACTCE", StringType(), True),
                    StructField("TABBLK", StringType(), True),
                    StructField("TABBLKGRPCE", StringType(), True),
                    StructField("REGIONCE", StringType(), True),
                    StructField("DIVISIONCE", StringType(), True),
                    StructField("PLACEFP", StringType(), True),
                    StructField("AIANNHCE", StringType(), True),
                ]
            ),
        )
        new_input = PHSafeInput(
            persons=self.input_sdfs.persons,
            units=self.input_sdfs.units,
            geo=geo_with_duplicate_mafid,
        )
        okay = validate_input(new_input)
        assert not okay
        assert "Errors found in phase 3. See above." in caplog.text
        assert (
            "Found two rows with the same MAFID in the geo file: [100000001]"
            in caplog.text
        )
