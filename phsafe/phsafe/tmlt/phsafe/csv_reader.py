"""Dummy csv reader module for synthetic data."""

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

import os
from typing import List

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col

from tmlt.analytics._schema import Schema, analytics_to_spark_schema
from tmlt.phsafe.input_processing import GEO_SCHEMA, PERSON_SCHEMA, UNIT_SCHEMA
from tmlt.phsafe.reader_interface import AbstractPHSafeReader

GEO_FILENAME = "geo.csv"
"""Filename for the geo df."""

UNIT_FILENAME = "units.csv"
"""Filename for the unit df."""

PERSON_FILENAME = "persons.csv"
"""Filename for the person df."""


class CSVReader(AbstractPHSafeReader):
    """The reader for csv files."""

    def __init__(self, config_path: str, state_filter: List[str]):
        """Sets up csv reader.

        Args:
            config_path: path to directory containing geo, unit, and person files.
            state_filter: list of state codes to keep
        """
        self.config_path = config_path
        self.states = state_filter
        self.spark = SparkSession.builder.getOrCreate()

    def get_geo_df(self) -> DataFrame:
        """Returns geo df filtered per state_filter."""
        schema = analytics_to_spark_schema(Schema(GEO_SCHEMA))
        geo_df = self.spark.read.csv(
            os.path.join(self.config_path, GEO_FILENAME), schema=schema, header=True
        )
        return geo_df.filter(col("TABBLKST").isin(self.states))

    def get_unit_df(self) -> DataFrame:
        """Returns unit df filtered per state_filter."""
        # Get MAFIDs restricted to the filtered states in order to filter unit_df.
        geo_df = self.get_geo_df().select("MAFID")
        schema = analytics_to_spark_schema(Schema(UNIT_SCHEMA))
        unit_df = self.spark.read.csv(
            os.path.join(self.config_path, UNIT_FILENAME), schema=schema, header=True
        )
        columns = unit_df.columns
        return unit_df.join(geo_df, on="MAFID", how="inner").select(columns)

    def get_person_df(self) -> DataFrame:
        """Return person df filtered per state_filter."""
        # Get MAFIDs restricted to the filtered states in order to filter person_df.
        geo_df = self.get_geo_df().select("MAFID")
        schema = analytics_to_spark_schema(Schema(PERSON_SCHEMA))
        person_df = self.spark.read.csv(
            os.path.join(self.config_path, PERSON_FILENAME), schema=schema, header=True
        )
        columns = person_df.columns
        return person_df.join(geo_df, on="MAFID", how="inner").select(columns)
