import os, sys, functools

if 'SPARK_HOME' not in os.environ:
   os.environ['SPARK_HOME'] = '/usr/lib/spark'
sys.path.append(os.path.join(os.environ['SPARK_HOME'], 'python'))
sys.path.append(os.path.join(os.environ['SPARK_HOME'], 'python', 'lib', 'py4j-src.zip'))
from pyspark.sql import SparkSession
import pyspark.sql.functions as F

from pyspark.sql.types import *
from pyspark.sql.functions import col, lit, concat

from ctools.s3 import s3open
from phsafe_safetab_reader.cef_validator_classes import CEF20_UNIT, CEF20_PER
import configparser

"""The interface for SafeTab readers."""

from abc import ABC
from typing import List
from pyspark.sql import DataFrame as SparkDataFrame

from phsafe_safetab_reader.safetab_cef_utils import get_pop_count, get_geo_pop_file_list

class CEFHReader(ABC):
    def __init__(self, config_path: str, state_filter: List[str]):
        config = configparser.ConfigParser()
        if config_path.startswith("s3://"):
            with s3open(config_path) as s3_config_file:
                config.read_file(s3_config_file)
        else:
            config.read(config_path)

        paths = config['paths']
        #print(paths)

        """Initialize reader with config path and the state FIPS codes to read."""
        spark = SparkSession.builder.getOrCreate()

        per_validator = CEF20_PER
        unit_validator = CEF20_UNIT

        per_s3_dir = paths["per_dir"]
        unit_s3_dir = paths["unit_dir"]
        per_s3_dir_pr = paths["per_dir_pr"]
        unit_s3_dir_pr = paths["unit_dir_pr"]
        geo_s3_dir = paths["grfc_dir"]
        pop_s3_dir = paths["pop_dir"]
        per_file_format = paths["per_file_format"]
        unit_file_format = paths["unit_file_format"]
        geo_file_format = paths["geo_file_format"]

        # Puerto Rico files may be in a different directory
        # create person dataframe for specified states/PR

        if len(state_filter):
            per_files = [per_s3_dir + per_file_format % (code,) for code in state_filter if code != "72"]
            if "72" in state_filter:
                per_files.append(per_s3_dir_pr + per_file_format % "72")
            per_files = ",".join(per_files)
        else:
            per_files = per_s3_dir
            per_files = per_files + "," + per_s3_dir_pr + per_file_format % "72"

        per_rdd = spark.sparkContext.textFile(per_files)
        per_rdd = per_rdd.map(lambda line: per_validator.parse_line(line))
        self.per_df = spark.createDataFrame(per_rdd).cache()

        # create unit dataframe for specified states/PR

        if len(state_filter):
            unit_files = [unit_s3_dir + unit_file_format % (code,) for code in state_filter if code != "72"]
            if "72" in state_filter:
                unit_files.append(unit_s3_dir_pr + unit_file_format % "72")
            unit_files = ",".join(unit_files)
        else:
            unit_files = unit_s3_dir
            unit_files = unit_files + "," + unit_s3_dir_pr + unit_file_format % "72"

        unit_rdd = spark.sparkContext.textFile(unit_files)
        unit_rdd = unit_rdd.map(lambda line: unit_validator.parse_line(line))
        self.unit_df = spark.createDataFrame(unit_rdd).cache()

        self.unit_df_input = self.unit_df

        # filter unit_df to only include rows where RTYPE == "2"
        self.unit_df = self.unit_df.filter(col("RTYPE") == "2")

        if len(state_filter):
            geo_files = get_geo_pop_file_list(True, geo_s3_dir, pop_s3_dir, state_filter)
        else:
            geo_files = geo_s3_dir

        self.geo_df = spark.read \
                .option("header", "true") \
                .option("delimiter", "|") \
                .csv(geo_files) \
                .withColumn("OIDTABBLK", F.col("OIDTABBLK").cast(LongType()))

        self.geo_df_input = self.geo_df

        # Create final per_df by joining per_df with unit_df using MAFID and then joining with geo_df using OIDTB
        joined_df = self.per_df.join(self.unit_df, self.per_df.mafid == self.unit_df.mafid, "inner")
        multi_joined_df = joined_df.join(self.geo_df, joined_df.oidtb == self.geo_df.OIDTABBLK, "inner")

        # filter GRFC to include only geographic entries with units
        # first, get a de-duplicated list of all Unit's oidtb values
        unit_id_df = self.unit_df[["oidtb"]].drop_duplicates()
        # then outer-join unique unit oidtb to GRFC entries
        self.geo_df = self.geo_df.join(unit_id_df, unit_id_df.oidtb == self.geo_df.OIDTABBLK, "rightouter")[list(self.geo_df.columns)]


        # test GRFC state -- if this fails there exists a unit with an OIDTB that is nonexistent in the GRFC
        # num_null = self.geo_df.filter(self.geo_df.TABBLKST.isNull()).count()
        # assert num_null == 0, f'There are {num_null} blocks in the CEF units file that are not in the GRFC.'

        # Add column "Householder" that is true when relationship is 20, false otherwise
        """
        self.per_df = multi_joined_df.withColumn("HOUSEHOLDER", F.when(F.col("relship") == "20", "True")\
                .otherwise(F.lit("False")))
        """

        # Final unit_df is similar to the final per_df. Instead of adding column "Householder", we add column
        # "Household_type" which is derived from HHT in CEF20_UNIT. Nonfamily households no longer separated by gender
        self.unit_df = multi_joined_df.withColumn("HOUSEHOLD_TYPE", F.when(F.col("hht") == 6, 4)\
                .when(F.col("hht") == 7, 5)\
                .otherwise(F.lit(F.col("hht"))))
        # Only get households
        self.unit_df = self.unit_df.filter(self.unit_df.relship == "20")
        # Drop households where hht is 0
        self.unit_df = self.unit_df.filter(self.unit_df.hht != 0)

        def capitalize_column_names(df):
            for col in df.columns:
                df = df.withColumnRenamed(col, col.upper())
            return df

        # Replace HHRACE whitespace with "Null" to match validation
        self.unit_df = self.unit_df.withColumn("HHRACE", F.regexp_replace(F.col("HHRACE"), "^\s+$", "Null"))

        #self.per_df = capitalize_column_names(self.per_df)
        self.unit_df = capitalize_column_names(self.unit_df)
        self.geo_df = capitalize_column_names(self.geo_df)

      # Process the T1 from SafeTab-P output
        """ get the list of pop_group_totals files """
        pop_files = get_geo_pop_file_list(False, geo_s3_dir, pop_s3_dir, state_filter)

        """ Can use spark to read csv files into data frame instead of pandas. This need to test """
        pop_group_total_df = spark.read \
                .option("header", "true") \
                .option("delimiter", "|") \
                .csv(pop_files)

        self.pop_df_input = pop_group_total_df

        self.pop_counts = get_pop_count(pop_files)

        # This filters all regions except the AIANNH ones which can overlap states.
        # AIANNH is filtered below based on which AIANNH values pass the state
        # filter from the GRF-C.
        # Note this assumes REGION_IDs are read in as strings and that state codes in the state
        # filter list are strings.
        state_filter_algo = col("REGION_TYPE").isin(["USA"]) | (
            ~col("REGION_TYPE").isin(["USA", "AIANNH"])
            & functools.reduce(
                lambda x, y: x | y,
                [col("REGION_ID").startswith(state) for state in state_filter],
                lit(False),
            )
        )

        pop_group_non_aiannh = pop_group_total_df.filter(state_filter_algo)

        # This filters for all AIANNH values that can exist in the states passed into
        # the state filter. It then inner joins onto the pop groups to leave only
        # possible aiannh values.
        t1_AIANNH = pop_group_total_df.filter(col("REGION_TYPE") == "AIANNH")

        aiannh_regions = (
            self.get_geo_df()
            .withColumnRenamed("AIANNHCE", "REGION_ID")
            .select("REGION_ID")
            .distinct()
        )

        # Note this join assumes the join columns  are loaded as string datatypes in both the geo df and the t1 dataframe
        t1_AIANNH = t1_AIANNH.join(
            aiannh_regions, on="REGION_ID"
        )

        # inner join pop_df to geo_df based on region IDs (which need to be constructed for each type)
        t1_USA = pop_group_non_aiannh.filter(col("REGION_TYPE") == "USA")

        t1_STATE = pop_group_non_aiannh.filter(col("REGION_TYPE").isin(["STATE", "PR-STATE"]))
        states = (
            self.geo_df
            .withColumnRenamed("TABBLKST", "REGION_ID")
            .select("REGION_ID")
            .distinct()
        )
        t1_STATE = t1_STATE.join(states, on="REGION_ID")

        t1_COUNTY = pop_group_non_aiannh.filter(col("REGION_TYPE").isin(["COUNTY", "PR-COUNTY"]))
        counties = (
            self.geo_df
            .withColumn("REGION_ID", concat(self.geo_df.TABBLKST, self.geo_df.TABBLKCOU))
            .select("REGION_ID")
            .distinct()
        )
        t1_COUNTY = t1_COUNTY.join(counties, on="REGION_ID")

        t1_TRACT = pop_group_non_aiannh.filter(col("REGION_TYPE").isin(["TRACT", "PR-TRACT"]))
        tracts = (
            self.geo_df
            .withColumn("REGION_ID", concat(self.geo_df.TABBLKST, self.geo_df.TABBLKCOU, self.geo_df.TABTRACTCE))
            .select("REGION_ID")
            .distinct()
        )
        t1_TRACT = t1_TRACT.join(tracts, on="REGION_ID")

        t1_PLACE = pop_group_non_aiannh.filter(col("REGION_TYPE").isin(["PLACE", "PR-PLACE"]))
        places = (
            self.geo_df
            .withColumn("REGION_ID", concat(self.geo_df.TABBLKST, self.geo_df.PLACEFP))
            .select("REGION_ID")
            .distinct()
        )
        t1_PLACE = t1_PLACE.join(places, on="REGION_ID")

        self.pop_df = t1_USA.union(t1_STATE).union(t1_COUNTY).union(t1_TRACT).union(t1_PLACE).union(t1_AIANNH)

        self.pop_df = self.pop_df.withColumn("COUNT", col("COUNT").cast(IntegerType()))

        # Cast these columns to strings to match validation
        self.unit_df = self.unit_df.withColumn("QRACE1", F.col("QRACE1").cast(StringType()))
        self.unit_df = self.unit_df.withColumn("QRACE2", F.col("QRACE2").cast(StringType()))
        self.unit_df = self.unit_df.withColumn("QRACE3", F.col("QRACE3").cast(StringType()))
        self.unit_df = self.unit_df.withColumn("QRACE4", F.col("QRACE4").cast(StringType()))
        self.unit_df = self.unit_df.withColumn("QRACE5", F.col("QRACE5").cast(StringType()))
        self.unit_df = self.unit_df.withColumn("QRACE6", F.col("QRACE6").cast(StringType()))
        self.unit_df = self.unit_df.withColumn("QRACE7", F.col("QRACE7").cast(StringType()))
        self.unit_df = self.unit_df.withColumn("QRACE8", F.col("QRACE8").cast(StringType()))
        self.unit_df = self.unit_df.withColumn("QSPAN", F.col("QSPAN").cast(StringType()))

        # Cast these columns to integers to match validation
        self.unit_df = self.unit_df.withColumn("HOUSEHOLD_TYPE", F.col("HOUSEHOLD_TYPE").cast(IntegerType()))
        self.unit_df = self.unit_df.withColumn("TEN", F.col("TEN").cast(IntegerType()))

        # Replace QRACE1-8 whitespace with "Null" per SafeTab Spec
        self.unit_df = self.unit_df.withColumn("QRACE1", F.when(F.col("QRACE1") == "    ", "Null")\
                .otherwise(F.lit(F.col("QRACE1"))))
        self.unit_df = self.unit_df.withColumn("QRACE2", F.when(F.col("QRACE2") == "    ", "Null")\
                .otherwise(F.lit(F.col("QRACE2"))))
        self.unit_df = self.unit_df.withColumn("QRACE3", F.when(F.col("QRACE3") == "    ", "Null")\
                .otherwise(F.lit(F.col("QRACE3"))))
        self.unit_df = self.unit_df.withColumn("QRACE4", F.when(F.col("QRACE4") == "    ", "Null")\
                .otherwise(F.lit(F.col("QRACE4"))))
        self.unit_df = self.unit_df.withColumn("QRACE5", F.when(F.col("QRACE5") == "    ", "Null")\
                .otherwise(F.lit(F.col("QRACE5"))))
        self.unit_df = self.unit_df.withColumn("QRACE6", F.when(F.col("QRACE6") == "    ", "Null")\
                .otherwise(F.lit(F.col("QRACE6"))))
        self.unit_df = self.unit_df.withColumn("QRACE7", F.when(F.col("QRACE7") == "    ", "Null")\
                .otherwise(F.lit(F.col("QRACE7"))))
        self.unit_df = self.unit_df.withColumn("QRACE8", F.when(F.col("QRACE8") == "    ", "Null")\
                .otherwise(F.lit(F.col("QRACE8"))))

        # cature pre-projection Unit DF state for testing
        self.unit_df_full_fields = self.unit_df

        # perform projection and ordering of columns per specification
        #self.per_df = self.per_df.select("QAGE", "QSEX", "HOUSEHOLDER", "TABBLKST", "TABBLKCOU", "TABTRACTCE", "TABBLK", "CENRACE", "QRACE1", "QRACE2", "QRACE3", "QRACE4", "QRACE5", "QRACE6", "QRACE7", "QRACE8", "QSPAN")
        self.unit_df = self.unit_df.select("TABBLKST", "TABBLKCOU", "TABTRACTCE", "TABBLK", "HHRACE", "QRACE1", "QRACE2", "QRACE3", "QRACE4", "QRACE5", "QRACE6", "QRACE7", "QRACE8", "QSPAN", "HOUSEHOLD_TYPE", "TEN")

        def rename_duplicate_columns(dataframe):
            columns = dataframe.columns
            duplicate_column_indices = list(set([columns.index(col) for col in columns if columns.count(col) == 2]))
            for index in duplicate_column_indices:
                columns[index] = columns[index]+'2'
            dataframe = dataframe.toDF(*columns)
            return dataframe
        self.unit_df_full_fields = rename_duplicate_columns(self.unit_df_full_fields)


    def get_geo_df(self) -> SparkDataFrame:
        """Return a spark dataframe derived from the CEF Unit file and the GRFC.
        This should contain only the state codes specified by the state_filter.
        Returns:
            A spark dataframe containing all of the expected columns from `geo_df`_ in
                `Appendix A`_.
        """
        return self.geo_df

    def get_unit_df(self) -> SparkDataFrame:
        """Return a spark dataframe derived from the CEF person file.
        This should contain only records with MAFIDs corresponding to the state codes
        specified by the state_filter.
        Returns:
            A spark dataframe containing all of the expected columns from `unit_df`_
                in `Appendix A`_.
        """
        return self.unit_df

    def get_pop_group_details_df(self):
        """Return a spark dataframe with population group details
        Returns:
            A total counts of all the population group details
        """
        return self.pop_df

    def get_unit_df_input(self):
        """ Return a unit spark data stream that read directly from the csv files
        Return:
           A spark dataframe contains all the expected collumns from unit csv files
        """
        return self.unit_df_input

    def get_unit_df_full_fields(self):
        """ Return a unit spark data stream that has all fields prior to limiting down the fields
        Return:
           A spark dataframe contains all the unit DF fields prior to limiting down the fields
        """
        return self.unit_df_full_fields

    def get_geo_df_input(self):
        """ Return a geo spark data stream that read directly from the files
        Return:
           A spark dataframe contains all the expected collumns from geo files
        """
        return self.geo_df_input

    def get_pop_df_input(self):
        """ Return a population spark data stream that read directly from the csv files
        Return:
           A spark dataframe contains all the expected collumns from pop csv files
        """
        return self.pop_df_input

    def get_pop_counts(self):
        """ Return a total counnts of population group details
        Return:
           A total counts of population group details
        """
        return self.pop_counts


if __name__ == "__main__":
    h_reader = CEFHReader("safetab_h_cef_config_2010.ini", ['44'])


