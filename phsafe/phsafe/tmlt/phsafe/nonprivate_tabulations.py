"""Performs PH1, PH2, PH3, PH4, PH5, PH6, PH7, PH8 tabulations."""

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

import logging
from typing import Tuple

from pyspark.sql import DataFrame
from pyspark.sql.functions import array, col, explode, lit, when

from tmlt.phsafe import PHSafeInput, PHSafeOutput
from tmlt.phsafe.constants import (
    CPLT,
    HHT,
    HHT2,
    ITERATION_CODE,
    PH1_DATA_CELL,
    PH2_DATA_CELL,
    PH3_DATA_CELL,
    PH4_DATA_CELL,
    PH6_DATA_CELL,
    PH7_DATA_CELL,
    PH8_DATA_CELL,
    RELSHIP,
    RTYPE_PERSON,
    RTYPE_UNIT,
    TEN,
    EthCode,
    RaceCode,
)


class NonPrivateTabulations:
    """Performs non private tabulations on Person in Household data."""

    phg_join_df: DataFrame
    unit_geo: DataFrame

    def __call__(self, input_sdfs: PHSafeInput) -> PHSafeOutput:
        """Initialize Person Houselhold data by joining and perform tabulations.

        Args:
            input_sdfs: PHSafeInput NamedTuple containing persons, units and processed
                        geo dataframes.

        Returns:
            A PHSafeOutput object containing:
                PH1_num: Average household size by age (numerator).
                PH1_denom: Average household size by age (denominator).
                PH2: Household type for the population in households.
                PH3: Household type by relationship for population under 18 years.
                PH4: Population in families.
                PH5_num: Average family size by age (numerator).
                PH5_denom: Average family size by age (denominator).
                PH6: Family type and age for own children under 18 years.
                PH7: Total population in occupied housing units by tenure.
                PH8_num: Average household size of occupied
                         housing units by tenure (numerator).
                PH8_denom: Average household size of occupied
                            housing units by tenure (denominator).
            None (PrivateTabulations returns a Session)
        """
        logger = logging.getLogger(__name__)
        logger.info("Starting non-private algorithm execution...")
        geo_level_columns = set(input_sdfs.geo.columns) - {"MAFID"}
        stack_string_arg = ", ".join(
            f'"{column}", {column}' for column in geo_level_columns
        )
        stacked_geo_df = input_sdfs.geo.selectExpr(
            "MAFID",
            f"stack({len(geo_level_columns)}, {stack_string_arg}) "
            "as (REGION_TYPE, REGION_ID)",
        )
        self.phg_join_df = (
            input_sdfs.persons.withColumnRenamed("RTYPE", "RTYPE_PERSON")
            .join(input_sdfs.units.withColumnRenamed("RTYPE", "RTYPE_UNIT"), on="MAFID")
            .join(stacked_geo_df, on="MAFID")
        )
        self.unit_geo = input_sdfs.units.withColumnRenamed("RTYPE", "RTYPE_UNIT").join(
            stacked_geo_df, on="MAFID"
        )
        PH1_num, PH1_denom = self.ph1()
        PH5_num, PH5_denom = self.ph5()
        PH8_num, PH8_denom = self.ph8()
        return PHSafeOutput(
            PH1_num=PH1_num,
            PH1_denom=PH1_denom,
            PH2=self.ph2(),
            PH3=self.ph3(),
            PH4=self.ph4(),
            PH5_num=PH5_num,
            PH5_denom=PH5_denom,
            PH6=self.ph6(),
            PH7=self.ph7(),
            PH8_num=PH8_num,
            PH8_denom=PH8_denom,
        )

    def ph1(self) -> Tuple[DataFrame, DataFrame]:
        """Returns num and denom tables to calculate average household size by age."""
        PH1_peoples = (
            self.phg_join_df.where((col("RTYPE_UNIT") == "2") & (col("HHT") != "0"))
            .withColumn(
                "ITERATION_CODE",
                explode(
                    array(
                        when(
                            col("HHRACE") == RaceCode.WHITE_ALONE,
                            ITERATION_CODE.WHITE_ALONE.value,
                        )
                        .when(
                            col("HHRACE") == RaceCode.BLACK_ALONE,
                            ITERATION_CODE.BLACK_ALONE,
                        )
                        .when(
                            col("HHRACE") == RaceCode.AIAN_ALONE,
                            ITERATION_CODE.AIAN_ALONE.value,
                        )
                        .when(
                            col("HHRACE") == RaceCode.ASIAN_ALONE,
                            ITERATION_CODE.ASIAN_ALONE.value,
                        )
                        .when(
                            col("HHRACE") == RaceCode.NHPI_ALONE,
                            ITERATION_CODE.NHPI_ALONE,
                        )
                        .when(
                            col("HHRACE") == RaceCode.SOR_ALONE,
                            ITERATION_CODE.SOME_OTHER_RACE_ALONE.value,
                        )
                        .when(
                            ~col("HHRACE").isin(
                                RaceCode.WHITE_ALONE,
                                RaceCode.BLACK_ALONE,
                                RaceCode.AIAN_ALONE,
                                RaceCode.ASIAN_ALONE,
                                RaceCode.NHPI_ALONE,
                                RaceCode.SOR_ALONE,
                            ),
                            ITERATION_CODE.TWO_OR_MORE_RACES,
                        ),
                        when(
                            col("HHSPAN") == EthCode.HISPANIC,
                            ITERATION_CODE.HISPANIC_OR_LATINO,
                        ),
                        when(
                            (col("HHRACE") == RaceCode.WHITE_ALONE)
                            & (col("HHSPAN") == EthCode.NOT_HISPANIC),
                            ITERATION_CODE.WHITE_ALONE_NOT_HISPANIC,
                        ),
                        lit(ITERATION_CODE.UNATTRIBUTED),
                    )
                ),
            )
            .dropna()  # not every record is included in every iteration code level
            .withColumn(
                "PH1_NUM_DATA_CELL",
                explode(
                    # For first [condition] a record matches, assign to [table cells]
                    # [QAGE < 18]  ==>  [ 2, 1 ]
                    # [QAGE >= 18]  ==>  [ 3, 1 ]
                    when(
                        col("QAGE") < 18,
                        array(lit(PH1_DATA_CELL.UNDER_18), lit(PH1_DATA_CELL.TOTAL)),
                    ).when(
                        col("QAGE") >= 18,
                        array(
                            lit(PH1_DATA_CELL.AGE_18_AND_OVER), lit(PH1_DATA_CELL.TOTAL)
                        ),
                    )
                ),
            )
        )

        PH1_num = (
            PH1_peoples.groupby(
                "REGION_ID", "REGION_TYPE", "ITERATION_CODE", "PH1_NUM_DATA_CELL"
            )
            .count()
            .select(
                col("REGION_ID"),
                col("REGION_TYPE"),
                col("ITERATION_CODE"),
                col("PH1_NUM_DATA_CELL"),
                col("count").alias("COUNT"),
            )
        )

        PH1_units = (
            self.unit_geo.where(
                (col("RTYPE_UNIT") == RTYPE_UNIT.HOUSING_UNIT)
                & (col("HHT") != HHT.NOT_IN_UNIVERSE)
            )
            .withColumn(
                "ITERATION_CODE",
                explode(
                    array(
                        when(
                            col("HHRACE") == RaceCode.WHITE_ALONE,
                            ITERATION_CODE.WHITE_ALONE.value,
                        )
                        .when(
                            col("HHRACE") == RaceCode.BLACK_ALONE.value,
                            ITERATION_CODE.BLACK_ALONE.value,
                        )
                        .when(
                            col("HHRACE") == RaceCode.AIAN_ALONE,
                            ITERATION_CODE.AIAN_ALONE.value,
                        )
                        .when(
                            col("HHRACE") == RaceCode.ASIAN_ALONE,
                            ITERATION_CODE.ASIAN_ALONE.value,
                        )
                        .when(
                            col("HHRACE") == RaceCode.NHPI_ALONE,
                            ITERATION_CODE.NHPI_ALONE,
                        )
                        .when(
                            col("HHRACE") == RaceCode.SOR_ALONE,
                            ITERATION_CODE.SOME_OTHER_RACE_ALONE.value,
                        )
                        .when(
                            ~col("HHRACE").isin(
                                RaceCode.WHITE_ALONE,
                                RaceCode.BLACK_ALONE,
                                RaceCode.AIAN_ALONE,
                                RaceCode.ASIAN_ALONE,
                                RaceCode.NHPI_ALONE,
                                RaceCode.SOR_ALONE,
                            ),
                            ITERATION_CODE.TWO_OR_MORE_RACES,
                        ),
                        when(
                            col("HHSPAN") == EthCode.HISPANIC,
                            ITERATION_CODE.HISPANIC_OR_LATINO,
                        ),
                        when(
                            (col("HHRACE") == RaceCode.WHITE_ALONE)
                            & (col("HHSPAN") == EthCode.NOT_HISPANIC),
                            ITERATION_CODE.WHITE_ALONE_NOT_HISPANIC,
                        ),
                        lit(ITERATION_CODE.UNATTRIBUTED),
                    )
                ),
            )
            .dropna()  # not every record is included in every iteration code level
        )

        PH1_denom = (
            PH1_units.groupby("REGION_ID", "REGION_TYPE", "ITERATION_CODE")
            .count()
            .select(
                col("REGION_ID"),
                col("REGION_TYPE"),
                col("ITERATION_CODE"),
                col("count").alias("COUNT"),
            )
            .withColumn(
                "PH1_DENOM_DATA_CELL",
                explode(
                    array(
                        lit(PH1_DATA_CELL.UNDER_18),
                        lit(PH1_DATA_CELL.AGE_18_AND_OVER),
                        lit(PH1_DATA_CELL.TOTAL),
                    )
                ),
            )
            .select(
                "REGION_ID",
                "REGION_TYPE",
                "ITERATION_CODE",
                "PH1_DENOM_DATA_CELL",
                "COUNT",
            )
        )
        return PH1_num, PH1_denom

    def ph2(self) -> DataFrame:
        """Returns table with household type of population in households."""
        return (
            self.phg_join_df.where(col("RTYPE_PERSON") == RTYPE_PERSON.IN_HOUSING_UNIT)
            # Create a new record for each instance of a person in a data cell
            .withColumn(
                "PH2_DATA_CELL",
                # First add an array of data cells to each record, then explode.
                # For the first [condition] a record matches, assign to [table cells]
                explode(
                    when(
                        # [HHT2 in {01, 02} and CPLT = 1]  ==>  [ 3, 2, 1 ]
                        # [HHT2 in {01, 02} and CPLT = 2]  ==>  [ 4, 2, 1 ]
                        # [HHT2 in {01, 02}             ]  ==>  [ 2, 1 ]
                        col("HHT2").isin(
                            [
                                HHT2.MARRIED_COUPLE_WITH_OWN_CHILDREN,
                                HHT2.MARRIED_COUPLE_WITHOUT_OWN_CHILDREN,
                            ]
                        ),
                        when(
                            col("CPLT") == CPLT.OPPOSITE_SEX_MARRIED,
                            array(
                                lit(PH2_DATA_CELL.OPPOSITE_SEX_MARRIED_COUPLE),
                                lit(2),
                                lit(1),
                            ),
                        )
                        .when(
                            col("CPLT") == CPLT.SAME_SEX_MARRIED,
                            array(
                                lit(PH2_DATA_CELL.SAME_SEX_MARRIED_COUPLE),
                                lit(2),
                                lit(1),
                            ),
                        )
                        .otherwise(array(lit(2), lit(1))),
                    )
                    .when(
                        # [HHT2 in {03, 04} and CPLT = 3]  ==>  [ 6, 5, 1 ]
                        # [HHT2 in {03, 04} and CPLT = 4]  ==>  [ 7, 5, 1 ]
                        # [HHT2 in {03, 04}             ]  ==>  [ 5, 1 ]
                        col("HHT2").isin(
                            [
                                HHT2.COHABITING_COUPLE_WITH_OWN_CHILDREN,
                                HHT2.COHABITING_COUPLE_WITHOUT_OWN_CHILDREN,
                            ]
                        ),
                        (
                            when(
                                col("CPLT") == CPLT.OPPOSITE_SEX_UNMARRIED,
                                array(
                                    lit(PH2_DATA_CELL.OPPOSITE_SEX_COHABITING_COUPLE),
                                    lit(5),
                                    lit(1),
                                ),
                            )
                            .when(
                                col("CPLT") == CPLT.SAME_SEX_UNMARRIED,
                                array(
                                    lit(PH2_DATA_CELL.SAME_SEX_COHABITING_COUPLE),
                                    lit(5),
                                    lit(1),
                                ),
                            )
                            .otherwise(array(lit(5), lit(1)))
                        ),
                    )
                    .when(
                        # [HHT2 = 09]  ==>  [ 9, 8, 1]
                        col("HHT2") == HHT2.UNPARTNERED_MALE_HH_ALONE,
                        array(
                            lit(PH2_DATA_CELL.UNPARTNERED_MALE_HOUSEHOLDER_ALONE),
                            lit(8),
                            lit(1),
                        ),
                    )
                    .when(
                        # [HHT2 in {10, 11, 12}]  ==>  [ 10, 8, 1]
                        col("HHT2").isin(
                            [
                                HHT2.UNPARTNERED_MALE_HH_WITH_OWN_CHILDREN,
                                HHT2.UNPARTNERED_MALE_HH_WITH_ADULT_RELATIVES,
                                HHT2.UNPARTNERED_MALE_HH_ONLY_NONRELATIVES,
                            ]
                        ),
                        array(
                            lit(PH2_DATA_CELL.UNPARTNERED_MALE_HOUSEHOLDER_WITH_OTHERS),
                            lit(8),
                            lit(1),
                        ),
                    )
                    .when(
                        # [HHT2 = 05]  ==>  [ 12, 11, 1]
                        col("HHT2") == HHT2.UNPARTNERED_FEMALE_HH_ALONE,
                        array(
                            lit(PH2_DATA_CELL.UNPARTNERED_FEMALE_HOUSEHOLDER_ALONE),
                            lit(11),
                            lit(1),
                        ),
                    )
                    .when(
                        # [HHT2 in {06, 07, 08}]  ==>  [ 13, 11, 1]
                        col("HHT2").isin(
                            [
                                HHT2.UNPARTNERED_FEMALE_HH_WITH_OWN_CHILDREN,
                                HHT2.UNPARTNERED_FEMALE_HH_WITH_ADULT_RELATIVES,
                                HHT2.UNPARTNERED_FEMALE_HH_ONLY_NONRELATIVES,
                            ]
                        ),
                        array(
                            lit(
                                PH2_DATA_CELL.UNPARTNERED_FEMALE_HOUSEHOLDER_WITH_OTHERS
                            ),
                            lit(11),
                            lit(1),
                        ),
                    )
                    .otherwise(array(lit(1)))  # [True]  ==>  [1]
                ),
            )
            .groupby("REGION_ID", "REGION_TYPE", "PH2_DATA_CELL")
            .count()
            .select(
                col("REGION_ID"),
                col("REGION_TYPE"),
                col("PH2_DATA_CELL"),
                col("count").alias("COUNT"),
            )
        )

    def ph3(self) -> DataFrame:
        """Returns table with household type by relationship for population <18."""
        return (
            self.phg_join_df.where(
                (col("RTYPE_PERSON") == RTYPE_PERSON.IN_HOUSING_UNIT)
                & (col("QAGE") < 18)
            )
            .withColumn(
                "ITERATION_CODE",
                explode(
                    array(
                        when(
                            col("CENRACE") == RaceCode.WHITE_ALONE,
                            ITERATION_CODE.WHITE_ALONE,
                        )
                        .when(
                            col("CENRACE") == RaceCode.BLACK_ALONE,
                            ITERATION_CODE.BLACK_ALONE,
                        )
                        .when(
                            col("CENRACE") == RaceCode.AIAN_ALONE,
                            ITERATION_CODE.AIAN_ALONE.value,
                        )
                        .when(
                            col("CENRACE") == RaceCode.ASIAN_ALONE,
                            ITERATION_CODE.ASIAN_ALONE.value,
                        )
                        .when(
                            col("CENRACE") == RaceCode.NHPI_ALONE,
                            ITERATION_CODE.NHPI_ALONE,
                        )
                        .when(
                            col("CENRACE") == RaceCode.SOR_ALONE,
                            ITERATION_CODE.SOME_OTHER_RACE_ALONE.value,
                        )
                        .when(
                            ~col("CENRACE").isin(
                                RaceCode.WHITE_ALONE,
                                RaceCode.BLACK_ALONE,
                                RaceCode.AIAN_ALONE,
                                RaceCode.ASIAN_ALONE,
                                RaceCode.NHPI_ALONE,
                                RaceCode.SOR_ALONE,
                            ),
                            ITERATION_CODE.TWO_OR_MORE_RACES,
                        ),
                        when(
                            col("CENHISP") == EthCode.HISPANIC,
                            ITERATION_CODE.HISPANIC_OR_LATINO,
                        ),
                        when(
                            (col("CENHISP") == RaceCode.WHITE_ALONE)
                            & (col("CENRACE") == EthCode.NOT_HISPANIC),
                            ITERATION_CODE.WHITE_ALONE_NOT_HISPANIC,
                        ),
                        lit(ITERATION_CODE.UNATTRIBUTED),
                    )
                ),
            )
            .dropna()  # not every record is included in every iteration code level
            .withColumn(
                "PH3_DATA_CELL",
                # For the first [condition] a record matches, assign to [table cells]
                # [RTYPE(person) = 3 and RELSHIP in {20, 21, 22, 23, 24, 34, 35, 36}]
                #                                                 ==> [2, 1]
                # [RTYPE(person) = 3 and RELSHIP in {25, 26, 27} and HHT2 = 01]
                #                                                 ==> [4, 3, 1]
                # [RTYPE(person) = 3 and RELSHIP in {25, 26, 27} and HHT2 = 03]
                #                                                 ==> [5, 3, 1]
                # [RTYPE(person) = 3 and RELSHIP in {25, 26, 27} and HHT2 = 10]
                #                                                 ==> [6, 3, 1]
                # [RTYPE(person) = 3 and RELSHIP in {25, 26, 27} and HHT2 = 06]
                #                                                 ==> [7, 3, 1]
                # [RTYPE(person) = 3 and RELSHIP in {25, 26, 27}] ==> [3, 1]
                # [RTYPE(person) = 3 and RELSHIP = 30] ==> [9, 8, 1]
                # [RTYPE(person) = 3 and RELSHIP in {28, 29, 31, 32, 33}]
                #                                              ==> [10, 8, 1]
                # [RTYPE(person) = 3] ==> [1,]
                explode(
                    when(
                        col("RTYPE_PERSON") == RTYPE_PERSON.IN_HOUSING_UNIT,
                        when(
                            col("RELSHIP").isin(
                                RELSHIP.HOUSEHOLDER,
                                RELSHIP.OPPOSITE_SEX_SPOUSE,
                                RELSHIP.OPPOSITE_SEX_UNMARRIED,
                                RELSHIP.SAME_SEX_SPOUSE,
                                RELSHIP.SAME_SEX_UNMARRIED,
                                RELSHIP.ROOMMATE_OR_HOUSEMATE,
                                RELSHIP.FOSTER_CHILD,
                                RELSHIP.OTHER_NONRELATIVE,
                            ),
                            array(
                                lit(PH3_DATA_CELL.HOUSEHOLDER_PARTNER_OR_NONRELATIVE),
                                lit(1),
                            ),
                        )
                        .when(
                            col("RELSHIP").isin(
                                RELSHIP.BIOLOGICAL_CHILD,
                                RELSHIP.ADOPTED_CHILD,
                                RELSHIP.STEPCHILD,
                            ),
                            when(
                                col("HHT2") == HHT2.MARRIED_COUPLE_WITH_OWN_CHILDREN,
                                array(
                                    lit(PH3_DATA_CELL.OWN_CHILD_MARRIED_FAMILY),
                                    lit(3),
                                    lit(1),
                                ),
                            )
                            .when(
                                col("HHT2") == HHT2.COHABITING_COUPLE_WITH_OWN_CHILDREN,
                                array(
                                    lit(PH3_DATA_CELL.OWN_CHILD_COHABITING_FAMILY),
                                    lit(3),
                                    lit(1),
                                ),
                            )
                            .when(
                                col("HHT2")
                                == HHT2.UNPARTNERED_MALE_HH_WITH_OWN_CHILDREN,
                                array(
                                    lit(
                                        PH3_DATA_CELL.OWN_CHILD_UNPARTNERED_MALE_FAMILY
                                    ),
                                    lit(3),
                                    lit(1),
                                ),
                            )
                            .when(
                                col("HHT2")
                                == HHT2.UNPARTNERED_FEMALE_HH_WITH_OWN_CHILDREN,
                                array(
                                    lit(
                                        PH3_DATA_CELL.OWN_CHILD_UNPARTNERED_FEMALE_FAMILY  # pylint:disable=line-too-long
                                    ),
                                    lit(3),
                                    lit(1),
                                ),
                            )
                            .otherwise(array(lit(3), lit(1))),
                        )
                        .when(
                            col("RELSHIP") == RELSHIP.GRANDCHILD,
                            array(lit(PH3_DATA_CELL.GRANDCHILD), lit(8), lit(1)),
                        )
                        .when(
                            col("RELSHIP").isin(
                                RELSHIP.SIBLING,
                                RELSHIP.PARENT,
                                RELSHIP.GRANDCHILD,
                                RELSHIP.PARENT_IN_LAW,
                                RELSHIP.CHILD_IN_LAW,
                                RELSHIP.OTHER_RELATIVE,
                            ),
                            array(lit(PH3_DATA_CELL.OTHER_RELATIVES), lit(8), lit(1)),
                        )
                        .otherwise(array(lit(1))),
                    )
                ),
            )
            .groupby("REGION_ID", "REGION_TYPE", "ITERATION_CODE", "PH3_DATA_CELL")
            .count()
            .select(
                col("REGION_ID"),
                col("REGION_TYPE"),
                col("ITERATION_CODE"),
                col("PH3_DATA_CELL"),
                col("count").alias("COUNT"),
            )
        )

    def ph4(self) -> DataFrame:
        """Returns PH4 table."""
        return (
            self.phg_join_df.where(
                (
                    col("HHT").isin(
                        HHT.MARRIED_COUPLE_HOUSEHOLD,
                        HHT.OTHER_FAMILY_HOUSEHOLD_MALE_HOUSEHOLDER,
                        HHT.OTHER_FAMILY_HOUSEHOLD_FEMALE_HOUSEHOLDER,
                    )
                )
                & (
                    col("RELSHIP").isin(
                        RELSHIP.HOUSEHOLDER,
                        RELSHIP.OPPOSITE_SEX_SPOUSE,
                        RELSHIP.SAME_SEX_SPOUSE,
                        RELSHIP.BIOLOGICAL_CHILD,
                        RELSHIP.ADOPTED_CHILD,
                        RELSHIP.STEPCHILD,
                        RELSHIP.SIBLING,
                        RELSHIP.PARENT,
                        RELSHIP.GRANDCHILD,
                        RELSHIP.PARENT_IN_LAW,
                        RELSHIP.CHILD_IN_LAW,
                        RELSHIP.OTHER_RELATIVE,
                    )
                )
            )
            .withColumn(
                "ITERATION_CODE",
                explode(
                    array(
                        when(
                            col("HHRACE") == RaceCode.WHITE_ALONE,
                            ITERATION_CODE.WHITE_ALONE.value,
                        )
                        .when(
                            col("HHRACE") == RaceCode.BLACK_ALONE.value,
                            ITERATION_CODE.BLACK_ALONE,
                        )
                        .when(
                            col("HHRACE") == RaceCode.AIAN_ALONE,
                            ITERATION_CODE.AIAN_ALONE.value,
                        )
                        .when(
                            col("HHRACE") == RaceCode.ASIAN_ALONE,
                            ITERATION_CODE.ASIAN_ALONE.value,
                        )
                        .when(
                            col("HHRACE") == RaceCode.NHPI_ALONE,
                            ITERATION_CODE.NHPI_ALONE,
                        )
                        .when(
                            col("HHRACE") == RaceCode.SOR_ALONE,
                            ITERATION_CODE.SOME_OTHER_RACE_ALONE.value,
                        )
                        .when(
                            ~col("HHRACE").isin(
                                RaceCode.WHITE_ALONE,
                                RaceCode.BLACK_ALONE,
                                RaceCode.AIAN_ALONE,
                                RaceCode.ASIAN_ALONE,
                                RaceCode.NHPI_ALONE,
                                RaceCode.SOR_ALONE,
                            ),
                            ITERATION_CODE.TWO_OR_MORE_RACES,
                        ),
                        when(
                            col("HHSPAN") == EthCode.HISPANIC,
                            ITERATION_CODE.HISPANIC_OR_LATINO,
                        ),
                        when(
                            (col("HHSPAN") == EthCode.NOT_HISPANIC)
                            & (col("HHRACE") == RaceCode.WHITE_ALONE),
                            ITERATION_CODE.WHITE_ALONE_NOT_HISPANIC,
                        ),
                        lit(ITERATION_CODE.UNATTRIBUTED),
                    )
                ),
            )
            .dropna()  # not every record is included in every iteration code level
            .withColumn(
                "PH4_DATA_CELL",
                # For the first [condition] a record matches, assign to [table cells]
                # QAGE < 18 ==> [2, 1]
                # QAGE >= 18 ==> [3, 1]
                explode(
                    when(
                        col("QAGE") < 18, array(lit(PH4_DATA_CELL.UNDER_18), lit(1))
                    ).when(
                        col("QAGE") >= 18,
                        array(lit(PH4_DATA_CELL.AGE_18_AND_OVER), lit(1)),
                    )
                ),
            )
            .groupby("REGION_ID", "REGION_TYPE", "ITERATION_CODE", "PH4_DATA_CELL")
            .count()
            .select(
                col("REGION_ID"),
                col("REGION_TYPE"),
                col("ITERATION_CODE"),
                col("PH4_DATA_CELL"),
                col("count").alias("COUNT"),
            )
        )

    def ph5(self) -> Tuple[DataFrame, DataFrame]:
        """Returns PH5 output."""
        PH5_peoples = (
            self.phg_join_df.where(
                (
                    col("HHT").isin(
                        HHT.MARRIED_COUPLE_HOUSEHOLD,
                        HHT.OTHER_FAMILY_HOUSEHOLD_MALE_HOUSEHOLDER,
                        HHT.OTHER_FAMILY_HOUSEHOLD_FEMALE_HOUSEHOLDER,
                    )
                )
                & (
                    col("RELSHIP").isin(
                        RELSHIP.HOUSEHOLDER,
                        RELSHIP.OPPOSITE_SEX_SPOUSE,
                        RELSHIP.SAME_SEX_SPOUSE,
                        RELSHIP.BIOLOGICAL_CHILD,
                        RELSHIP.ADOPTED_CHILD,
                        RELSHIP.STEPCHILD,
                        RELSHIP.SIBLING,
                        RELSHIP.PARENT,
                        RELSHIP.GRANDCHILD,
                        RELSHIP.PARENT_IN_LAW,
                        RELSHIP.CHILD_IN_LAW,
                        RELSHIP.OTHER_RELATIVE,
                    )
                )
            )
            .withColumn(
                "ITERATION_CODE",
                explode(
                    array(
                        when(
                            col("HHRACE") == RaceCode.WHITE_ALONE,
                            ITERATION_CODE.WHITE_ALONE.value,
                        )
                        .when(
                            col("HHRACE") == RaceCode.BLACK_ALONE.value,
                            ITERATION_CODE.BLACK_ALONE,
                        )
                        .when(
                            col("HHRACE") == RaceCode.AIAN_ALONE,
                            ITERATION_CODE.AIAN_ALONE.value,
                        )
                        .when(
                            col("HHRACE") == RaceCode.ASIAN_ALONE,
                            ITERATION_CODE.ASIAN_ALONE.value,
                        )
                        .when(
                            col("HHRACE") == RaceCode.NHPI_ALONE,
                            ITERATION_CODE.NHPI_ALONE,
                        )
                        .when(
                            col("HHRACE") == RaceCode.SOR_ALONE,
                            ITERATION_CODE.SOME_OTHER_RACE_ALONE.value,
                        )
                        .when(
                            ~col("HHRACE").isin(
                                RaceCode.WHITE_ALONE,
                                RaceCode.BLACK_ALONE,
                                RaceCode.AIAN_ALONE,
                                RaceCode.ASIAN_ALONE,
                                RaceCode.NHPI_ALONE,
                                RaceCode.SOR_ALONE,
                            ),
                            ITERATION_CODE.TWO_OR_MORE_RACES,
                        ),
                        when(
                            col("HHSPAN") == EthCode.HISPANIC,
                            ITERATION_CODE.HISPANIC_OR_LATINO,
                        ),
                        when(
                            (col("HHRACE") == RaceCode.WHITE_ALONE)
                            & (col("HHSPAN") == EthCode.NOT_HISPANIC),
                            ITERATION_CODE.WHITE_ALONE_NOT_HISPANIC,
                        ),
                        lit(ITERATION_CODE.UNATTRIBUTED),
                    )
                ),
            )
            .dropna()  # not every record is included in every iteration code level
            .withColumn(
                "PH5_NUM_DATA_CELL",
                explode(
                    # For first [condition] a record matches, assign to [table cells]
                    # [QAGE < 18]  ==>  [ 2, 1 ]
                    # [QAGE >= 18]  ==>  [ 3, 1 ]
                    when(col("QAGE") < 18, array(lit(2), lit(1))).when(
                        col("QAGE") >= 18, array(lit(3), lit(1))
                    )
                ),
            )
        )

        PH5_num = (
            PH5_peoples.groupby(
                "REGION_ID", "REGION_TYPE", "ITERATION_CODE", "PH5_NUM_DATA_CELL"
            )
            .count()
            .select(
                col("REGION_ID"),
                col("REGION_TYPE"),
                col("ITERATION_CODE"),
                col("PH5_NUM_DATA_CELL"),
                col("count").alias("COUNT"),
            )
        )

        PH5_units = (
            self.unit_geo.where(
                (
                    col("HHT").isin(
                        HHT.MARRIED_COUPLE_HOUSEHOLD,
                        HHT.OTHER_FAMILY_HOUSEHOLD_MALE_HOUSEHOLDER,
                        HHT.OTHER_FAMILY_HOUSEHOLD_FEMALE_HOUSEHOLDER,
                    )
                )
            )
            .withColumn(
                "ITERATION_CODE",
                explode(
                    array(
                        when(
                            col("HHRACE") == RaceCode.WHITE_ALONE,
                            ITERATION_CODE.WHITE_ALONE.value,
                        )
                        .when(
                            col("HHRACE") == RaceCode.BLACK_ALONE.value,
                            ITERATION_CODE.BLACK_ALONE,
                        )
                        .when(
                            col("HHRACE") == RaceCode.AIAN_ALONE,
                            ITERATION_CODE.AIAN_ALONE.value,
                        )
                        .when(
                            col("HHRACE") == RaceCode.ASIAN_ALONE,
                            ITERATION_CODE.ASIAN_ALONE.value,
                        )
                        .when(
                            col("HHRACE") == RaceCode.NHPI_ALONE,
                            ITERATION_CODE.NHPI_ALONE,
                        )
                        .when(
                            col("HHRACE") == RaceCode.SOR_ALONE,
                            ITERATION_CODE.SOME_OTHER_RACE_ALONE.value,
                        )
                        .when(
                            ~col("HHRACE").isin(
                                RaceCode.WHITE_ALONE,
                                RaceCode.BLACK_ALONE,
                                RaceCode.AIAN_ALONE,
                                RaceCode.ASIAN_ALONE,
                                RaceCode.NHPI_ALONE,
                                RaceCode.SOR_ALONE,
                            ),
                            ITERATION_CODE.TWO_OR_MORE_RACES,
                        ),
                        when(
                            col("HHSPAN") == EthCode.HISPANIC,
                            ITERATION_CODE.HISPANIC_OR_LATINO,
                        ),
                        when(
                            (col("HHRACE") == RaceCode.WHITE_ALONE)
                            & (col("HHSPAN") == EthCode.NOT_HISPANIC),
                            ITERATION_CODE.WHITE_ALONE_NOT_HISPANIC,
                        ),
                        lit(ITERATION_CODE.UNATTRIBUTED),
                    )
                ),
            )
            .dropna()  # not every record is included in every iteration code level
        )

        PH5_denom = (
            PH5_units.groupby("REGION_ID", "REGION_TYPE", "ITERATION_CODE")
            .count()
            .select(
                col("REGION_ID"),
                col("REGION_TYPE"),
                col("ITERATION_CODE"),
                col("count").alias("COUNT"),
            )
            .withColumn("PH5_DENOM_DATA_CELL", explode(array(lit(1), lit(2), lit(3))))
            .select(
                "REGION_ID",
                "REGION_TYPE",
                "ITERATION_CODE",
                "PH5_DENOM_DATA_CELL",
                "COUNT",
            )
        )
        return PH5_num, PH5_denom

    def ph6(self) -> DataFrame:
        """Returns PH6 output."""
        return (
            self.phg_join_df.where(
                (
                    col("RELSHIP").isin(
                        RELSHIP.BIOLOGICAL_CHILD,
                        RELSHIP.ADOPTED_CHILD,
                        RELSHIP.STEPCHILD,
                    )
                )
                & (col("QAGE") < 18)
            )
            # Create a new record for each instance of a person in a data cell
            .withColumn(
                "PH6_DATA_CELL",
                # First add an array of data cells to each record, then explode.
                # For the first [condition] a record matches, assign to [table cells]
                explode(
                    when(
                        # [HHT2 == 01 and QAGE<4]  ==>  [ 3, 2, 1 ]
                        # [HHT2 in {01} and QAGE == 4 or 5]  ==>  [ 4, 2, 1 ]
                        # [HHT2 in {01} and QAGE in 6 to 11]  ==>  [ 5, 2, 1 ]
                        # [HHT2 in {01} and QAGE >=12] ==> [6, 2, 1]
                        col("HHT2").isin(HHT2.MARRIED_COUPLE_WITH_OWN_CHILDREN),
                        when(
                            col("QAGE") < 4,
                            array(
                                lit(PH6_DATA_CELL.MARRIED_FAMILY_UNDER_4),
                                lit(2),
                                lit(1),
                            ),
                        )
                        .when(
                            (col("QAGE") == 4) | (col("QAGE") == 5),
                            array(
                                lit(PH6_DATA_CELL.MARRIED_FAMILY_4_AND_5),
                                lit(2),
                                lit(1),
                            ),
                        )
                        .when(
                            (col("QAGE") >= 6) & (col("QAGE") <= 11),
                            array(
                                lit(PH6_DATA_CELL.MARRIED_FAMILY_6_TO_11),
                                lit(2),
                                lit(1),
                            ),
                        )
                        .when(
                            col("QAGE") >= 12,
                            array(
                                lit(PH6_DATA_CELL.MARRIED_FAMILY_12_TO_17),
                                lit(2),
                                lit(1),
                            ),
                        )
                        .otherwise(array(lit(2), lit(1))),
                    )
                    .when(
                        # [HHT2 in {03} and QAGE<4]  ==>  [ 8, 7, 1 ]
                        # [HHT2 in {03} and QAGE == 4 or 5]  ==>  [ 9, 7, 1 ]
                        # [HHT2 in {03} and QAGE in 6 to 11]  ==>  [ 10, 7, 1 ]
                        # [HHT2 in {03} and QAGE >=12] ==> [11, 7, 1]
                        col("HHT2").isin("03"),
                        when(col("QAGE") < 4, array(lit(8), lit(7), lit(1)))
                        .when(
                            (col("QAGE") == 4) | (col("QAGE") == 5),
                            array(lit(9), lit(7), lit(1)),
                        )
                        .when(
                            (col("QAGE") >= 6) & (col("QAGE") <= 11),
                            array(lit(10), lit(7), lit(1)),
                        )
                        .when(col("QAGE") >= 12, array(lit(11), lit(7), lit(1)))
                        .otherwise(array(lit(7), lit(1))),
                    )
                    .when(
                        # [HHT2 in {10} and QAGE<4]  ==>  [ 13, 12, 1 ]
                        # [HHT2 in {10} and QAGE == 4 or 5]  ==>  [ 14, 12, 1 ]
                        # [HHT2 in {10} and QAGE in 6 to 11]  ==>  [ 15, 12, 1 ]
                        # [HHT2 in {10} and QAGE >=12] ==> [16, 12, 1]
                        col("HHT2").isin(HHT2.UNPARTNERED_MALE_HH_WITH_OWN_CHILDREN),
                        when(
                            col("QAGE") < 4,
                            array(
                                lit(PH6_DATA_CELL.UNPARTNERED_MALE_FAMILY_UNDER_4),
                                lit(12),
                                lit(1),
                            ),
                        )
                        .when(
                            (col("QAGE") == 4) | (col("QAGE") == 5),
                            array(
                                lit(PH6_DATA_CELL.UNPARTNERED_MALE_FAMILY_4_AND_5),
                                lit(12),
                                lit(1),
                            ),
                        )
                        .when(
                            (col("QAGE") >= 6) & (col("QAGE") <= 11),
                            array(
                                lit(PH6_DATA_CELL.UNPARTNERED_MALE_FAMILY_6_TO_11),
                                lit(12),
                                lit(1),
                            ),
                        )
                        .when(col("QAGE") >= 12, array(lit(16), lit(12), lit(1)))
                        .otherwise(
                            array(
                                lit(PH6_DATA_CELL.UNPARTNERED_MALE_FAMILY_12_TO_17),
                                lit(1),
                            )
                        ),
                    )
                    .when(
                        # [HHT2 in {06} and QAGE<4]  ==>  [ 18, 17, 1 ]
                        # [HHT2 in {06} and QAGE == 4 or 5]  ==>  [ 19, 17, 1 ]
                        # [HHT2 in {06} and QAGE in 6 to 11]  ==>  [ 20, 17, 1 ]
                        # [HHT2 in {06} and QAGE >=12] ==> [21, 17, 1]
                        col("HHT2").isin(HHT2.UNPARTNERED_FEMALE_HH_WITH_OWN_CHILDREN),
                        when(
                            col("QAGE") < 4,
                            array(
                                lit(PH6_DATA_CELL.UNPARTNERED_FEMALE_FAMILY_UNDER_4),
                                lit(17),
                                lit(1),
                            ),
                        )
                        .when(
                            (col("QAGE") == 4) | (col("QAGE") == 5),
                            array(
                                lit(PH6_DATA_CELL.UNPARTNERED_FEMALE_FAMILY_4_AND_5),
                                lit(17),
                                lit(1),
                            ),
                        )
                        .when(
                            (col("QAGE") >= 6) & (col("QAGE") <= 11),
                            array(
                                lit(PH6_DATA_CELL.UNPARTNERED_FEMALE_FAMILY_6_TO_11),
                                lit(17),
                                lit(1),
                            ),
                        )
                        .when(col("QAGE") >= 12, array(lit(21), lit(17), lit(1)))
                        .otherwise(
                            array(
                                lit(PH6_DATA_CELL.UNPARTNERED_FEMALE_FAMILY_12_TO_17),
                                lit(1),
                            )
                        ),
                    )
                    .otherwise(array(lit(1)))  # [True]  ==>  [1]
                ),
            )
            .groupby("REGION_ID", "REGION_TYPE", "PH6_DATA_CELL")
            .count()
            .select(
                col("REGION_ID"),
                col("REGION_TYPE"),
                col("PH6_DATA_CELL"),
                col("count").alias("COUNT"),
            )
        )

    def ph7(self) -> DataFrame:
        """Returns PH7 table."""
        PH7_step1 = (
            self.phg_join_df.where(
                (col("RTYPE_PERSON") == RTYPE_PERSON.IN_HOUSING_UNIT)
            )
            .withColumn(
                "ITERATION_CODE",
                explode(
                    array(
                        when(
                            col("HHRACE") == RaceCode.WHITE_ALONE,
                            ITERATION_CODE.WHITE_ALONE.value,
                        )
                        .when(
                            col("HHRACE") == RaceCode.BLACK_ALONE.value,
                            ITERATION_CODE.BLACK_ALONE,
                        )
                        .when(
                            col("HHRACE") == RaceCode.AIAN_ALONE,
                            ITERATION_CODE.AIAN_ALONE.value,
                        )
                        .when(
                            col("HHRACE") == RaceCode.ASIAN_ALONE,
                            ITERATION_CODE.ASIAN_ALONE.value,
                        )
                        .when(
                            col("HHRACE") == RaceCode.NHPI_ALONE,
                            ITERATION_CODE.NHPI_ALONE,
                        )
                        .when(
                            col("HHRACE") == RaceCode.SOR_ALONE,
                            ITERATION_CODE.SOME_OTHER_RACE_ALONE.value,
                        )
                        .when(
                            ~col("HHRACE").isin(
                                RaceCode.WHITE_ALONE,
                                RaceCode.BLACK_ALONE,
                                RaceCode.AIAN_ALONE,
                                RaceCode.ASIAN_ALONE,
                                RaceCode.NHPI_ALONE,
                                RaceCode.SOR_ALONE,
                            ),
                            ITERATION_CODE.TWO_OR_MORE_RACES,
                        ),
                        when(
                            col("HHSPAN") == EthCode.HISPANIC,
                            ITERATION_CODE.HISPANIC_OR_LATINO,
                        ),
                        when(
                            (col("HHSPAN") == EthCode.NOT_HISPANIC)
                            & (col("HHRACE") == RaceCode.WHITE_ALONE),
                            ITERATION_CODE.WHITE_ALONE_NOT_HISPANIC,
                        ),
                        lit(ITERATION_CODE.UNATTRIBUTED),
                    )
                ),
            )
            .dropna()
        )

        # not every record is included in every iteration code level
        PH7_step2 = PH7_step1.withColumn(
            "PH7_DATA_CELL",
            # For the first [condition] a record matches, assign to [table cells]
            # [TEN in {1}] ==> [2, 1]
            # [TEN in {2}] ==> [3, 1]
            # [TEN in {3,4}] ==> [4, 1]
            explode(
                when(
                    col("TEN") == TEN.OWNED_WITH_A_MORTGAGE,
                    array(lit(PH7_DATA_CELL.MORTGAGE_OR_LOAN), lit(1)),
                )
                .when(
                    col("TEN") == TEN.OWNED_FREE_AND_CLEAR,
                    array(lit(PH7_DATA_CELL.FREE_AND_CLEAR), lit(1)),
                )
                .when(
                    col("TEN").isin(TEN.RENTED, TEN.OCCUPIED_WITHOUT_PAYMENT_OF_RENT),
                    array(lit(PH7_DATA_CELL.RENTER_OCCUPIED), lit(1)),
                )
            ),
        )
        return (
            PH7_step2.groupby(
                "REGION_ID", "REGION_TYPE", "ITERATION_CODE", "PH7_DATA_CELL"
            )
            .count()
            .select(
                col("REGION_ID"),
                col("REGION_TYPE"),
                col("ITERATION_CODE"),
                col("PH7_DATA_CELL"),
                col("count").alias("COUNT"),
            )
        )

    def ph8(self) -> Tuple[DataFrame, DataFrame]:
        """Returns PH8 output."""
        PH8_households = (
            self.unit_geo.where(
                (col("RTYPE_UNIT") == RTYPE_UNIT.HOUSING_UNIT)
                & (col("HHT") != HHT.NOT_IN_UNIVERSE)
            )
            .withColumn(
                "ITERATION_CODE",
                explode(
                    array(
                        when(
                            col("HHRACE") == RaceCode.WHITE_ALONE,
                            ITERATION_CODE.WHITE_ALONE.value,
                        )
                        .when(
                            col("HHRACE") == RaceCode.BLACK_ALONE.value,
                            ITERATION_CODE.BLACK_ALONE,
                        )
                        .when(
                            col("HHRACE") == RaceCode.AIAN_ALONE,
                            ITERATION_CODE.AIAN_ALONE.value,
                        )
                        .when(
                            col("HHRACE") == RaceCode.ASIAN_ALONE,
                            ITERATION_CODE.ASIAN_ALONE.value,
                        )
                        .when(
                            col("HHRACE") == RaceCode.NHPI_ALONE,
                            ITERATION_CODE.NHPI_ALONE,
                        )
                        .when(
                            col("HHRACE") == RaceCode.SOR_ALONE,
                            ITERATION_CODE.SOME_OTHER_RACE_ALONE.value,
                        )
                        .when(
                            ~col("HHRACE").isin(
                                RaceCode.WHITE_ALONE,
                                RaceCode.BLACK_ALONE,
                                RaceCode.AIAN_ALONE,
                                RaceCode.ASIAN_ALONE,
                                RaceCode.NHPI_ALONE,
                                RaceCode.SOR_ALONE,
                            ),
                            ITERATION_CODE.TWO_OR_MORE_RACES,
                        ),
                        when(
                            col("HHSPAN") == EthCode.HISPANIC,
                            ITERATION_CODE.HISPANIC_OR_LATINO,
                        ),
                        when(
                            (col("HHRACE") == RaceCode.WHITE_ALONE)
                            & (col("HHSPAN") == EthCode.NOT_HISPANIC),
                            ITERATION_CODE.WHITE_ALONE_NOT_HISPANIC,
                        ),
                        lit(ITERATION_CODE.UNATTRIBUTED),
                    )
                ),
            )
            .dropna()  # not every record is included in every iteration code level
            .withColumn(
                "PH8_NUM_DATA_CELL",
                explode(
                    # For the first [condition] a record matches,
                    # assign to [table cells]
                    # [TEN in {1, 2}] ==> [2, 1]
                    # [TEN in {3,4}] ==> [3, 1]
                    when(
                        col("TEN").isin(
                            TEN.OWNED_WITH_A_MORTGAGE, TEN.OWNED_FREE_AND_CLEAR
                        ),
                        array(lit(PH8_DATA_CELL.OWNER_OCCUPIED), lit(1)),
                    ).when(
                        col("TEN").isin(
                            TEN.RENTED, TEN.OCCUPIED_WITHOUT_PAYMENT_OF_RENT
                        ),
                        array(lit(PH8_DATA_CELL.RENTER_OCCUPIED), lit(1)),
                    )
                ),
            )
        )
        PH8_num = (
            PH8_households.groupby(
                "REGION_ID", "REGION_TYPE", "ITERATION_CODE", "PH8_NUM_DATA_CELL"
            )
            .sum("FINAL_POP")
            .select(
                col("REGION_ID"),
                col("REGION_TYPE"),
                col("ITERATION_CODE"),
                col("PH8_NUM_DATA_CELL"),
                col("sum(FINAL_POP)").alias("COUNT"),
            )
        )
        PH8_units = (
            self.unit_geo.where(
                (col("RTYPE_UNIT") == RTYPE_UNIT.HOUSING_UNIT)
                & (col("HHT") != HHT.NOT_IN_UNIVERSE)
            )
            .withColumn(
                "ITERATION_CODE",
                explode(
                    array(
                        when(
                            col("HHRACE") == RaceCode.WHITE_ALONE,
                            ITERATION_CODE.WHITE_ALONE.value,
                        )
                        .when(
                            col("HHRACE") == RaceCode.BLACK_ALONE.value,
                            ITERATION_CODE.BLACK_ALONE,
                        )
                        .when(
                            col("HHRACE") == RaceCode.AIAN_ALONE,
                            ITERATION_CODE.AIAN_ALONE.value,
                        )
                        .when(
                            col("HHRACE") == RaceCode.ASIAN_ALONE,
                            ITERATION_CODE.ASIAN_ALONE.value,
                        )
                        .when(
                            col("HHRACE") == RaceCode.NHPI_ALONE,
                            ITERATION_CODE.NHPI_ALONE,
                        )
                        .when(
                            col("HHRACE") == RaceCode.SOR_ALONE,
                            ITERATION_CODE.SOME_OTHER_RACE_ALONE.value,
                        )
                        .when(
                            ~col("HHRACE").isin(
                                RaceCode.WHITE_ALONE,
                                RaceCode.BLACK_ALONE,
                                RaceCode.AIAN_ALONE,
                                RaceCode.ASIAN_ALONE,
                                RaceCode.NHPI_ALONE,
                                RaceCode.SOR_ALONE,
                            ),
                            ITERATION_CODE.TWO_OR_MORE_RACES,
                        ),
                        when(
                            col("HHSPAN") == EthCode.HISPANIC,
                            ITERATION_CODE.HISPANIC_OR_LATINO,
                        ),
                        when(
                            (col("HHRACE") == RaceCode.WHITE_ALONE)
                            & (col("HHSPAN") == EthCode.NOT_HISPANIC),
                            ITERATION_CODE.WHITE_ALONE_NOT_HISPANIC,
                        ),
                        lit(ITERATION_CODE.UNATTRIBUTED),
                    )
                ),
            )
            .dropna()  # not every record is included in every iteration code level
            .withColumn(
                "PH8_DENOM_DATA_CELL",
                explode(
                    # For the first [condition] a record matches,
                    # assign to [table cells]
                    # [TEN in {1, 2}] ==> [2, 1]
                    # [TEN in {3,4}] ==> [3, 1]
                    when(
                        col("TEN").isin(
                            TEN.OWNED_WITH_A_MORTGAGE, TEN.OWNED_FREE_AND_CLEAR
                        ),
                        array(lit(PH8_DATA_CELL.OWNER_OCCUPIED), lit(1)),
                    ).when(
                        col("TEN").isin(
                            TEN.RENTED, TEN.OCCUPIED_WITHOUT_PAYMENT_OF_RENT
                        ),
                        array(lit(PH8_DATA_CELL.RENTER_OCCUPIED), lit(1)),
                    )
                ),
            )
        )

        PH8_denom = (
            PH8_units.groupby(
                "REGION_ID", "REGION_TYPE", "ITERATION_CODE", "PH8_DENOM_DATA_CELL"
            )
            .count()
            .select(
                col("REGION_ID"),
                col("REGION_TYPE"),
                col("ITERATION_CODE"),
                col("PH8_DENOM_DATA_CELL"),
                col("count").alias("COUNT"),
            )
        )
        return PH8_num, PH8_denom
