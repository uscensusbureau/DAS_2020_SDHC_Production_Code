"""Performs private PH1, PH2, PH3, PH4, PH5, PH6, PH7, PH8 tabulations."""

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
import functools
import itertools
import logging
from math import exp
from typing import Dict, List, Mapping, Tuple, Union

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, lit, when
from typing_extensions import Literal

from tmlt.analytics.keyset import KeySet
from tmlt.analytics.privacy_budget import PureDPBudget, RhoZCDPBudget
from tmlt.analytics.protected_change import AddMaxRows, AddOneRow
from tmlt.analytics.query_builder import ColumnType, QueryBuilder
from tmlt.analytics.query_expr import CountMechanism, QueryExpr
from tmlt.analytics.session import Session
from tmlt.analytics.truncation_strategy import TruncationStrategy
from tmlt.common.configuration import Row
from tmlt.phsafe import (
    TABULATION_OUTPUT_COLUMNS,
    TABULATIONS_KEY,
    PHSafeInput,
    PHSafeOutput,
)
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

ITERATION_LEVELS = ["A-G", "H,I", "*"]
"""All possible iteration levels."""

ITERATION_CODES = {
    "A-G": [
        ITERATION_CODE.WHITE_ALONE.value,
        ITERATION_CODE.BLACK_ALONE.value,
        ITERATION_CODE.AIAN_ALONE.value,
        ITERATION_CODE.ASIAN_ALONE.value,
        ITERATION_CODE.NHPI_ALONE.value,
        ITERATION_CODE.SOME_OTHER_RACE_ALONE.value,
        ITERATION_CODE.TWO_OR_MORE_RACES.value,
    ],
    "H,I": [
        ITERATION_CODE.HISPANIC_OR_LATINO.value,
        ITERATION_CODE.WHITE_ALONE_NOT_HISPANIC.value,
    ],
    "*": [ITERATION_CODE.UNATTRIBUTED.value],
}
"""All possible values for ITERATION_CODE by iteration level."""

# Note: These domains will become more complicated when the queries are adaptive.
# These are just the domains for the most granular possible workloads.
_PH2_map_domain = [value.value for name, value in PH2_DATA_CELL.__members__.items()]
"""All possible values for PH2_DATA_CELL"""

_PH3_map_domain = [value.value for name, value in PH3_DATA_CELL.__members__.items()]
"""All possible values for PH3_DATA_CELL"""

_PH1_map_domain = [PH1_DATA_CELL.UNDER_18.value, PH1_DATA_CELL.AGE_18_AND_OVER.value]
"""All possible values for PH1_NUM_DATA_CELL"""

_PH4_map_domain = [value.value for name, value in PH4_DATA_CELL.__members__.items()]
"""All possible values for PH4_DATA_CELL"""

_PH5_map_domain = [2, 3]
"""All possible values for PH5_NUM_DATA_CELL"""

_PH6_map_domain = [value.value for name, value in PH6_DATA_CELL.__members__.items()]
"""All possible values for PH6_DATA_CELL"""

_PH7_map_domain = [value.value for name, value in PH7_DATA_CELL.__members__.items()]
"""All possible values for PH7_DATA_CELL"""

_PH8_map_domain = [
    PH8_DATA_CELL.OWNER_OCCUPIED.value,
    PH8_DATA_CELL.RENTER_OCCUPIED.value,
]
"""All possible values for PH8_DATA_CELL"""


def _iteration_map(
    row: Row, race_column: str, hisp_column: str, iteration_level: str
) -> Row:
    """Return iteration as a row given an iteration_level.

    Args:
        row: The record to map to iterations.
        race_column: The column to use for major iterations, either CENRACE or HHRACE.
        hisp_column: The column to use for ethnicity iterations, either CENHISP or
            HHSPAN.
        iteration_level: Iteration level used.

    Returns:
        Each record can be mapped to three iterations:

        * One major iteration (A-G)
        * One ethnicity iteration (H/I)
        * Every record gets mapped to "*"

        Some rows do not map to an ethnicity iteration. For those rows we return 'N/A'.

        The created row contains one column named "ITERATION_CODE".
    """
    if iteration_level == "A-G":
        if row[race_column] == RaceCode.WHITE_ALONE:
            return {"ITERATION_CODE": ITERATION_CODE.WHITE_ALONE.value}
        elif row[race_column] == RaceCode.BLACK_ALONE:
            return {"ITERATION_CODE": ITERATION_CODE.BLACK_ALONE.value}
        elif row[race_column] == RaceCode.AIAN_ALONE:
            return {"ITERATION_CODE": ITERATION_CODE.AIAN_ALONE.value}
        elif row[race_column] == RaceCode.ASIAN_ALONE:
            return {"ITERATION_CODE": ITERATION_CODE.ASIAN_ALONE.value}
        elif row[race_column] == RaceCode.NHPI_ALONE:
            return {"ITERATION_CODE": ITERATION_CODE.NHPI_ALONE.value}
        elif row[race_column] == RaceCode.SOR_ALONE:
            return {"ITERATION_CODE": ITERATION_CODE.SOME_OTHER_RACE_ALONE.value}
        elif row[race_column] in {
            value.value
            for name, value in RaceCode.__members__.items()
            if value
            not in {
                RaceCode.WHITE_ALONE,
                RaceCode.BLACK_ALONE,
                RaceCode.AIAN_ALONE,
                RaceCode.ASIAN_ALONE,
                RaceCode.NHPI_ALONE,
                RaceCode.SOR_ALONE,
            }
        }:  # in "07" to "63"
            return {"ITERATION_CODE": ITERATION_CODE.TWO_OR_MORE_RACES.value}

    if iteration_level == "H,I":
        if row[hisp_column] == EthCode.HISPANIC:
            return {"ITERATION_CODE": ITERATION_CODE.HISPANIC_OR_LATINO.value}
        elif (
            row[race_column] == RaceCode.WHITE_ALONE
            and row[hisp_column] == EthCode.NOT_HISPANIC
        ):
            return {"ITERATION_CODE": ITERATION_CODE.WHITE_ALONE_NOT_HISPANIC.value}
        else:
            return {"ITERATION_CODE": "N/A"}

    if iteration_level == "*":
        return {"ITERATION_CODE": ITERATION_CODE.UNATTRIBUTED.value}

    raise AssertionError("This should be unreachable.")


def _ph2_map(row: Row) -> int:
    """Return the value of PH2_DATA_CELL for the given record.

    Args:
        row: The record to map.
    """
    if row["HHT2"] in {
        HHT2.MARRIED_COUPLE_WITH_OWN_CHILDREN,
        HHT2.MARRIED_COUPLE_WITHOUT_OWN_CHILDREN,
    }:
        if row["CPLT"] == CPLT.OPPOSITE_SEX_MARRIED:
            return PH2_DATA_CELL.OPPOSITE_SEX_MARRIED_COUPLE.value
        if row["CPLT"] == CPLT.SAME_SEX_MARRIED:
            return PH2_DATA_CELL.SAME_SEX_MARRIED_COUPLE.value
    if row["HHT2"] in {
        HHT2.COHABITING_COUPLE_WITH_OWN_CHILDREN,
        HHT2.COHABITING_COUPLE_WITHOUT_OWN_CHILDREN,
    }:
        if row["CPLT"] == CPLT.OPPOSITE_SEX_UNMARRIED:
            return PH2_DATA_CELL.OPPOSITE_SEX_COHABITING_COUPLE.value
        if row["CPLT"] == CPLT.SAME_SEX_UNMARRIED:
            return PH2_DATA_CELL.SAME_SEX_COHABITING_COUPLE.value
    if row["HHT2"] == HHT2.UNPARTNERED_MALE_HH_ALONE:
        return PH2_DATA_CELL.UNPARTNERED_MALE_HOUSEHOLDER_ALONE.value
    if row["HHT2"] in {
        HHT2.UNPARTNERED_MALE_HH_WITH_OWN_CHILDREN,
        HHT2.UNPARTNERED_MALE_HH_WITH_ADULT_RELATIVES,
        HHT2.UNPARTNERED_MALE_HH_ONLY_NONRELATIVES,
    }:
        return PH2_DATA_CELL.UNPARTNERED_MALE_HOUSEHOLDER_WITH_OTHERS.value
    if row["HHT2"] == HHT2.UNPARTNERED_FEMALE_HH_ALONE:
        return PH2_DATA_CELL.UNPARTNERED_FEMALE_HOUSEHOLDER_ALONE.value
    if row["HHT2"] in {
        HHT2.UNPARTNERED_FEMALE_HH_WITH_OWN_CHILDREN,
        HHT2.UNPARTNERED_FEMALE_HH_WITH_ADULT_RELATIVES,
        HHT2.UNPARTNERED_FEMALE_HH_ONLY_NONRELATIVES,
    }:
        return PH2_DATA_CELL.UNPARTNERED_FEMALE_HOUSEHOLDER_WITH_OTHERS.value
    raise ValueError(f"Invalid record: {row}")


def _ph3_map(row: Row) -> int:
    """Return the value of PH3_DATA_CELL for the given record.

    Args:
        row: The record to map.
    """
    # RTYPE == 3  (RTYPE from person)
    if row["RELSHIP"] in {
        RELSHIP.HOUSEHOLDER,
        RELSHIP.OPPOSITE_SEX_SPOUSE,
        RELSHIP.OPPOSITE_SEX_UNMARRIED,
        RELSHIP.SAME_SEX_SPOUSE,
        RELSHIP.SAME_SEX_UNMARRIED,
        RELSHIP.ROOMMATE_OR_HOUSEMATE,
        RELSHIP.FOSTER_CHILD,
        RELSHIP.OTHER_NONRELATIVE,
    }:
        return PH3_DATA_CELL.HOUSEHOLDER_PARTNER_OR_NONRELATIVE.value
    if row["RELSHIP"] in {
        RELSHIP.BIOLOGICAL_CHILD,
        RELSHIP.ADOPTED_CHILD,
        RELSHIP.STEPCHILD,
    }:
        if row["HHT2"] == HHT2.MARRIED_COUPLE_WITH_OWN_CHILDREN:
            return PH3_DATA_CELL.OWN_CHILD_MARRIED_FAMILY.value
        if row["HHT2"] == HHT2.COHABITING_COUPLE_WITH_OWN_CHILDREN:
            return PH3_DATA_CELL.OWN_CHILD_COHABITING_FAMILY.value
        if row["HHT2"] == HHT2.UNPARTNERED_MALE_HH_WITH_OWN_CHILDREN:
            return PH3_DATA_CELL.OWN_CHILD_UNPARTNERED_MALE_FAMILY.value
        if row["HHT2"] == HHT2.UNPARTNERED_FEMALE_HH_WITH_OWN_CHILDREN:
            return PH3_DATA_CELL.OWN_CHILD_UNPARTNERED_FEMALE_FAMILY.value
    if (
        row["RELSHIP"] == RELSHIP.GRANDCHILD.value
    ):  # Notice that 30 goes here and not to the next check
        return PH3_DATA_CELL.GRANDCHILD.value
    if row["RELSHIP"] in {
        RELSHIP.SIBLING,
        RELSHIP.PARENT,
        RELSHIP.PARENT_IN_LAW,
        RELSHIP.CHILD_IN_LAW,
        RELSHIP.OTHER_RELATIVE,
    }:
        return PH3_DATA_CELL.OTHER_RELATIVES.value
    raise ValueError(f"Invalid record: {row}")


def _ph1_map(row: Row) -> int:
    """Return the value of PH1_NUM_DATA_CELL/PH1_DENOM_DATA_CELL for the given record.

    Args:
        row: The record to map.
    """
    return (
        PH1_DATA_CELL.UNDER_18.value
        if row["QAGE"] < 18
        else PH1_DATA_CELL.AGE_18_AND_OVER.value
    )


def _ph4_map(row: Row) -> int:
    """Return the value of PH4_DATA_CELL for the given record.

    Args:
        row: The record to map.
    """
    return (
        PH4_DATA_CELL.UNDER_18.value
        if row["QAGE"] < 18
        else PH4_DATA_CELL.AGE_18_AND_OVER.value
    )


def _ph6_map(row: Row) -> int:
    """Return the value of PH6_DATA_CELL for the given record.

    Args:
        row: The record to map.
    """
    # RTYPE == 3  (RTYPE from person)
    if row["HHT2"] == HHT2.MARRIED_COUPLE_WITH_OWN_CHILDREN:
        if row["QAGE"] < 4:
            return PH6_DATA_CELL.MARRIED_FAMILY_UNDER_4.value
        if row["QAGE"] == 4 or row["QAGE"] == 5:
            return PH6_DATA_CELL.MARRIED_FAMILY_4_AND_5.value
        if row["QAGE"] >= 6 and row["QAGE"] <= 11:
            return PH6_DATA_CELL.MARRIED_FAMILY_6_TO_11.value
        if row["QAGE"] >= 12 and row["QAGE"] < 18:
            return PH6_DATA_CELL.MARRIED_FAMILY_12_TO_17.value
    if row["HHT2"] == HHT2.COHABITING_COUPLE_WITH_OWN_CHILDREN:
        if row["QAGE"] < 4:
            return PH6_DATA_CELL.COHABITING_FAMILY_UNDER_4.value
        if row["QAGE"] == 4 or row["QAGE"] == 5:
            return PH6_DATA_CELL.COHABITING_FAMILY_4_AND_5.value
        if row["QAGE"] >= 6 and row["QAGE"] <= 11:
            return PH6_DATA_CELL.COHABITING_FAMILY_6_TO_11.value
        if row["QAGE"] >= 12 and row["QAGE"] < 18:
            return PH6_DATA_CELL.COHABITING_FAMILY_12_TO_17.value
    if row["HHT2"] == HHT2.UNPARTNERED_MALE_HH_WITH_OWN_CHILDREN:
        if row["QAGE"] < 4:
            return PH6_DATA_CELL.UNPARTNERED_MALE_FAMILY_UNDER_4.value
        if row["QAGE"] == 4 or row["QAGE"] == 5:
            return PH6_DATA_CELL.UNPARTNERED_MALE_FAMILY_4_AND_5.value
        if row["QAGE"] >= 6 and row["QAGE"] <= 11:
            return PH6_DATA_CELL.UNPARTNERED_MALE_FAMILY_6_TO_11.value
        if row["QAGE"] >= 12 and row["QAGE"] < 18:
            return PH6_DATA_CELL.UNPARTNERED_MALE_FAMILY_12_TO_17.value
    if row["HHT2"] == HHT2.UNPARTNERED_FEMALE_HH_WITH_OWN_CHILDREN:
        if row["QAGE"] < 4:
            return PH6_DATA_CELL.UNPARTNERED_FEMALE_FAMILY_UNDER_4.value
        if row["QAGE"] == 4 or row["QAGE"] == 5:
            return PH6_DATA_CELL.UNPARTNERED_FEMALE_FAMILY_4_AND_5.value
        if row["QAGE"] >= 6 and row["QAGE"] <= 11:
            return PH6_DATA_CELL.UNPARTNERED_FEMALE_FAMILY_6_TO_11.value
        if row["QAGE"] >= 12 and row["QAGE"] < 18:
            return PH6_DATA_CELL.UNPARTNERED_FEMALE_FAMILY_12_TO_17.value
    raise ValueError(f"Invalid record: {row}")


def _ph7_map(row: Row) -> int:
    """Return the value of PH7_DATA_CELL for the given record.

    Args:
        row: The record to map.
    """
    if row["TEN"] == TEN.OWNED_WITH_A_MORTGAGE:
        return PH7_DATA_CELL.MORTGAGE_OR_LOAN.value
    if row["TEN"] == TEN.OWNED_FREE_AND_CLEAR:
        return PH7_DATA_CELL.FREE_AND_CLEAR.value
    if row["TEN"] == TEN.RENTED or row["TEN"] == TEN.OCCUPIED_WITHOUT_PAYMENT_OF_RENT:
        return PH7_DATA_CELL.RENTER_OCCUPIED.value
    raise ValueError(f"Invalid record: {row}")


def _ph8_map(row: Row) -> int:
    """Return the value of PH8_DATA_CELL for the given record.

    Args:
        row: The record to map.
    """
    if (
        row["TEN"] == TEN.OWNED_WITH_A_MORTGAGE
        or row["TEN"] == TEN.OWNED_FREE_AND_CLEAR.value
    ):
        return PH8_DATA_CELL.OWNER_OCCUPIED.value
    if row["TEN"] == TEN.RENTED or row["TEN"] == TEN.OCCUPIED_WITHOUT_PAYMENT_OF_RENT:
        return PH8_DATA_CELL.RENTER_OCCUPIED.value
    raise ValueError(f"Invalid record: {row}")


def _create_geo_domains(processed_geo_df: DataFrame) -> Dict[str, List[str]]:
    """Return the domain for each region.

    Args:
        processed_geo_df: The processed geo table.
    """
    geo_domains = {
        column: [row[0] for row in processed_geo_df.select(column).distinct().collect()]
        for column in set(processed_geo_df.columns) - {"MAFID"}
    }
    return geo_domains


def _create_ph2_queries(
    tau: int,
    geo_domains: Mapping[str, List[str]],
    session: Session,
    noise_mechanism: CountMechanism,
) -> Mapping[Tuple[str, str], QueryExpr]:
    """Return the PH2 query for each region and iteration level (*).

    Args:
        tau: The maximum number of people per household.
        geo_domains: The domain for each region.
        session: The session being used to evaluate queries.
        noise_mechanism: Noise distribution for the PH2 count() queries.
    """
    PH2_builder = (
        QueryBuilder("persons")
        .rename({"RTYPE": "RTYPE_PERSON"})
        .filter(f"RTYPE_PERSON == '{RTYPE_PERSON.IN_HOUSING_UNIT}'")
        .join_private(
            QueryBuilder("unit_and_geo"),
            join_columns=["MAFID"],
            truncation_strategy_left=TruncationStrategy.DropExcess(tau),
            truncation_strategy_right=TruncationStrategy.DropNonUnique(),
        )
        .map(
            lambda row: {"PH2_DATA_CELL": _ph2_map(row)},
            new_column_types={"PH2_DATA_CELL": ColumnType.INTEGER},
            augment=True,
        )
        .select(list(geo_domains.keys()) + ["PH2_DATA_CELL"])
    )
    session.create_view(PH2_builder, source_id="PH2_root", cache=True)
    return {
        (region_type, "*"): QueryBuilder("PH2_root")
        .groupby(
            KeySet.from_dict(
                {region_type: region_domain, "PH2_DATA_CELL": _PH2_map_domain}
            )
        )
        .count(name="COUNT", mechanism=noise_mechanism)
        for region_type, region_domain in geo_domains.items()
        if region_domain
    }


def _create_ph3_queries(
    tau: int,
    geo_domains: Mapping[str, List[str]],
    session: Session,
    noise_mechanism: CountMechanism,
) -> Mapping[Tuple[str, str], QueryExpr]:
    """Return the PH3 query for each region and iteration level.

    Args:
        tau: The maximum number of people per household.
        geo_domains: The domain for each region.
        session: The session being used to evaluate queries.
        noise_mechanism: Noise distribution for the PH3 count() queries.
    """
    _iteration_map_dict = {
        "A-G": lambda row: _iteration_map(row, "CENRACE", "CENHISP", "A-G"),
        "H,I": lambda row: _iteration_map(row, "CENRACE", "CENHISP", "H,I"),
        "*": lambda row: _iteration_map(row, "CENRACE", "CENHISP", "*"),
    }
    PH3_builder = (
        QueryBuilder("persons")
        .rename({"RTYPE": "RTYPE_PERSON"})
        .filter(f"RTYPE_PERSON == '{RTYPE_PERSON.IN_HOUSING_UNIT}' and QAGE < 18")
        .join_private(
            QueryBuilder("unit_and_geo"),
            join_columns=["MAFID"],
            truncation_strategy_left=TruncationStrategy.DropExcess(tau),
            truncation_strategy_right=TruncationStrategy.DropNonUnique(),
        )
        .map(
            lambda row: {"PH3_DATA_CELL": _ph3_map(row)},
            new_column_types={"PH3_DATA_CELL": ColumnType.INTEGER},
            augment=True,
        )
        .select(list(geo_domains.keys()) + ["CENRACE", "CENHISP", "PH3_DATA_CELL"])
    )
    session.create_view(PH3_builder, source_id="PH3_root", cache=True)
    return {  # pylint: disable=cell-var-from-loop
        (region_type, iteration_level): QueryBuilder("PH3_root")
        .map(
            _iteration_map_dict[iteration_level],
            new_column_types={"ITERATION_CODE": ColumnType.VARCHAR},
            augment=True,
        )
        .filter(
            "ITERATION_CODE != 'N/A'"
        )  # Added for clarity. The KeySet will also filter out these rows.
        .groupby(
            KeySet.from_dict(
                {
                    region_type: region_domain,
                    "ITERATION_CODE": ITERATION_CODES[iteration_level],
                    "PH3_DATA_CELL": _PH3_map_domain,
                }
            )
        )
        .count(name="COUNT", mechanism=noise_mechanism)
        for (region_type, region_domain), iteration_level in itertools.product(
            geo_domains.items(), ITERATION_LEVELS
        )
        if region_domain
    }


def _create_ph1_num_queries(
    tau: int,
    geo_domains: Mapping[str, List[str]],
    session: Session,
    noise_mechanism: CountMechanism,
) -> Mapping[Tuple[str, str], QueryExpr]:
    """Return the PH1 numerator query for each region.

    Args:
        tau: The maximum number of people per household.
        geo_domains: The domain for each region.
        session: The session being used to evaluate queries.
        noise_mechanism: Noise distribution for the PH1 count() queries.
    """
    _iteration_map_dict = {
        "A-G": lambda row: _iteration_map(row, "HHRACE", "HHSPAN", "A-G"),
        "H,I": lambda row: _iteration_map(row, "HHRACE", "HHSPAN", "H,I"),
        "*": lambda row: _iteration_map(row, "HHRACE", "HHSPAN", "*"),
    }

    num_builder = (
        QueryBuilder("persons")
        .rename({"RTYPE": "RTYPE_PERSON"})
        .filter(f"RTYPE_PERSON == '{RTYPE_PERSON.IN_HOUSING_UNIT}'")
        .join_private(
            QueryBuilder("unit_and_geo"),
            join_columns=["MAFID"],
            truncation_strategy_left=TruncationStrategy.DropExcess(tau),
            truncation_strategy_right=TruncationStrategy.DropNonUnique(),
        )
        .map(
            lambda row: {"PH1_NUM_DATA_CELL": _ph1_map(row)},
            new_column_types={"PH1_NUM_DATA_CELL": ColumnType.INTEGER},
            augment=True,
        )
        .select(list(geo_domains.keys()) + ["HHRACE", "HHSPAN", "PH1_NUM_DATA_CELL"])
    )
    session.create_view(num_builder, source_id="PH1_num_root", cache=True)
    numerator_queries: Dict = dict()
    for (region_type, region_domain), iteration_level in itertools.product(
        geo_domains.items(), ITERATION_LEVELS
    ):  # pylint: disable=cell-var-from-loop, undefined-loop-variable
        if region_domain:
            region_population_groups = (
                QueryBuilder("PH1_num_root")
                .map(
                    _iteration_map_dict[iteration_level],
                    new_column_types={"ITERATION_CODE": ColumnType.VARCHAR},
                    augment=True,
                )
                .filter(
                    "ITERATION_CODE != 'N/A'"
                )  # Added for clarity. The KeySet will also filter out these rows.
                .groupby(
                    KeySet.from_dict(
                        {
                            region_type: region_domain,
                            "ITERATION_CODE": ITERATION_CODES[iteration_level],
                            "PH1_NUM_DATA_CELL": _PH1_map_domain,
                        }
                    )
                )
            )
            numerator_queries[
                (region_type, iteration_level)
            ] = region_population_groups.count(name="COUNT", mechanism=noise_mechanism)
    return numerator_queries


def _create_ph1_denom_queries(
    geo_domains: Mapping[str, List[str]],
    session: Session,
    noise_mechanism: CountMechanism,
) -> Mapping[Tuple[str, str], QueryExpr]:
    """Return the PH1 denominator query for each region.

    Args:
        geo_domains: The domain for each region.
        session: The session being used to evaluate queries.
        noise_mechanism: Noise distribution for the PH1 count() queries.
    """
    _iteration_map_dict = {
        "A-G": lambda row: _iteration_map(row, "HHRACE", "HHSPAN", "A-G"),
        "H,I": lambda row: _iteration_map(row, "HHRACE", "HHSPAN", "H,I"),
        "*": lambda row: _iteration_map(row, "HHRACE", "HHSPAN", "*"),
    }

    denom_builder = (
        QueryBuilder("unit_and_geo")
        .filter(
            f"RTYPE_UNIT == '{RTYPE_UNIT.HOUSING_UNIT}' and HHT !="
            f" '{HHT.NOT_IN_UNIVERSE}'"
        )
        .map(
            lambda row: {"PH1_DENOM_DATA_CELL": 1},
            new_column_types={"PH1_DENOM_DATA_CELL": ColumnType.INTEGER},
            augment=True,
        )
        .select(list(geo_domains.keys()) + ["HHRACE", "HHSPAN", "PH1_DENOM_DATA_CELL"])
    )
    session.create_view(denom_builder, source_id="PH1_denom_root", cache=True)
    denominator_queries: Dict = dict()
    for (region_type, region_domain), iteration_level in itertools.product(
        geo_domains.items(), ITERATION_LEVELS
    ):
        if region_domain:
            region_population_groups = (
                QueryBuilder("PH1_denom_root")
                .map(
                    _iteration_map_dict[iteration_level],
                    new_column_types={"ITERATION_CODE": ColumnType.VARCHAR},
                    augment=True,
                )
                .filter(
                    "ITERATION_CODE != 'N/A'"
                )  # Added for clarity. The KeySet will also filter out these rows.
                .groupby(
                    KeySet.from_dict(
                        {
                            region_type: region_domain,
                            "ITERATION_CODE": ITERATION_CODES[iteration_level],
                            "PH1_DENOM_DATA_CELL": [1],
                        }
                    )
                )
            )
            denominator_queries[
                (region_type, iteration_level)
            ] = region_population_groups.count(name="COUNT", mechanism=noise_mechanism)
    return denominator_queries


def _create_ph4_queries(
    tau: int,
    geo_domains: Mapping[str, List[str]],
    session: Session,
    noise_mechanism: CountMechanism,
) -> Mapping[Tuple[str, str], QueryExpr]:
    """Return the PH4 query for each region and iteration level.

    Args:
        tau: The maximum number of people per household.
        geo_domains: The domain for each region.
        session: The session being used to evaluate queries.
        noise_mechanism: Noise distribution for the PH4 count() queries.
    """
    _iteration_map_dict = {
        "A-G": lambda row: _iteration_map(row, "HHRACE", "HHSPAN", "A-G"),
        "H,I": lambda row: _iteration_map(row, "HHRACE", "HHSPAN", "H,I"),
        "*": lambda row: _iteration_map(row, "HHRACE", "HHSPAN", "*"),
    }

    PH4_builder = (
        QueryBuilder("persons")
        .rename({"RTYPE": "RTYPE_PERSON"})
        .filter(
            f"(RELSHIP in ('{RELSHIP.HOUSEHOLDER}', '{RELSHIP.OPPOSITE_SEX_SPOUSE}',"
            f" '{RELSHIP.SAME_SEX_SPOUSE}', '{RELSHIP.BIOLOGICAL_CHILD}',"
            f" '{RELSHIP.ADOPTED_CHILD}', '{RELSHIP.STEPCHILD}', '{RELSHIP.SIBLING}',"
            f" '{RELSHIP.PARENT}', '{RELSHIP.GRANDCHILD}', '{RELSHIP.PARENT_IN_LAW}',"
            f" '{RELSHIP.CHILD_IN_LAW}', '{RELSHIP.OTHER_RELATIVE}'))"
        )
        .join_private(
            QueryBuilder("unit_and_geo").filter("HHT <= '3'"),
            join_columns=["MAFID"],
            truncation_strategy_left=TruncationStrategy.DropExcess(tau),
            truncation_strategy_right=TruncationStrategy.DropNonUnique(),
        )
        .map(
            lambda row: {"PH4_DATA_CELL": _ph4_map(row)},
            new_column_types={"PH4_DATA_CELL": ColumnType.INTEGER},
            augment=True,
        )
        .select(list(geo_domains.keys()) + ["HHRACE", "HHSPAN", "PH4_DATA_CELL"])
    )
    session.create_view(PH4_builder, source_id="PH4_root", cache=True)
    return {  # pylint: disable=cell-var-from-loop
        (region_type, iteration_level): QueryBuilder("PH4_root")
        .map(
            _iteration_map_dict[iteration_level],
            new_column_types={"ITERATION_CODE": ColumnType.VARCHAR},
            augment=True,
        )
        .filter(
            "ITERATION_CODE != 'N/A'"
        )  # Added for clarity. The KeySet will also filter out these rows.
        .groupby(
            KeySet.from_dict(
                {
                    region_type: region_domain,
                    "ITERATION_CODE": ITERATION_CODES[iteration_level],
                    "PH4_DATA_CELL": _PH4_map_domain,
                }
            )
        )
        .count(name="COUNT", mechanism=noise_mechanism)
        for (region_type, region_domain), iteration_level in itertools.product(
            geo_domains.items(), ITERATION_LEVELS
        )
        if region_domain
    }


def _create_ph5_denom_queries(
    geo_domains: Mapping[str, List[str]],
    session: Session,
    noise_mechanism: CountMechanism,
) -> Mapping[Tuple[str, str], QueryExpr]:
    """Return the PH5 denominator query for each region.

    Args:
        geo_domains: The domain for each region.
        session: The session being used to evaluate queries.
        noise_mechanism: Noise distribution for the PH5 count() queries.
    """
    _iteration_map_dict = {
        "A-G": lambda row: _iteration_map(row, "HHRACE", "HHSPAN", "A-G"),
        "H,I": lambda row: _iteration_map(row, "HHRACE", "HHSPAN", "H,I"),
        "*": lambda row: _iteration_map(row, "HHRACE", "HHSPAN", "*"),
    }

    denom_builder = (
        QueryBuilder("unit_and_geo")
        .filter(
            f"HHT == '{HHT.MARRIED_COUPLE_HOUSEHOLD}' or HHT =="
            f" '{HHT.OTHER_FAMILY_HOUSEHOLD_MALE_HOUSEHOLDER}' or HHT =="
            f" '{HHT.OTHER_FAMILY_HOUSEHOLD_FEMALE_HOUSEHOLDER}'"
        )
        .map(
            lambda row: {"PH5_DENOM_DATA_CELL": 1},
            new_column_types={"PH5_DENOM_DATA_CELL": ColumnType.INTEGER},
            augment=True,
        )
        .select(list(geo_domains.keys()) + ["HHRACE", "HHSPAN", "PH5_DENOM_DATA_CELL"])
    )
    session.create_view(denom_builder, source_id="PH5_denom_root", cache=True)
    denominator_queries: Dict = dict()
    for (region_type, region_domain), iteration_level in itertools.product(
        geo_domains.items(), ITERATION_LEVELS
    ):
        if region_domain:
            region_population_groups = (
                QueryBuilder("PH5_denom_root")
                .map(
                    _iteration_map_dict[iteration_level],
                    new_column_types={"ITERATION_CODE": ColumnType.VARCHAR},
                    augment=True,
                )
                .filter(
                    "ITERATION_CODE != 'N/A'"
                )  # Added for clarity. The KeySet will also filter out these rows.
                .groupby(
                    KeySet.from_dict(
                        {
                            region_type: region_domain,
                            "ITERATION_CODE": ITERATION_CODES[iteration_level],
                            "PH5_DENOM_DATA_CELL": [1],
                        }
                    )
                )
            )
            denominator_queries[
                (region_type, iteration_level)
            ] = region_population_groups.count(name="COUNT", mechanism=noise_mechanism)
    return denominator_queries


def _create_ph6_queries(
    tau: int,
    geo_domains: Mapping[str, List[str]],
    session: Session,
    noise_mechanism: CountMechanism,
) -> Mapping[Tuple[str, str], QueryExpr]:
    """Return the PH6 query for each region and iteration level.

    Args:
        tau: The maximum number of people per household.
        geo_domains: The domain for each region.
        session: The session being used to evaluate queries.
        noise_mechanism: Noise distribution for the PH6 count() queries.
    """
    PH6_builder = (
        QueryBuilder("persons")
        .rename({"RTYPE": "RTYPE_PERSON"})
        .filter(
            f"QAGE < 18 and (RELSHIP == '{RELSHIP.BIOLOGICAL_CHILD}' or RELSHIP =="
            f" '{RELSHIP.ADOPTED_CHILD}' or RELSHIP == '{RELSHIP.STEPCHILD}')"
        )
        .join_private(
            QueryBuilder("unit_and_geo"),
            join_columns=["MAFID"],
            truncation_strategy_left=TruncationStrategy.DropExcess(tau),
            truncation_strategy_right=TruncationStrategy.DropNonUnique(),
        )
        .map(
            lambda row: {"PH6_DATA_CELL": _ph6_map(row)},
            new_column_types={"PH6_DATA_CELL": ColumnType.INTEGER},
            augment=True,
        )
        .select(list(geo_domains.keys()) + ["PH6_DATA_CELL"])
    )
    session.create_view(PH6_builder, source_id="PH6_root", cache=True)
    return {
        (region_type, "*"): QueryBuilder("PH6_root")
        .groupby(
            KeySet.from_dict(
                {region_type: region_domain, "PH6_DATA_CELL": _PH6_map_domain}
            )
        )
        .count(name="COUNT", mechanism=noise_mechanism)
        for region_type, region_domain in geo_domains.items()
        if region_domain
    }


def _create_ph7_queries(
    tau: int,
    geo_domains: Mapping[str, List[str]],
    session: Session,
    noise_mechanism: CountMechanism,
) -> Mapping[Tuple[str, str], QueryExpr]:
    """Return the PH7 query for each region and iteration level.

    Args:
        tau: The maximum number of people per household.
        geo_domains: The domain for each region.
        session: The session being used to evaluate queries.
        noise_mechanism: Noise distribution for the PH7 count() queries.
    """
    _iteration_map_dict = {
        "A-G": lambda row: _iteration_map(row, "HHRACE", "HHSPAN", "A-G"),
        "H,I": lambda row: _iteration_map(row, "HHRACE", "HHSPAN", "H,I"),
        "*": lambda row: _iteration_map(row, "HHRACE", "HHSPAN", "*"),
    }

    PH7_builder = (
        QueryBuilder("persons")
        .rename({"RTYPE": "RTYPE_PERSON"})
        .filter(f"RTYPE_PERSON == '{RTYPE_PERSON.IN_HOUSING_UNIT}'")
        .join_private(
            QueryBuilder("unit_and_geo"),
            join_columns=["MAFID"],
            truncation_strategy_left=TruncationStrategy.DropExcess(tau),
            truncation_strategy_right=TruncationStrategy.DropNonUnique(),
        )
        .map(
            lambda row: {"PH7_DATA_CELL": _ph7_map(row)},
            new_column_types={"PH7_DATA_CELL": ColumnType.INTEGER},
            augment=True,
        )
        .select(list(geo_domains.keys()) + ["HHRACE", "HHSPAN", "PH7_DATA_CELL"])
    )
    session.create_view(PH7_builder, source_id="PH7_root", cache=True)
    return {  # pylint: disable=cell-var-from-loop
        (region_type, iteration_level): QueryBuilder("PH7_root")
        .map(
            _iteration_map_dict[iteration_level],
            new_column_types={"ITERATION_CODE": ColumnType.VARCHAR},
            augment=True,
        )
        .filter(
            "ITERATION_CODE != 'N/A'"
        )  # Added for clarity. The KeySet will also filter out these rows.
        .groupby(
            KeySet.from_dict(
                {
                    region_type: region_domain,
                    "ITERATION_CODE": ITERATION_CODES[iteration_level],
                    "PH7_DATA_CELL": _PH7_map_domain,
                }
            )
        )
        .count(name="COUNT", mechanism=noise_mechanism)
        for (region_type, region_domain), iteration_level in itertools.product(
            geo_domains.items(), ITERATION_LEVELS
        )
        if region_domain
    }


def _create_ph8_denom_queries(
    geo_domains: Mapping[str, List[str]],
    session: Session,
    noise_mechanism: CountMechanism,
) -> Mapping[Tuple[str, str], QueryExpr]:
    """Return the PH8 denominator query for each region.

    Args:
        geo_domains: The domain for each region.
        session: The session being used to evaluate queries.
        noise_mechanism: Noise distribution for the PH8 count() queries.
    """
    _iteration_map_dict = {
        "A-G": lambda row: _iteration_map(row, "HHRACE", "HHSPAN", "A-G"),
        "H,I": lambda row: _iteration_map(row, "HHRACE", "HHSPAN", "H,I"),
        "*": lambda row: _iteration_map(row, "HHRACE", "HHSPAN", "*"),
    }

    denom_builder = (
        QueryBuilder("unit_and_geo")
        .filter(
            f"RTYPE_UNIT == '{RTYPE_UNIT.HOUSING_UNIT}' and HHT !="
            f" '{HHT.NOT_IN_UNIVERSE}'"
        )
        .map(
            lambda row: {"PH8_DENOM_DATA_CELL": _ph8_map(row)},
            new_column_types={"PH8_DENOM_DATA_CELL": ColumnType.INTEGER},
            augment=True,
        )
        .select(list(geo_domains.keys()) + ["HHRACE", "HHSPAN", "PH8_DENOM_DATA_CELL"])
    )
    session.create_view(denom_builder, source_id="PH8_denom_root", cache=True)
    denominator_queries: Dict = dict()
    for (region_type, region_domain), iteration_level in itertools.product(
        geo_domains.items(), ITERATION_LEVELS
    ):
        if region_domain:
            region_population_groups = (
                QueryBuilder("PH8_denom_root")
                .map(
                    _iteration_map_dict[iteration_level],
                    new_column_types={"ITERATION_CODE": ColumnType.VARCHAR},
                    augment=True,
                )
                .filter(
                    "ITERATION_CODE != 'N/A'"
                )  # Added for clarity. The KeySet will also filter out these rows.
                .groupby(
                    KeySet.from_dict(
                        {
                            region_type: region_domain,
                            "ITERATION_CODE": ITERATION_CODES[iteration_level],
                            "PH8_DENOM_DATA_CELL": _PH8_map_domain,
                        }
                    )
                )
            )
            denominator_queries[
                (region_type, iteration_level)
            ] = region_population_groups.count(name="COUNT", mechanism=noise_mechanism)
    return denominator_queries


def _stack_answers(answers: Mapping[Tuple[str, str], DataFrame]) -> DataFrame:
    """Return the union of all the answers with the region columns standardized.

    Args:
        answers: The answer for each region.

    Returns:
        A single dataframe with the region columns standardized to "REGION_TYPE" and
        "REGION_ID".
    """
    postprocessed_answers = [
        answer.withColumnRenamed(region_type, "REGION_ID").withColumn(
            "REGION_TYPE", lit(region_type)
        )
        for (region_type, iteration_level), answer in answers.items()
    ]
    return functools.reduce(DataFrame.union, postprocessed_answers)


class PrivateTabulations:
    """Performs private tabulations on Person in Household data."""

    def __init__(
        self,
        tau: Mapping[str, int],
        privacy_budget: Mapping[str, Mapping[str, float]],
        privacy_defn: Union[Literal["puredp"], Literal["zcdp"]],
    ):
        """Constructor.

        Args:
            tau: The maximum number of people per household per tabulation.
            privacy_budget: The total privacy budget per tabulation.
            privacy_defn: The privacy definition with which to interpret the
                privacy budgets.
        """
        self._tau = tau
        self._privacy_budget = privacy_budget
        self._privacy_defn = privacy_defn

    def __call__(self, input_sdfs: PHSafeInput) -> PHSafeOutput:
        """Return PH2, PH3, PH1, PH4, PH5, PH6, PH7, PH8 tabulations.

        Args:
            input_sdfs: PHSafeInput NamedTuple containing persons, units and processed
                        geo dataframes.

        Returns:
            A PHSafeOutput object containing:
                PH1_num: Average household size by age (numerator).
                PH1_denom: Average household size by age (denominator).
                PH2: Household type for the population in households.
                PH3: Household type by relationship for population under 18 years.
                PH4: Population in familY.
                PH5_num: Average family size by age (numerator).
                PH5_denom: Average family size by age (denominator).
                PH6: Family type and age for own children under 18 years.
                PH7: Total population in occupied housing units by tenure.
                PH8_num: Average household size of occupied
                         housing units by tenure (numerator).
                PH8_denom: Average household size of occupied
                            housing units by tenure (denominator).
            The Session that builds the outputs, so that any views can be cleared from
                it after materializing answers.
        """
        logger = logging.getLogger(__name__)
        logger.info("Starting private algorithm execution...")

        # Allocate 0 budget to PH5_num and PH8_num as these can be computed from ph4
        # and PH7 respectively as post-processing.
        updated_privacy_budget = {
            "PH5_num": {
                geo_iteration: float(0)
                for geo_iteration in self._privacy_budget["PH5_denom"].keys()
            },
            "PH8_num": {
                geo_iteration: float(0)
                for geo_iteration in self._privacy_budget["PH8_denom"].keys()
            },
        }

        for tabulation in TABULATIONS_KEY:
            updated_privacy_budget[tabulation] = {
                geo_iteration: float(budget)
                for geo_iteration, budget in self._privacy_budget[tabulation].items()
            }

        total_privacy_budget_per_tabulation = {}
        for tabulation, geo_iteration_budget_dict in updated_privacy_budget.items():
            total_privacy_budget_per_tabulation[tabulation] = sum(
                geo_iteration_budget_dict.values()
            )

        total_budget_float = sum(total_privacy_budget_per_tabulation.values())
        (total_budget, noise_mechanism) = (
            (PureDPBudget(total_budget_float), CountMechanism.LAPLACE)
            if self._privacy_defn == "puredp"
            else (RhoZCDPBudget(total_budget_float), CountMechanism.GAUSSIAN)
        )

        # Create Session
        session = (
            Session.Builder()
            .with_privacy_budget(total_budget)
            .with_private_dataframe(
                source_id="persons",
                dataframe=input_sdfs.persons,
                protected_change=AddOneRow(),
            )
            .with_private_dataframe(
                source_id="units",
                dataframe=input_sdfs.units,
                protected_change=AddMaxRows(2),
            )
            .with_public_dataframe(source_id="geo", dataframe=input_sdfs.geo)
            .build()
        )

        logger.info("Creating queries...")
        # Create queries
        geo_domains = _create_geo_domains(input_sdfs.geo)
        unit_and_geo_builder = (
            QueryBuilder("units")
            .join_public("geo", join_columns=["MAFID"])
            .rename({"RTYPE": "RTYPE_UNIT"})
        )
        session.create_view(unit_and_geo_builder, "unit_and_geo", cache=True)
        query_dict = dict()
        if total_privacy_budget_per_tabulation["PH2"] != 0:
            query_dict["PH2"] = _create_ph2_queries(
                self._tau["PH2"], geo_domains, session, noise_mechanism
            )
        if total_privacy_budget_per_tabulation["PH3"] != 0:
            query_dict["PH3"] = _create_ph3_queries(
                self._tau["PH3"], geo_domains, session, noise_mechanism
            )
        if total_privacy_budget_per_tabulation["PH1_num"] != 0:
            query_dict["PH1_num"] = _create_ph1_num_queries(
                self._tau["PH1_num"], geo_domains, session, noise_mechanism
            )
        if total_privacy_budget_per_tabulation["PH1_denom"] != 0:
            query_dict["PH1_denom"] = _create_ph1_denom_queries(
                geo_domains, session, noise_mechanism
            )
        if total_privacy_budget_per_tabulation["PH4"] != 0:
            query_dict["PH4"] = _create_ph4_queries(
                self._tau["PH4"], geo_domains, session, noise_mechanism
            )
        if total_privacy_budget_per_tabulation["PH5_denom"] != 0:
            query_dict["PH5_denom"] = _create_ph5_denom_queries(
                geo_domains, session, noise_mechanism
            )
        if total_privacy_budget_per_tabulation["PH6"] != 0:
            query_dict["PH6"] = _create_ph6_queries(
                self._tau["PH6"], geo_domains, session, noise_mechanism
            )
        if total_privacy_budget_per_tabulation["PH7"] != 0:
            query_dict["PH7"] = _create_ph7_queries(
                self._tau["PH7"], geo_domains, session, noise_mechanism
            )
        if total_privacy_budget_per_tabulation["PH8_denom"] != 0:
            query_dict["PH8_denom"] = _create_ph8_denom_queries(
                geo_domains, session, noise_mechanism
            )

        logger.info("Answering queries...")
        # Answer queries

        tabulation_answers: Dict[str, List[DataFrame]] = dict()
        for tabulation in TABULATION_OUTPUT_COLUMNS:
            if total_privacy_budget_per_tabulation[tabulation] != 0:
                tabulation_answers[tabulation] = []
                for region_type, iteration_level in itertools.product(
                    geo_domains, ITERATION_LEVELS
                ):
                    # Drop population group query in tabulation if privacy_budget=0
                    if (region_type, iteration_level) in query_dict[
                        tabulation
                    ] and updated_privacy_budget[tabulation][
                        f"{region_type.lower()}_{iteration_level}"
                    ] != 0:
                        per_query_value = updated_privacy_budget[tabulation][
                            f"{region_type.lower()}_{iteration_level}"
                        ]
                        per_query_privacy_budget = (
                            PureDPBudget(per_query_value)
                            if isinstance(total_budget, PureDPBudget)
                            else RhoZCDPBudget(per_query_value)
                        )
                        noise_info = session._noise_info(
                            query_expr=query_dict[tabulation][
                                (region_type, iteration_level)
                            ],
                            privacy_budget=per_query_privacy_budget,
                        )
                        answer = session.evaluate(
                            query_expr=query_dict[tabulation][
                                (region_type, iteration_level)
                            ],
                            privacy_budget=per_query_privacy_budget,
                        )
                        answer = answer.withColumn(
                            "NOISE_DISTRIBUTION",
                            lit(
                                _get_noise_distribution(
                                    noise_info[0]["noise_mechanism"].name
                                )
                            ),
                        ).withColumn(
                            "VARIANCE",
                            lit(
                                _get_variance(
                                    self._privacy_defn, noise_info[0]["noise_parameter"]
                                )
                            ),
                        )
                        tabulation_answers[tabulation].append(answer)
                session.delete_view(f"{tabulation}_root")
        session.delete_view("unit_and_geo")

        logger.info("Evaluation completed and postprocessing answers...")

        # Reorganize answers
        answer_dict: Dict[str, DataFrame] = dict()
        for tabulation, _ in tabulation_answers.items():
            answer_iter = iter(tabulation_answers[tabulation])
            tab_geo_answers = dict()
            for region_type, iteration_level in itertools.product(
                geo_domains, ITERATION_LEVELS
            ):
                if (region_type, iteration_level) in query_dict[
                    tabulation
                ] and updated_privacy_budget[tabulation][
                    f"{region_type.lower()}_{iteration_level}"
                ] != 0:
                    tab_geo_answers[(region_type, iteration_level)] = next(answer_iter)
            answer_dict[tabulation] = tab_geo_answers

            # Postprocess answers
            answer_dict[tabulation] = _stack_answers(answer_dict[tabulation]).select(
                list(TABULATION_OUTPUT_COLUMNS[tabulation].keys())
            )

        stacked_answers = PHSafeOutput(
            PH1_num=answer_dict.get("PH1_num"),
            PH1_denom=answer_dict.get("PH1_denom"),
            PH2=answer_dict.get("PH2"),
            PH3=answer_dict.get("PH3"),
            PH4=answer_dict.get("PH4"),
            PH5_num=(
                answer_dict["PH4"].withColumnRenamed(
                    "PH4_DATA_CELL", "PH5_NUM_DATA_CELL"
                )
                if (
                    answer_dict.get("PH4") is not None
                    and isinstance(answer_dict.get("PH4"), DataFrame)
                )
                else None
            ),
            PH5_denom=answer_dict.get("PH5_denom"),
            PH6=answer_dict.get("PH6"),
            PH7=answer_dict.get("PH7"),
            PH8_num=(
                answer_dict["PH7"]
                .withColumn(
                    "PH8_NUM_DATA_CELL",
                    when(
                        col("PH7_DATA_CELL").isin(
                            [
                                PH7_DATA_CELL.MORTGAGE_OR_LOAN,
                                PH7_DATA_CELL.FREE_AND_CLEAR,
                            ]
                        ),
                        lit(PH8_DATA_CELL.OWNER_OCCUPIED),
                    ).when(
                        col("PH7_DATA_CELL") == PH7_DATA_CELL.RENTER_OCCUPIED,
                        lit(PH8_DATA_CELL.RENTER_OCCUPIED),
                    ),
                )
                .select(
                    [
                        "REGION_ID",
                        "REGION_TYPE",
                        "ITERATION_CODE",
                        "PH8_NUM_DATA_CELL",
                        "COUNT",
                        "NOISE_DISTRIBUTION",
                        "VARIANCE",
                    ]
                )
                .groupBy(
                    [
                        "REGION_ID",
                        "REGION_TYPE",
                        "ITERATION_CODE",
                        "PH8_NUM_DATA_CELL",
                        "NOISE_DISTRIBUTION",
                    ]
                )
                .agg({"COUNT": "sum", "VARIANCE": "sum"})
                .select(
                    col("REGION_ID"),
                    col("REGION_TYPE"),
                    col("ITERATION_CODE"),
                    col("PH8_NUM_DATA_CELL"),
                    col("sum(COUNT)").alias("COUNT"),
                    col("NOISE_DISTRIBUTION"),
                    col("sum(VARIANCE)").alias("VARIANCE"),
                )
                if (
                    answer_dict.get("PH7") is not None
                    and isinstance(answer_dict.get("PH7"), DataFrame)
                )
                else None
            ),
            PH8_denom=answer_dict.get("PH8_denom"),
        )
        return stacked_answers


def _get_noise_distribution(noise_mech: str):
    """Maps _NoiseMechanism returned by analytics to solutions."""
    if noise_mech == "DISCRETE_GAUSSIAN":
        distribution = "Discrete Gaussian"
    else:
        assert noise_mech == "GEOMETRIC"
        distribution = "Two-Sided Geometric"
    return distribution


def _get_variance(privacy_defn: str, noise_parameter: int):
    """Return variance."""
    if privacy_defn == "puredp":
        if noise_parameter == 0:
            return 0
        variance = (
            2 * exp(-1 / noise_parameter) / ((1 - exp(-1 / noise_parameter)) ** 2)
        )
    else:
        assert privacy_defn == "zcdp"
        variance = noise_parameter
    return variance
