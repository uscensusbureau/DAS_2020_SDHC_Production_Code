"""A set of named constants for commonly-used values."""

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

from enum import IntEnum, StrEnum


class RTYPE_PERSON(StrEnum):
    """All the possible values of RTYPE in persons_df.

    RTYPE is the record type, and records the type of housing a person lives in.
    """

    IN_HOUSING_UNIT = "3"
    IN_GQ = "5"


class RaceCode(StrEnum):
    """All the possible race code values.

    These appear in the CENRACE and HHRACE columns of persons_df and units_df
    respectively. Note that unlike most other enums in this file, the name of
    the enum is not the name of the column where these values appear.
    """

    WHITE_ALONE = "01"
    BLACK_ALONE = "02"
    AIAN_ALONE = "03"
    ASIAN_ALONE = "04"
    NHPI_ALONE = "05"
    SOR_ALONE = "06"
    WHITE_BLACK = "07"
    WHITE_AIAN = "08"
    WHITE_ASIAN = "09"
    WHITE_NHPI = "10"
    WHITE_SOR = "11"
    BLACK_AIAN = "12"
    BLACK_ASIAN = "13"
    BLACK_NHPI = "14"
    BLACK_SOR = "15"
    AIAN_ASIAN = "16"
    AIAN_NHPI = "17"
    AIAN_SOR = "18"
    ASIAN_NHPI = "19"
    ASIAN_SOR = "20"
    NHPI_SOR = "21"
    WHITE_BLACK_AIAN = "22"
    WHITE_BLACK_ASIAN = "23"
    WHITE_BLACK_NHPI = "24"
    WHITE_BLACK_SOR = "25"
    WHITE_AIAN_ASIAN = "26"
    WHITE_AIAN_NHPI = "27"
    WHITE_AIAN_SOR = "28"
    WHITE_ASIAN_NHPI = "29"
    WHITE_ASIAN_SOR = "30"
    WHITE_NHPI_SOR = "31"
    BLACK_AIAN_ASIAN = "32"
    BLACK_AIAN_NHPI = "33"
    BLACK_AIAN_SOR = "34"
    BLACK_ASIAN_NHPI = "35"
    BLACK_ASIAN_SOR = "36"
    BLACK_NHPI_SOR = "37"
    AIAN_ASIAN_NHPI = "38"
    AIAN_ASIAN_SOR = "39"
    AIAN_NHPI_SOR = "40"
    ASIAN_NHPI_SOR = "41"
    WHITE_BLACK_AIAN_ASIAN = "42"
    WHITE_BLACK_AIAN_NHPI = "43"
    WHITE_BLACK_AIAN_SOR = "44"
    WHITE_BLACK_ASIAN_NHPI = "45"
    WHITE_BLACK_ASIAN_SOR = "46"
    WHITE_BLACK_NHPI_SOR = "47"
    WHITE_AIAN_ASIAN_NHPI = "48"
    WHITE_AIAN_ASIAN_SOR = "49"
    WHITE_AIAN_NHPI_SOR = "50"
    WHITE_ASIAN_NHPI_SOR = "51"
    BLACK_AIAN_ASIAN_NHPI = "52"
    BLACK_AIAN_ASIAN_SOR = "53"
    BLACK_AIAN_NHPI_SOR = "54"
    BLACK_ASIAN_NHPI_SOR = "55"
    AIAN_ASIAN_NHPI_SOR = "56"
    WHITE_BLACK_AIAN_ASIAN_NHPI = "57"
    WHITE_BLACK_AIAN_ASIAN_SOR = "58"
    WHITE_BLACK_AIAN_NHPI_SOR = "59"
    WHITE_BLACK_ASIAN_NHPI_SOR = "60"
    WHITE_AIAN_ASIAN_NHPI_SOR = "61"
    BLACK_AIAN_ASIAN_NHPI_SOR = "62"
    WHITE_BLACK_AIAN_ASIAN_NHPI_SOR = "63"


class EthCode(IntEnum):
    """All the possible ethnicity code values.

    These appear in the CENHISP and HHSPAN columns of persons_df and units_df
    respectively. Note that unlike most other enums in this file, the name of
    the enum is not the name of the column where these values appear.
    """

    NOT_HISPANIC = 1
    HISPANIC = 2


class ITERATION_CODE(StrEnum):
    """All the iteration codes.

    These are the possible values of the ITERATION_CODE columns in the output datasets.
    """

    UNATTRIBUTED = "*"
    WHITE_ALONE = "A"
    BLACK_ALONE = "B"
    AIAN_ALONE = "C"
    ASIAN_ALONE = "D"
    NHPI_ALONE = "E"
    SOME_OTHER_RACE_ALONE = "F"
    TWO_OR_MORE_RACES = "G"
    HISPANIC_OR_LATINO = "H"
    WHITE_ALONE_NOT_HISPANIC = "I"


class RELSHIP(StrEnum):
    """The values for the RELSHIP column in persons_df.

    RELSHIP encodes the person's relationshuip with the householder of their unit.
    """

    HOUSEHOLDER = "20"
    OPPOSITE_SEX_SPOUSE = "21"
    OPPOSITE_SEX_UNMARRIED = "22"
    SAME_SEX_SPOUSE = "23"
    SAME_SEX_UNMARRIED = "24"
    BIOLOGICAL_CHILD = "25"
    ADOPTED_CHILD = "26"
    STEPCHILD = "27"
    SIBLING = "28"
    PARENT = "29"
    GRANDCHILD = "30"
    PARENT_IN_LAW = "31"
    CHILD_IN_LAW = "32"
    OTHER_RELATIVE = "33"
    ROOMMATE_OR_HOUSEMATE = "34"
    FOSTER_CHILD = "35"
    OTHER_NONRELATIVE = "36"
    INSTITUTIONAL_GQ_PERSON = "37"
    NON_INSTITUTIONAL_GQ_PERSON = "38"


class RTYPE_UNIT(StrEnum):
    """All the values for the RTYPE column in unit_df.

    RTYPE is the record type. It indicates the type of housing unit.
    """

    HOUSING_UNIT = "2"
    GQ = "4"


class TEN(StrEnum):
    """The values for the TEN column in unit_df.

    TEN is the tenure, and documents on what terms the unit is occupied.
    """

    NOT_IN_UNIVERSE = "0"
    OWNED_WITH_A_MORTGAGE = "1"
    OWNED_FREE_AND_CLEAR = "2"
    RENTED = "3"
    OCCUPIED_WITHOUT_PAYMENT_OF_RENT = "4"


class HHT(StrEnum):
    """The values for the HHT column in unit_df.

    HHT is the household type.
    """

    NOT_IN_UNIVERSE = "0"
    MARRIED_COUPLE_HOUSEHOLD = "1"
    OTHER_FAMILY_HOUSEHOLD_MALE_HOUSEHOLDER = "2"
    OTHER_FAMILY_HOUSEHOLD_FEMALE_HOUSEHOLDER = "3"
    NONFAMILY_HOUSEHOLD_MALE_HOUSEHOLDER_LIVING_ALONE = "4"
    NONFAMILY_HOUSEHOLD_MALE_HOUSEHOLDER_NOT_LIVING_ALONE = "5"
    NONFAMILY_HOUSEHOLD_FEMALE_HOUSEHOLDER_LIVING_ALONE = "6"
    NONFAMILY_HOUSEHOLD_FEMALE_HOUSEHOLDER_NOT_LIVING_ALONE = "7"


class HHT2(StrEnum):
    """The values for the HHT column in unit_df.

    HHT2 is a more detailed version of the household type.
    """

    NOT_IN_UNIVERSE = "00"
    MARRIED_COUPLE_WITH_OWN_CHILDREN = "01"
    MARRIED_COUPLE_WITHOUT_OWN_CHILDREN = "02"
    COHABITING_COUPLE_WITH_OWN_CHILDREN = "03"
    COHABITING_COUPLE_WITHOUT_OWN_CHILDREN = "04"
    UNPARTNERED_FEMALE_HH_ALONE = "05"
    UNPARTNERED_FEMALE_HH_WITH_OWN_CHILDREN = "06"
    UNPARTNERED_FEMALE_HH_WITH_ADULT_RELATIVES = "07"
    UNPARTNERED_FEMALE_HH_ONLY_NONRELATIVES = "08"
    UNPARTNERED_MALE_HH_ALONE = "09"
    UNPARTNERED_MALE_HH_WITH_OWN_CHILDREN = "10"
    UNPARTNERED_MALE_HH_WITH_ADULT_RELATIVES = "11"
    UNPARTNERED_MALE_HH_ONLY_NONRELATIVES = "12"


class CPLT(StrEnum):
    """The values for the CPLT column in unit_df.

    CPLT is the couple type.
    """

    NOT_IN_UNIVERSE = "0"
    OPPOSITE_SEX_MARRIED = "1"
    SAME_SEX_MARRIED = "2"
    OPPOSITE_SEX_UNMARRIED = "3"
    SAME_SEX_UNMARRIED = "4"
    ALL_OTHER_HOUSEHOLDS = "5"


class PH1_DATA_CELL(IntEnum):
    """All possible PH1_DATA_CELL values."""

    TOTAL = 1
    UNDER_18 = 2
    AGE_18_AND_OVER = 3


class PH2_DATA_CELL(IntEnum):
    """The values for PH2_DATA_CELL."""

    OPPOSITE_SEX_MARRIED_COUPLE = 3
    SAME_SEX_MARRIED_COUPLE = 4
    OPPOSITE_SEX_COHABITING_COUPLE = 6
    SAME_SEX_COHABITING_COUPLE = 7
    UNPARTNERED_MALE_HOUSEHOLDER_ALONE = 9
    UNPARTNERED_MALE_HOUSEHOLDER_WITH_OTHERS = 10
    UNPARTNERED_FEMALE_HOUSEHOLDER_ALONE = 12
    UNPARTNERED_FEMALE_HOUSEHOLDER_WITH_OTHERS = 13


class PH3_DATA_CELL(IntEnum):
    """The values for PH3_DATA_CELL."""

    HOUSEHOLDER_PARTNER_OR_NONRELATIVE = 2
    OWN_CHILD_MARRIED_FAMILY = 4
    OWN_CHILD_COHABITING_FAMILY = 5
    OWN_CHILD_UNPARTNERED_MALE_FAMILY = 6
    OWN_CHILD_UNPARTNERED_FEMALE_FAMILY = 7
    GRANDCHILD = 9
    OTHER_RELATIVES = 10


class PH4_DATA_CELL(IntEnum):
    """All possible PH4_DATA_CELL values."""

    UNDER_18 = 2
    AGE_18_AND_OVER = 3


class PH6_DATA_CELL(IntEnum):
    """All possible PH6_DATA_CELL values."""

    MARRIED_FAMILY_UNDER_4 = 3
    MARRIED_FAMILY_4_AND_5 = 4
    MARRIED_FAMILY_6_TO_11 = 5
    MARRIED_FAMILY_12_TO_17 = 6
    COHABITING_FAMILY_UNDER_4 = 8
    COHABITING_FAMILY_4_AND_5 = 9
    COHABITING_FAMILY_6_TO_11 = 10
    COHABITING_FAMILY_12_TO_17 = 11
    UNPARTNERED_MALE_FAMILY_UNDER_4 = 13
    UNPARTNERED_MALE_FAMILY_4_AND_5 = 14
    UNPARTNERED_MALE_FAMILY_6_TO_11 = 15
    UNPARTNERED_MALE_FAMILY_12_TO_17 = 16
    UNPARTNERED_FEMALE_FAMILY_UNDER_4 = 18
    UNPARTNERED_FEMALE_FAMILY_4_AND_5 = 19
    UNPARTNERED_FEMALE_FAMILY_6_TO_11 = 20
    UNPARTNERED_FEMALE_FAMILY_12_TO_17 = 21


class PH7_DATA_CELL(IntEnum):
    """All possible PH7_DATA_CELL values."""

    MORTGAGE_OR_LOAN = 2
    FREE_AND_CLEAR = 3
    RENTER_OCCUPIED = 4


class PH8_DATA_CELL(IntEnum):
    """All possible PH8_DATA_CELL values."""

    TOTAL = 1
    OWNER_OCCUPIED = 2
    RENTER_OCCUPIED = 3
