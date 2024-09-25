"""The interface for PHSafe readers."""

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
from abc import ABC, abstractmethod
from typing import List

from pyspark.sql import DataFrame as SparkDataFrame


class AbstractPHSafeReader(ABC):
    """The interface for PHSafe readers."""

    @abstractmethod
    def __init__(self, config_path: str, state_filter: List[str]):
        """Initialize reader with config path and the state FIPS codes to read.

        Args:
            config_path: The path to a config file used to configure the reader.
            state_filter: A list of state FIPS codes to filter on.
        """

    @abstractmethod
    def get_geo_df(self) -> SparkDataFrame:
        """Return a spark dataframe derived from the CEF Unit file and the GRFC.

        This should contain only the state codes specified by the state_filter.

        Returns:
            A spark dataframe containing all of the expected columns from `geo_df` in
                `Appendix A`.
        """

    @abstractmethod
    def get_unit_df(self) -> SparkDataFrame:
        """Return a spark dataframe derived from the CEF unit file.

        This should contain only records with MAFIDs corresponding to the state codes
        specified by the state_filter.

        Returns:
            A spark dataframe containing all of the expected columns from `unit_df` in
                `Appendix A`.
        """

    @abstractmethod
    def get_person_df(self) -> SparkDataFrame:
        """Return a spark dataframe derived from the CEF person file.

        This should contain only records with MAFIDs corresponding to the state codes
        specified by the state_filter.

        Returns:
            A spark dataframe containing all of the expected columns from `person_df`
                in `Appendix A`.
        """
