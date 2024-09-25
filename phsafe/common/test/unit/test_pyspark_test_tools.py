"""Unit tests for :mod:`~tmlt.common.pyspark_test_tools`."""

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

# pylint:disable=no-self-use

from operator import add

import pandas as pd
import pytest
from pandas import DataFrame
from parameterized import parameterized
from pyspark.sql import SparkSession

from tmlt.common.pyspark_test_tools import (  # pylint: disable=unused-import
    PySparkTest,
    assert_frame_equal_with_sort,
)


class TestSparkTestHarness(PySparkTest):
    """Test pyspark testing base class."""

    def test_basic(self):
        """Word count test."""
        test_rdd = self.spark.sparkContext.parallelize(
            ["hello spark", "hello again spark spark"], 2
        )
        results = (
            test_rdd.flatMap(lambda line: line.split())
            .map(lambda word: (word, 1))
            .reduceByKey(add)
            .collect()
        )
        expected_results = [("hello", 2), ("spark", 3), ("again", 1)]
        assert set(results) == set(expected_results)

    def test_get_session(self):
        """Tests that *getOrCreate()* connects to test harness SparkSession."""
        # pylint: disable=no-member
        spark = SparkSession.builder.getOrCreate()
        assert spark.conf.get("spark.app.name") == "TestSparkTestHarness"
        # pylint: enable=no-member

    @parameterized.expand(
        [
            (
                pd.DataFrame.from_records(
                    [
                        {"name": "Jane Doe", "grade": 4},
                        {"name": "John Smith", "grade": 5},
                    ]
                ),
                pd.DataFrame.from_records(
                    [
                        {"name": "Jane Doe", "grade": 4},
                        {"name": "John Smith", "grade": 3},
                    ]
                ),
                False,
            ),
            (
                pd.DataFrame.from_records(
                    [
                        {"name": "Jane Doe", "grade": 4},
                        {"name": "John Smith", "grade": 5},
                    ]
                ),
                pd.DataFrame.from_records(
                    [
                        {"name": "Jane Doe", "grade": 4},
                        {"name": "John Smith", "grade": 5},
                    ]
                ),
                True,
            ),
            (pd.DataFrame.from_records([]), pd.DataFrame.from_records([]), True),
            (
                pd.DataFrame.from_records(
                    [
                        {"name": "Jane Doe", "grade": 4},
                        {"name": "John Smith", "grade": 5},
                    ]
                ),
                pd.DataFrame.from_records(
                    [
                        {"name": "John Smith", "grade": 5},
                        {"name": "Jane Doe", "grade": 4},
                    ]
                ),
                True,
            ),
            (
                pd.DataFrame.from_records(
                    [
                        {"name": "Jane Doe", "grade": 4},
                        {"name": "John Smith", "grade": 5},
                    ]
                ),
                pd.DataFrame.from_records(
                    [
                        {"name": "John Smith", "grade": 5},
                        {"name": "Jane Doe", "grade": 4},
                        {"name": "Dana", "grade": 3},
                    ]
                ),
                False,
            ),
            (
                pd.DataFrame.from_records([]),
                pd.DataFrame.from_records(
                    [
                        {"name": "John Smith", "grade": 5},
                        {"name": "Jane Doe", "grade": 4},
                    ]
                ),
                False,
            ),
            (
                pd.DataFrame.from_records(
                    [
                        {"name": "John Smith", "grade": 5, "score": 1},
                        {"name": "Jane Doe", "grade": 4, "score": 2},
                    ]
                ),
                pd.DataFrame.from_records(
                    [
                        {"name": "John Smith", "grade": 5},
                        {"name": "Jane Doe", "grade": 4},
                    ]
                ),
                False,
            ),
        ]
    )
    def test_class_assert_frame_equal_with_sort(
        self, df_one: DataFrame, df_two: DataFrame, assertion: bool
    ):
        """Tests the assert_frame_equal_with_sort method, which is commonly used in
        PySpark testing."""

        if assertion:
            self.assert_frame_equal_with_sort(df_one, df_two)
        else:
            with pytest.raises(AssertionError):
                self.assert_frame_equal_with_sort(df_one, df_two)


def test_fixture_wordcount(spark):
    """Word count test."""
    test_rdd = spark.sparkContext.parallelize(
        ["hello spark", "hello again spark spark"], 2
    )
    results = (
        test_rdd.flatMap(lambda line: line.split())
        .map(lambda word: (word, 1))
        .reduceByKey(add)
        .collect()
    )
    expected_results = [("hello", 2), ("spark", 3), ("again", 1)]
    assert set(results) == set(expected_results)


# pylint: disable=no-member
def test_fixture_properties(spark):  # pylint: disable=unused-argument
    """Tests that *getOrCreate()* connects to test harness SparkSession."""
    other_spark = SparkSession.builder.getOrCreate()
    assert other_spark.conf.get("spark.app.name") == "tmlt.common.pyspark_test_tools"


# pylint: enable=no-member


@pytest.mark.parametrize(
    ("df_one", "df_two", "assertion"),
    [
        (
            pd.DataFrame.from_records(
                [{"name": "Jane Doe", "grade": 4}, {"name": "John Smith", "grade": 5}]
            ),
            pd.DataFrame.from_records(
                [{"name": "Jane Doe", "grade": 4}, {"name": "John Smith", "grade": 3}]
            ),
            False,
        ),
        (
            pd.DataFrame.from_records(
                [{"name": "Jane Doe", "grade": 4}, {"name": "John Smith", "grade": 5}]
            ),
            pd.DataFrame.from_records(
                [{"name": "Jane Doe", "grade": 4}, {"name": "John Smith", "grade": 5}]
            ),
            True,
        ),
        (pd.DataFrame.from_records([]), pd.DataFrame.from_records([]), True),
        (
            pd.DataFrame.from_records(
                [{"name": "Jane Doe", "grade": 4}, {"name": "John Smith", "grade": 5}]
            ),
            pd.DataFrame.from_records(
                [{"name": "John Smith", "grade": 5}, {"name": "Jane Doe", "grade": 4}]
            ),
            True,
        ),
        (
            pd.DataFrame.from_records(
                [{"name": "Jane Doe", "grade": 4}, {"name": "John Smith", "grade": 5}]
            ),
            pd.DataFrame.from_records(
                [
                    {"name": "John Smith", "grade": 5},
                    {"name": "Jane Doe", "grade": 4},
                    {"name": "Dana", "grade": 3},
                ]
            ),
            False,
        ),
        (
            pd.DataFrame.from_records([]),
            pd.DataFrame.from_records(
                [{"name": "John Smith", "grade": 5}, {"name": "Jane Doe", "grade": 4}]
            ),
            False,
        ),
        (
            pd.DataFrame.from_records(
                [
                    {"name": "John Smith", "grade": 5, "score": 1},
                    {"name": "Jane Doe", "grade": 4, "score": 2},
                ]
            ),
            pd.DataFrame.from_records(
                [{"name": "John Smith", "grade": 5}, {"name": "Jane Doe", "grade": 4}]
            ),
            False,
        ),
    ],
)
def test_assert_frame_equal_with_sort(
    df_one: DataFrame, df_two: DataFrame, assertion: bool
):
    """Tests the assert_frame_equal_with_sort method, which is commonly used in
    PySpark testing."""

    if assertion:
        assert_frame_equal_with_sort(df_one, df_two)
    else:
        with pytest.raises(AssertionError):
            assert_frame_equal_with_sort(df_one, df_two)
