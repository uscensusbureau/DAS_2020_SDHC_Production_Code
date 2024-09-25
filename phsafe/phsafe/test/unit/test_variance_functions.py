"""Tests PHSafe variance functions."""

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


import pytest

from tmlt.phsafe.utils import (
    _dp_denom,
    _dp_double,
    _dp_standard,
    _zc_denom,
    _zc_double,
    _zc_standard,
)


class TestValidationFunctions:
    """Parameterized unit tests for functions used to validate variance"""

    @pytest.mark.parametrize(
        "tau, budget, expected",
        [
            (0.5, 0.5, 71.83356455990236),
            (1.0, 1.0, 31.83385287773731),
            (2.0, 2.0, 17.834255192513016),
        ],
    )
    def test_dp_standard(self, tau, budget, expected):
        """Tests the pure dp standard variance calculation"""
        assert _dp_standard(tau, budget) == pytest.approx(expected)

    @pytest.mark.parametrize(
        "tau, budget, expected",
        [
            (0.5, 0.5, 143.6671291198047),
            (1.0, 1.0, 63.66770575547462),
            (2.0, 2.0, 35.66851038502603),
        ],
    )
    def test_dp_double(self, tau, budget, expected):
        """Tests the pure dp double variance calculation"""
        assert _dp_double(tau, budget) == pytest.approx(expected)

    @pytest.mark.parametrize(
        "budget, expected",
        [(0.5, 31.83385287773731), (1.0, 7.835396178065527), (2.0, 1.8413471884155848)],
    )
    def test_dp_denom(self, budget, expected):
        """Tests the pure dp denomination variance calculation"""
        assert _dp_denom(budget) == pytest.approx(expected)

    @pytest.mark.parametrize(
        "tau, budget, expected", [(0.5, 0.5, 9.0), (1.0, 1.0, 8.0), (2.0, 2.0, 9.0)]
    )
    def test_zc_standard(self, tau, budget, expected):
        """Tests the zcdp standard variance calculation"""
        assert _zc_standard(tau, budget) == expected

    @pytest.mark.parametrize(
        "tau, budget, expected", [(0.5, 0.5, 18.0), (1.0, 1.0, 16.0), (2.0, 2.0, 18.0)]
    )
    def test_zc_double(self, tau, budget, expected):
        """Tests the zcdp double variance calculation"""
        assert _zc_double(tau, budget) == expected

    @pytest.mark.parametrize("budget, expected", [(0.5, 4.0), (1.0, 2.0), (2.0, 1.0)])
    def test_zc_denom(self, budget, expected):
        """Tests the zcdp denomination variance calculation"""
        assert _zc_denom(budget) == expected
