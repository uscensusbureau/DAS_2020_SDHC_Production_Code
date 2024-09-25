"""Creates a Spark Context to use for each testing session."""

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

# pylint: disable=unused-import, consider-using-generator
from typing import Any, Dict, List

import pytest


def parametrize(argnames: str, argvalues: List[Dict[str, Any]], **kwargs):
    """Parametrize a test function with the given arguments."""
    args = argnames.split(", ")
    return pytest.mark.parametrize(
        argnames, [tuple([case[arg] for arg in args]) for case in argvalues], **kwargs
    )
