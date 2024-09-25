"""Paths and file names used by PHSafe."""

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

import os

RESOURCES_PACKAGE_NAME = "resources"
"""The name of the directory containing resources"""

RESOURCES_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), RESOURCES_PACKAGE_NAME)
)
"""Path to directory containing resources for Person in Households."""

INPUT_CONFIG_DIR = RESOURCES_PACKAGE_NAME + "/config/input"
"""Directory containing initial configs for input files."""

OUTPUT_CONFIG_DIR = RESOURCES_PACKAGE_NAME + "/config/output"
"""Directory containing configs for output files validation."""

ALT_INPUT_CONFIG_DIR = "/tmp/phsafe_input_configs"
ALT_OUTPUT_CONFIG_DIR = "/tmp/phsafe_output_configs"
"""The config directories to use for spark-compatible version of PHSafe.

Config files are copied to this directory from tmlt.phsafe resources. They
    cannot be used
directly because PHSafe resources may be zipped.
"""

CONFIG_FILES_PHSAFE = ["persons.json", "units.json", "geo.json"]
"""List of all expected PHSafe config files that have a json format."""

VALIDATION_CONFIG_FILES = ["validate_puredp.json", "validate_zcdp.json"]
"""List of configuration files used for variance validation."""
