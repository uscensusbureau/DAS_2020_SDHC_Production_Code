"""Tests csv reader on toy data."""

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
import shutil
import tempfile

import pytest

from tmlt.common.pyspark_test_tools import pyspark  # pylint: disable=unused-import
from tmlt.phsafe.csv_reader import (
    GEO_FILENAME,
    PERSON_FILENAME,
    UNIT_FILENAME,
    CSVReader,
)


@pytest.fixture(scope="class")
def setup_csv_reader(request):
    """Set up test."""
    request.cls.tmp_dir = tempfile.mkdtemp()

    geo = """RTYPE,MAFID,TABBLKST,TABBLKCOU,TABTRACTCE,TABBLK,AIANNHCE\n
2,100000001,44,055,450200,4069,9999\n
2,100000002,44,161,890500,2078,9999\n
2,100000003,29,161,890400,1073,9999\n
4,100000004,29,065,960100,1123,9999"""

    person = """RTYPE,MAFID,QAGE,CENHISP,CENRACE,RELSHIP\n
3,100000001,19,1,01,20\n
3,100000001,18,1,01,21\n
3,100000002,19,1,01,20\n
5,100000004,94,1,01,38\n
5,100000004,19,2,01,38"""

    unit = """RTYPE,MAFID,FINAL_POP,NPF,HHSPAN,HHRACE,TEN,HHT,HHT2,CPLT\n
2,100000001,2,2,1,01,2,1,02,1\n
2,100000002,1,0,1,01,2,4,09,5\n
2,100000003,0,0,0,00,0,0,00,0\n
4,100000004,2,0,0,00,0,0,00,0"""

    geo_filename = os.path.join(request.cls.tmp_dir, GEO_FILENAME)
    unit_filename = os.path.join(request.cls.tmp_dir, UNIT_FILENAME)
    person_filename = os.path.join(request.cls.tmp_dir, PERSON_FILENAME)
    with open(geo_filename, "w") as f:
        f.write(geo)
    with open(unit_filename, "w") as f:
        f.write(unit)
    with open(person_filename, "w") as f:
        f.write(person)

    request.cls.reader = CSVReader(request.cls.tmp_dir, ["44", "29"])
    yield
    shutil.rmtree(request.cls.tmp_dir)


@pytest.mark.usefixtures("setup_csv_reader")
@pytest.mark.usefixtures("spark")
class TestCSVReader:
    """Parameterized unit tests for csv reader."""

    tmp_dir: str
    reader: CSVReader

    def test_read_geo_df(self):
        """Load geo df."""
        df = self.reader.get_geo_df()
        assert df.count() == 4

    def test_read_unit_df(self):
        """Load unit df."""
        df = self.reader.get_unit_df()
        assert df.count() == 4

    def test_read_person_df(self):
        """Load person df."""
        df = self.reader.get_person_df()
        assert df.count() == 5

    def test_state_filter(self):
        """Test filter."""
        reader = CSVReader(self.tmp_dir, ["44"])
        geo = reader.get_geo_df()
        person = reader.get_person_df()
        unit = reader.get_unit_df()
        assert geo.count() == 2
        assert person.count() == 3
        assert unit.count() == 2
