# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['tmlt', 'tmlt.common']

package_data = \
{'': ['*']}

install_requires = \
['boto3>=1.16.0,<2.0.0',
 'numpy>=1.21.0,<1.26.1',
 'pandas>=1.2.0,<2.0.0',
 'parameterized>=0.7.4,<0.8.0',
 'pyarrow>=14.0.1,<15.0.0',
 'pyspark[sql]>=3.0.0,<4.0.0',
 'pytest>=7.1.2,<8.0.0',
 'smart_open>=5.2.0,<6.0.0']

extras_require = \
{':python_version < "3.8"': ['scipy>=1.4.1,<1.8.0'],
 ':python_version >= "3.10" and python_version < "3.11"': ['scipy>=1.8.0,<2.0.0'],
 ':python_version >= "3.11"': ['scipy>=1.9.2,<2.0.0'],
 ':python_version >= "3.8" and python_version < "3.9"': ['scipy>=1.6.0,<1.11.0'],
 ':python_version >= "3.9" and python_version < "3.10"': ['scipy>=1.6.0,<2.0.0']}

setup_kwargs = {
    'name': 'tmlt-common',
    'version': '0.8.7.post1',
    'description': 'Common utility functions used by Tumult projects',
    'long_description': '# Common Utility\n\nThis module primarily contains common utility functions used by different Tumult projects.\n\n<placeholder: add notice if required>\n\n## Overview\n\nThe utility functions include:\n* Methods to serialize/deserialize objects into json format (marshallable).\n* Expected error computations.\n* A tool for creating error reports.\n* Helper functions to assist with reading tmlt.analytics outputs (io_helpers).\n* Helper functions to assist with data ingestion (schema and validation).\n\nSee [CHANGELOG](CHANGELOG.md) for version number information and changes from past versions.\n\n## Testing\n\nTo run the tests, install the required dependencies from the `test_requirements.txt`\n\n```\npip install -r test_requirements.txt\n```\n\n*All tests (including Doctest):*\n\n```bash\npytest tmlt/common\n```\n\nSee `examples` for examples of features of `common`.\n',
    'author': 'None',
    'author_email': 'None',
    'maintainer': 'None',
    'maintainer_email': 'None',
    'url': 'None',
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'extras_require': extras_require,
    'python_requires': '>=3.8.0,<3.12',
}


setup(**setup_kwargs)
