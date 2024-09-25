# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['tmlt',
 'tmlt.analytics',
 'tmlt.analytics._query_expr_compiler',
 'tmlt.analytics.constraints']

package_data = \
{'': ['*']}

install_requires = \
['sympy>=1.8,<1.10',
 'tmlt.core>=0.11.5,<0.13.0',
 'typeguard>=2.12.1,<2.13.0',
 'typing-extensions>=4.1.0,<5.0.0']

extras_require = \
{':python_version < "3.8"': ['pandas>=1.2.0,<1.4.0',
                             'pyspark[sql]>=3.0.0,<3.4.0'],
 ':python_version >= "3.10" and python_version < "3.11"': ['pandas>=1.4.0,<2.0.0'],
 ':python_version >= "3.11"': ['pandas>=1.5.0,<2.0.0',
                               'pyspark[sql]>=3.4.0,<3.6.0'],
 ':python_version >= "3.8" and python_version < "3.10"': ['pandas>=1.2.0,<2.0.0'],
 ':python_version >= "3.8" and python_version < "3.11"': ['pyspark[sql]>=3.0.0,<3.6.0']}

setup_kwargs = {
    'name': 'tmlt-analytics',
    'version': '0.8.3',
    'description': "Tumult's differential privacy analytics API",
    'long_description': "[![PyPI - Version](https://img.shields.io/pypi/v/tmlt-analytics?color=006dad)](https://pypi.org/project/tmlt-analytics/) |\n[![Documentation - Latest](https://img.shields.io/badge/documentation-latest-cc3d56)](https://docs.tmlt.dev/analytics/latest/) |\n[![Join our Slack!](https://img.shields.io/badge/Join%20our%20Slack!-634ad3?logo=slack)](https://tmlt.dev/slack)\n\n# Tumult Analytics\n\nTumult Analytics is a library that allows users to execute differentially private operations on\ndata without having to worry about the privacy implementation, which is handled\nautomatically by the API. It is built atop the [Tumult Core library](https://gitlab.com/tumult-labs/core).\n\n## Demo video\n\nWant to see Tumult Analytics in action? Check out this video introducing the\ninterface fundamentals:\n\n[![Screenshot of the demo video](https://img.youtube.com/vi/SNfbYOp0CEs/0.jpg)](https://www.youtube.com/watch?v=SNfbYOp0CEs)\n\nA selection of more advanced features is shown on the second part of this demo,\nin a [separate video](https://www.youtube.com/watch?v=BRUPlfwzHHo).\n\n## Installation\n\nSee the [installation instructions in the documentation](https://docs.tmlt.dev/analytics/latest/installation.html#prerequisites)\nfor information about setting up prerequisites such as Spark.\n\nOnce the prerequisites are installed, you can install Tumult Analytics using [pip](https://pypi.org/project/pip).\n\n```bash\npip install tmlt.analytics\n```\n\n## Documentation\n\nThe full documentation is located at https://docs.tmlt.dev/analytics/latest/.\n\n## Support\n\nIf you have any questions, feedback, or feature requests, please reach out to us on [Slack](https://tmlt.dev/slack).\n\n## Contributing\n\nWe do not yet have a process in place to accept external contributions, but we are open to collaboration opportunities.\nIf you are interested in contributing, please let us know [via Slack](https://tmlt.dev/slack).\n\nSee [CONTRIBUTING.md](CONTRIBUTING.md) for information about installing our development dependencies and running tests.\n\n## Citing Tumult Analytics\n\nIf you use Tumult Analytics for a scientific publication, we would appreciate citations to the published software or/and its whitepaper. Both citations can be found below; for the software citation, please replace the version with the version you are using.\n\n```\n@software{tumultanalyticssoftware,\n    author = {Tumult Labs},\n    title = {Tumult {{Analytics}}},\n    month = dec,\n    year = 2022,\n    version = {latest},\n    url = {https://tmlt.dev}\n}\n```\n\n```\n@article{tumultanalyticswhitepaper,\n  title={Tumult {{Analytics}}: a robust, easy-to-use, scalable, and expressive framework for differential privacy},\n  author={Berghel, Skye and Bohannon, Philip and Desfontaines, Damien and Estes, Charles and Haney, Sam and Hartman, Luke and Hay, Michael and Machanavajjhala, Ashwin and Magerlein, Tom and Miklau, Gerome and Pai, Amritha and Sexton, William and Shrestha, Ruchit},\n  journal={arXiv preprint arXiv:2212.04133},\n  month = dec,\n  year={2022}\n}\n```\n\n## License\n\nCopyright Tumult Labs 2023\n\nTumult Analytics' source code is licensed under the Apache License, version 2.0 (Apache-2.0).\nTumult Analytics' documentation is licensed under\nCreative Commons Attribution-ShareAlike 4.0 International (CC-BY-SA-4.0).\n",
    'author': 'None',
    'author_email': 'None',
    'maintainer': 'None',
    'maintainer_email': 'None',
    'url': 'https://www.tmlt.dev/',
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'extras_require': extras_require,
    'python_requires': '>=3.7.1,<3.12.0',
}


setup(**setup_kwargs)
