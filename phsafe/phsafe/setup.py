# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['tmlt', 'tmlt.phsafe']

package_data = \
{'': ['*'], 'tmlt.phsafe': ['resources/config/input/*',
                               'resources/config/output/*']}

install_requires = \
['tmlt.common==0.8.7.post1',
 'tmlt.analytics==0.8.3',
 'numpy>=1.21.0,<1.26.1',
 'pandas>=1.5.0,<2.0.0',
 'pyspark[sql]>=3.4.0',
 'smart-open==5.2.1',

]

setup_kwargs = {
    'name': 'tmlt-phsafe',
    'version': '3.0.0',
    'description': 'PHSafe',
    'long_description': "# PHSafe",
    'author': 'None',
    'author_email': 'None',
    'maintainer': 'None',
    'maintainer_email': 'None',
    'url': 'None',
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.9,<3.12',
}


setup(**setup_kwargs)
