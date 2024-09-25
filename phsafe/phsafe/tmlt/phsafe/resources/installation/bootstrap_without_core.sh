#! /bin/bash -xe

# Install necessary Python and Spark Dependencies
# Python 3.11 requires openssl 1.1.1. The default openssl is 1.0.1.
# We need to uninstall 1.0.1, otherwise python will use it preferentially.
sudo yum -y remove openssl openssl-devel
sudo yum -y install openssl11 openssl11-devel
sudo yum -y install gcc bzip2-devel libffi-devel git

# Collect Python 3.11.6
wget https://www.python.org/ftp/python/3.11.6/Python-3.11.6.tgz
sudo tar xzf Python-3.11.6.tgz

# Install Python 3.11.6
cd Python-3.11.6
sudo ./configure --enable-optimizations
sudo make altinstall

# Set default python call to point to Python 3.11
PYTHON=$(which python3.11)

# Once python is configured and installed (pointing to openssl 1.1.1),
# we re-install openssl 1.0.1 because EMR will crash without it.
sudo yum -y install openssl

# Setup Python 3.11 Pip to install all dependencies
sudo $PYTHON -m pip install --upgrade pip
sudo $PYTHON -m pip install boto3==1.28.29
sudo $PYTHON -m pip install botocore==1.31.29
sudo $PYTHON -m pip install colorama==0.4.6
sudo $PYTHON -m pip install iniconfig==2.0.0
sudo $PYTHON -m pip install jmespath==1.0.1
sudo $PYTHON -m pip install mpmath==1.3.0
sudo $PYTHON -m pip install numpy==1.26.0
sudo $PYTHON -m pip install packaging==23.1
sudo $PYTHON -m pip install pandas==1.5.3
sudo $PYTHON -m pip install parameterized==0.7.5
sudo $PYTHON -m pip install pluggy==1.2.0
sudo $PYTHON -m pip install py4j==0.10.9.7
sudo $PYTHON -m pip install pyarrow==14.0.1
sudo $PYTHON -m pip install pyspark[sql]==3.5.0
sudo $PYTHON -m pip install pytest==7.4.0
sudo $PYTHON -m pip install python-dateutil==2.8.2
sudo $PYTHON -m pip install pytz==2023.3
sudo $PYTHON -m pip install randomgen==1.26.0
sudo $PYTHON -m pip install s3transfer==0.6.2
sudo $PYTHON -m pip install scipy==1.11.3
sudo $PYTHON -m pip install six==1.16.0
sudo $PYTHON -m pip install smart-open==5.2.1
sudo $PYTHON -m pip install sympy==1.9
sudo $PYTHON -m pip install typeguard==2.12.1
sudo $PYTHON -m pip install typing-extensions==4.7.1
sudo $PYTHON -m pip install urllib3==1.26.16
