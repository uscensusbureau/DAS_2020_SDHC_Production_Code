# PHSafe for Supplemental DHC

PHSafe is a differentially private algorithm to facilitate the publication by Census of tables of statistics (counts, means) for the universe of population in households at varying levels of geography (national and state). PHSafe ensures privacy of the tabulations by adding noise to the statistics.

SPDX-License-Identifier: Apache-2.0
Copyright 2024 Tumult Labs

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

## Overview

PHSafe produces differentially private tables of statistics (counts, means) regarding populations in households grouped by demographic characteristics and detailed race and tribe characteristics. These tables create the Supplemental Demographic and Housing Characteristics File (S-DHC) at a fixed number of geography levels (national, state).

More information about the PHSafe algorithm can be found in the [PHSafe specifications document](PHSafe_Documentation.pdf), which describes the problem, general approach, and (in Appendix A), the input and output file formats.

See [CHANGELOG](CHANGELOG.md) for version number information and changes from past versions.

## System Requirements

PHSafe is designed to run on a Linux machine with Python 3.11 and PySpark 3. It can be run either locally on a single machine, or on a Spark cluster. We have developed and tested with Amazon EMR 6.12 in mind. When running on a nation-scale dataset, we recommend using an EMR cluster having at least 1 master and 2 core nodes of instance type: r4.16xlarge or higher.

## Installing and Running PHSafe locally

These instructions assume that the default Python3 version is Python 3.11. Once Python 3.11 is installed, you can add the following to your `.bashrc` to make it the default version:

```bash
alias python3=python3.11
```

### 1. Installing Dependencies
First, make sure that you have Java 8 or later with JAVA_HOME properly set (you can check this by runing `java -version`). If Java is not yet installed on your system, you can install [OpenJDK 8](https://openjdk.org/install/) (installation will vary based on the system type).

All python dependencies, as specified in [requirements.txt](requirements.txt), must be installed and available on the PYTHONPATH.

When running locally, dependencies can be installed by running:
```bash
sudo python3 -m pip install -r phsafe/requirements.txt
```

### 2. Installing Tumult Core 

PHSafe also requires the Tumult Core library to be installed. Tumult Core can either be installed from the wheel file provided with this repository, or from PyPI (like external dependencies in the previous step). Users like the Census who prefer to avoid installing packages from PyPI will likely prefer installing from a wheel. Users who do not have such concerns will likely find it easier to install from PyPI.

#### Wheel installation (Linux only)

Tumult Core can be installed by calling:

```bash
sudo python3 -m pip install --no-deps core/tmlt_core-0.12.0-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
```

This will only work on a Linux machine, as the provided wheel is Linux-only. All other operating systems should install from PyPI.

#### Pip installation

Tumult Core can also be installed from PyPI:

```sudo python3 -m pip install tmlt.core==0.12.0```

### 3. Setting up your environment

All machines require the following PYTHONPATH updates, which install the Tumult source directories. This assumes that required python dependencies have been installed.

```bash
DIR=<path of cloned tumult repository>
export PYTHONPATH=$PYTHONPATH:$DIR/phsafe
export PYTHONPATH=$PYTHONPATH:$DIR/common
export PYTHONPATH=$PYTHONPATH:$DIR/analytics
```

If you're planning on utilizing the MITRE Census Edited File (CEF) reader, the `PYTHONPATH` needs to be updated to include the CEF package. This should point to directory containing the  `phsafe_safetab_reader` package.

```bash
export PYTHONPATH=<path to directory containing phsafe_safetab_reader>:$PYTHONPATH
```

If using MITRE's CEF reader, you will also need to include its dependencies. Consult the CEF reader README for more details.

### 4. Running PHSafe

#### Command-line interface

The primary command line interface is driven by `tmlt/phsafe/ph-safe.py`. The PHSafe command line program supports several subcommands.

*Navigate to the directory `tmlt/phsafe`.*

To view the list of available subcommands on console, enter:

```bash
./ph-safe.py -h
```

To view the arguments for running a given subcommand, enter:

```bash
./ph-safe.py <subcommand> -h
```

The following subcommands are supported:
- `validate` validates the input data files against the input specification and reports any discrepancies.  Validation errors are written to the user-specified log file.
- `private` runs the private algorithm to produce tabulations. The output files are saved to the output folder specified on the command line.  See [Appendix A](PHSafe_Documentation.pdf) for output file specification.  If the `--validate` flag is added, the input will be validated before the algorithm is run. If the `--validate-private-output` flag is added, the outputs will be validated after the DP algorithm is run.
- `non-private` runs the non-private algorithm to produce tabulations. The output files are saved to the output folder specified on the command line.  See [Appendix A](PHSafe_Documentation.pdf) for output file specification.  If the `--validate` flag is added, the input will be validated before the algorithm is run.

The shell script `examples/run_phsafe_non_private_with_validation.sh` demonstrates running the PHSafe command line program in non-private mode with validation and a csv reader.  An excerpt is shown here with comments:

```bash
./ph-safe.py non-private \             # run the non-private algorithm
resources/toy_dataset/config_puredp.json \    # this config specifies a CSV reader
resources/toy_dataset \                # the data_path (see note below)
example_output/non_private \           # desired output location
-l example_output/non_private/phsafe_toydataset.log \  # desired log location
--validate                              # validate input before executing algorithm
```

The shell script `examples/run_phsafe_private_with_validation.sh` demonstrates running the PHSafe command line program in private mode with validation and a csv reader.  An excerpt is shown here with comments:

```bash
./ph-safe.py private \                 # run the private algorithm
resources/toy_dataset/config_puredp.json \    # this config specifies a CSV reader
resources/toy_dataset \                # the data_path (see note below)
example_output/private \               # desired output location
-l example_output/private/phsafe_toydataset.log \  # desired log location
--validate \                              # validate input before executing algorithm
--validate-private-output                 # validate output after executing algorithm
```

To run only input validation, use `spark-submit` as shown below.

```bash
spark-submit \
    --properties-file resources/spark_configs/spark_local_properties.conf \
    ./ph-safe.py validate \
    <path to unzipped data folder>/phsafe-300m-input/config_<puredp or zcdp>.json \
    <path to reader configuration> \
    -l <log_file_name>
```

Notes:

If using a CSV reader, then the `data_path` argument is the path to the directory containing the input CSV files. If using the MITRE CEF reader, this should point to the MITRE CEF reader config file.

If using the CSV reader and Tumult's sample dataset, replace the `<path to reader configuration>` with the `<path to unzipped data folder>/phsafe-300m-input`.

Input and output directories must be locations on the local machine (not S3).
A sample Spark custom properties file for local mode execution is located in `tmlt/phsafe/resources/spark_configs/spark_local_properties.conf`.
While Spark properties are often specific to the environment (number of cores,
memory allocation, etc.), we strongly recommend that the `spark.sql.execution.arrow.enabled`
be set to `true` as in the example config file for local mode.
When pyarrow is enabled, the data exchange between Python and Java is much faster and results
in orders-of-magnitude differences in runtime performance.

If Python 3.11 isn't the default Python version, you can point Spark to it by adding this configuration arguement `--conf spark.pyspark.python=<path to Python3.11>`.

See `examples` for examples of other features of the PHSafe command line program.

## Installing and Running PHSafe on an EMR Cluster

### 1. Dependencies and Tumult Core
Before you can run PHSafe, you must install all python dependencies specified in [requirements.txt](requirements.txt), plus Tumult Core.

We have designed with EMR 6.12 in mind, which does not come with Python 3.11. Therefore, you will need to install Python 3.11, and reinstall PySpark. This can be done with a [bootstrap action](https://docs.aws.amazon.com/emr/latest/ManagementGuide/emr-plan-bootstrap.html). We include sample bootstrap scripts that install Python 3.11, the public dependencies, and (optionally), Tumult Core. Bootstrap actions must be configured when starting a cluster, so start a new cluster with the bootstrap actions specified in this section (we recomend at least 1 master and 2 core nodes of instance type: r4.16xlarge or higher).

Tumult Core can either be installed from the wheel file provided with this repository, or from PyPI (like the other dependencies). Users like the Census who prefer to avoid installing packages from PyPI will likely prefer installing from a wheel. Users who do not have such concerns will likely find it easier to install from PyPI.

If you would like to install Tumult Core from PyPi, use [bootstrap_with_core.sh](phsafe/tmlt/phsafe/resources/installation/bootstrap_with_core.sh). Upload the script to an S3 bucket, create a new bootstrap action, and provide the s3 location of the script as its script location.

If you would like to install Tumult Core from the provided wheel, use [bootstrap_without_core.sh](phsafe/tmlt/phsafe/resources/installation/bootstrap_without_core.sh). Upload that script, [`core/bootstrap_script.sh`](../core/bootstrap_script.sh), and [`core/tmlt_core-0.12.0-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl`](../core/tmlt_core-0.12.0-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl) to an S3 bucket. Then, add the following bootstrap steps (in this order):
1. Set the script location to the S3 path where you uploaded `bootstrap_without_core.sh`.
1. Set the script location to the S3 path where you uploaded `core/bootstrap_script.sh`. Set the Optional Argument to the S3 path where you uploaded the wheel file.


### 2. Running PHSafe

Once your cluster has started, you can use the AWS Management Console to configure a step that will invoke the `spark-submit` command.  

#### Uploading PHSafe

There are three important preconditions: 

1. All of input files must be uploaded to S3.
1. A zip file containing the repository source must be created and placed in S3.
1. The main driver python program, [`phsafe/tmlt/phsafe/ph-safe.py`](phsafe/tmlt/phsafe/ph-safe.py), must be placed in S3.

The zip file can be created using the following command, which creates a packaged repository `repo.zip` that contains Tumult’s products and MITRE’s CEF reader - if a path to the CEF reader is provided. The zip file will not contain any other dependencies.

```bash
bash <path to cloned tumult repo>/phsafe/examples/repo_zip.sh \
-t <path to cloned tumult repo> [-r <optional path to directory containing cef_reader>]
```

Note: The `repo_zip.sh` script has a dependency on associative arrays and works with bash version 4.0 or newer.

#### <a id="steps"></a>Steps:

Once you've uploaded the zip file, the driver program, and the input files, you can [add a step](https://docs.aws.amazon.com/emr/latest/ManagementGuide/emr-work-with-steps.html) to your cluster:

1. In the [Amazon EMR console](https://console.aws.amazon.com/elasticmapreduce), on the Cluster List page, select the link for your cluster.
1. On the `Cluster Details` page, choose the `Steps` tab.
1. On the `Steps` tab, choose `Add step`.
1. Type appropriate values in the fields in the `Add Step` dialog, and then choose `Add`. Here are the sample values:

                Step type: Custom JAR

                Name: <any name that you want>

                JAR location: command-runner.jar

                Arguments:
                        spark-submit
                        --deploy-mode client --master yarn
                        --conf spark.pyspark.python=/usr/local/bin/python3.11
                        --conf spark.driver.maxResultSize=20g
                        --conf spark.sql.execution.arrow.enabled=true
                        --py-files s3://<s3 repo.zip path>/repo.zip
                        s3://<s3 phsafe main file path>/ph-safe.py
                        private
                        s3://<s3 config file path>
                        s3://<s3 data_path>
                        s3://<s3 output directory>
                        --log s3://<s3 output directory>/<log file path>
                        --validate
                        --validate-private-output

                Action on Failure: Cancel and wait

Notes:
- Output locations must correspond to locations on S3.
- Since Python 3.11 is not the default python version on an EMR cluster, you must manually install it and point PySpark to it (using the `spark.pyspark.python` configuration option) for your step. This example step assumes that Python 3.11 has been installed at `/usr/local/bin/python3.11`, which is where our sample bootstrap script installs it.
- You can run PHSafe this way on either the sample input files or the toy dataset. You can also run the program non-privately by replacing `private` with `non-private` and removing the `--validate-private-output` flag.
- Similar instructions can be used for the `validate` subcommand with some changes to the command - replacing `private` with `validate` and removing the output directory argument, `--validate` and `--validate-private-output` flags.

#### Spark properties

While Spark properties are often specific to the environment (number of cores, memory allocation, etc.), we strongly recommend that the `spark.sql.execution.arrow.enabled` be set to `true`. An example custom Spark properties config file for cluster mode is located in [`phsafe/tmlt/phsafe/resources/spark_configs/spark_cluster_properties.conf`](tmlt/phsafe/resources/spark_configs/spark_cluster_properties.conf).

A properties file must be located on the local machine (we recommend using a bootstrap action to accomplish this), and can be specified by adding the `--properties-file` option to `spark-submit` in the step specification.

## Testing

*See [TESTPLAN](TESTPLAN.md)*

## Known Warnings

These warnings can be safely ignored:

1. PyTest warning:

```
PytestUnknownMarkWarning: Unknown pytest.mark.slow - is this a typo?  You can register custom marks to avoid this warning - for details, see https://docs.pytest.org/en/stable/how-to/mark.html
    @pytest.mark.slow
```


2. In order to prevent the following warning:

```
WARN NativeCodeLoader: Unable to load native-hadoop library for your platform
```

`LD_LIBRARY_PATH` must be set correctly. Use the following:

```bash
export LD_LIBRARY_PATH=/usr/lib/hadoop/lib/native/
```

If `HADOOP_HOME` is set correctly (usually `/usr/lib/hadoop`), this may be replaced with

```bash
export LD_LIBRARY_PATH=$HADOOP_HOME/lib/native/
```

3. Spurius PySpark Unset Warning:
```
Did you install PySpark via a package manager such as pip or Conda? If so,
PySpark was not found in your Python environment. It is possible your
Python environment does not properly bind with your package manager.

Please check your default 'python' and if you set PYSPARK_PYTHON and/or
PYSPARK_DRIVER_PYTHON environment variables, and see if you can import
PySpark, for example, 'python -c 'import pyspark'.

If you cannot import, you can install by using the Python executable directly,
for example, 'python -m pip install pyspark [--user]'. Otherwise, you can also
explicitly set the Python executable, that has PySpark installed, to
PYSPARK_PYTHON or PYSPARK_DRIVER_PYTHON environment variables, for example,
'PYSPARK_PYTHON=python3 pyspark'.
```

This warning can appear when the `spark.pyspark.python` configuration option or the `PYSPARK_PYTHON` is set correctly and the program completes successfully. However, if this warning appears and the program does not complete, correctly configuring one of these options is probably required.

4. Other known warnings:

```
FutureWarning: elementwise comparison failed; returning scalar instead, but in the future will perform elementwise
comparison
```
```
UserWarning: In Python 3.6+ and Spark 3.0+, it is preferred to specify type hints for pandas UDF instead of specifying
pandas UDF type which will be deprecated in the future releases. See SPARK-28264 for more details.
```
```
UserWarning: It is preferred to use 'applyInPandas' over this API. This API will be deprecated in the future releases.
See SPARK-28264 for more details.
```

### Known Errors

SafeTab-H occasionally produces a `ValueError: I/O operation on closed file` about attempting to write to the logs file after all logs are written. This error can be safely ignored, as all log records should still be written.

