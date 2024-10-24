# das-phsafe-cef-reader
Safetab CEF Reader

## Tests
Tests are included in the tests folder.  Run, from the root directory:
```
./tests/run_safetab_tests.sh
./tests/run_phsafe_tests.sh
```
Note that the tests must be run on a system with Spark.

### Versioning

das-phsafe-cef-reader versioning uses a 4 digit semantic versioning convention:
```
<major>.<minor>.<patch>.<build>
```
A change to the major version number indicates a significant change in functionality that is not backwards compatible
with previous versions.

A change to the minor version number indicates a backwards compatible change to functionality.

A change in the patch version indicates a minor bug fix or documentation change.

A change in the build version indicates a packaging related fix that does not change functionality.

#### Version Updates
The version of the package is managed in ```__init__.py```.

Update the MAJOR, MINOR and PATCH versions as appropriate and commit the change.

Note:  Whenever a higher level value is updated, reset the lower to 0. If you increase the MAJOR version, set MINOR,
PATCH and BUILD to 0.

## Package Information

This is used for the SafeTab and PHSafe CEF Readers, which convert the CEF microdata files into Spark Dataframes that
SafeTab and PHSafe use as inputs.

SafeTab Documentation for the format of its inputs is here:
[PHSafe Documentation](../../phsafe/phsafe/PHSafe_Documentation.pdf)

### Important Files:

**safetab_cef_config.ini** - Contains the locations of the 2020 CEF microdata files on s3 which are the inputs for the
CEF Reader

**safetab_cef_config_2010.ini** - Used for safetab P Cef Reader (safetab_cef_reader). It contains the locations of the
2010 CEF microdata files on s3 which are the inputs for the CEF Reader

**safetab_h_cef_config_2010.ini** - Used for safetab_h_cef_reader. It contains the locations of the 2010 CEF microdata
files on s3 which are the inputs for the CEF Reader

**safetab_cef_reader.py** - Reads the fixed-width CEF files in and uses the CEF Validator to turn them into dataframes.
Does additional modifications to get dataframe to SafeTab's expected input.

**safetab_h_cef_reader.py** - Reads fixed-width CEF files and the outputs T1 files from
Safetab-H, and uses the CEF Validator to turn them into dataframes. Does additional modifications to get dataframe
to SafeTab's expected input.

**safetab_p_cef_reader.py** - Reads fixed-width CEF files and the outputs T1 files from
Safetab-P, and uses the CEF Validator to turn them into dataframes. Does additional modifications to get dataframe
to SafeTab's expected input.

**safetab_p_cef_runner.sh** - Used to run the CEF Reader (Safetab-P)

**safetab_h_cef_runner.sh** - Used to run the CEF Reader (Safetab-H)

**cef_runner.sh** - used to run the PHSafe CEF reader

**cef_reader.py** - Reads the fixed-width CEF files in and uses the CEF Validator to turn them into dataframes.
Does additional modifications to get dataframe to PHSafe's expected input.

**cef_config.ini** - Inputs input file locations used by PHSafe CEF reader

**cef_validator_classes.py** A dependency for some of the python files, originally from the reader in das_decennial's
programs/reader directory.
