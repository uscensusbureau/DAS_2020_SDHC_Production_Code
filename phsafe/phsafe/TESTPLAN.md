# PHSafe Test Plan

PHSafe and supporting libraries provide a range of tests to ensure that the software is working correctly.

Our tests are divided into unit and system tests. Unit tests verify that the implementation of a class, and its associated methods, match the behavior specified in its documentation. System tests are designed to determine whether the assembled system meets its specifications.

We also divide tests into fast and slow tests. Fast tests complete relatively quickly, and can be run often, while slow tests are longer-running and less frequently exercised. While unit tests tend to be fast and system tests tend to be slow, there are some slow unit tests and fast system tests.

All tests are run using pytest. Core is provided as a binary wheel, and thus does not have runnable tests in this release.

All tests are run on a single machine. Runtimes mentioned in this document were measured on an r4.16xlarge machine.

All test commands in this file should be run from the repository root, unless otherwise noted.

Execute the following to run the tests:

*Fast Tests:*

```bash
python3.11 -m pytest common -m 'not slow'
python3.11 -m pytest analytics -m "not slow"
python3.11 -m pytest phsafe -m 'not slow'
# (Runtime Estimate: 35 minutes)
```

*Slow Tests:*

Slow tests are tests that we run less frequently because they take a long time to run, or the functionality has been tested by other fast tests.

```bash
python3.11 -m pytest common -m 'slow' --suppress-no-test-exit-code
python3.11 -m pytest analytics -m 'slow'
python3.11 -m pytest phsafe -m 'slow'
# (Runtime estimate: 1 hour)
```

Note: Common has no slow tests, and pytest treats finding no tests as an error. The `suppress-no-test-exit-code` flag will prevent the error, but requires the `pytest-custom-exit-code` package be installed. If you do not have it, you can remove the flag and ignore the error.

##### **Accuracy Test**:

   * Run `examples/run_phsafe_error_report.sh` to compare the results from PHSafe algorithm execution against the ground truth (non-private) answers. This example script runs the PHSafe program on non-sensitive data present in input path (`tmlt/phsafe/resources/toy_dataset`) using `config_zcdp.json`. The aggregated error report is saved to output directory (`example_error_report/`).

```bash
bash phsafe/examples/run_phsafe_error_report.sh
# (Runtime estimate: 5 minutes)
```

Note: Error report uses the ground truth counts. It violates differential privacy, and should not be created using sensitive data. Its purpose is to test PHSafe on non-sensitive or synthetic datasets to help tune the algorithms and to predict the performance on the private data.
