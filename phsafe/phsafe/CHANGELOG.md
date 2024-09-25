# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## 3.0.0 - 2024-04-03
### Fixed
- Removed duplicate MAFID from the toy dataset.

### Changed
- Minimum Python version is now 3.11.
- Added a note to the README about the harmless end-of-run error.
- Re-named `noise_scale` parameter to avoid confusion with similar name in Core.
- Input validation will now fail if two records in the units or geos files have the same MAFID.
- Switched from using a flat_map to add iterations to a map and a filter.
- Switched expected variance formulas to use discrete laplace instead of continuous approximation.
- Re-organized instructions to make them easier to follow.
- Added a series of enumerations for common column values and used them throughout the tabulations.
- PHSafe now exits with a non-zero code when validation fails.
- Updated to tmlt.analytics 0.8.3 and tmlt.core 0.12.0 which incorporates an improved truncation hash function and Core documentation.

## 2.0.2 - 2023-12-15
### Changed
- Supports Python 3.11 and PySpark 3.4.0.
- Now requires PyArrow 14.0.1 to address CVE-2023-47248.

## 2.0.1 - 2023-09-01
### Changed
- Updated to tmlt.common 0.8.4, which includes the required setup.py file.

## 2.0.0 - 2023-08-31
### Changed
- PHSafe no longer tabulates County, Tract, Place, or AIANNH regions.
- Removed the suggestion to use extraLibraryPath options on EMR.
- Updated to tmlt.common 0.8.3.
- Modified the readers to remove uses of the mock CEF reader.
- Modified `repo_zip.sh` to no longer include the mock CEF reader by default.
- Updated the documentation file structure to match SafeTab-P's example.
- Rewrite out of date error report, and update expected error formulas.
- Updated to tmlt.analytics 0.8.0 and tmlt.core 0.11.0.
- Modified the bootstrap_script.sh script in the core directory to read the whl file from the provided argument.

## 1.0.2 - 2022-12-22
### Fixed
- Fixed `test/system/test_end_to_end.py` to run properly on a spark cluster.

## 1.0.1 - 2022-12-08
### Fixed
- Fixed `examples/repo_zip_pih.sh` to work with the production cef reader.

## 1.0.0 - 2022-12-02
### Changed
- Updated tmlt.core and switched to a wheel install process. 
- Updated tmlt.analytics.
- Removed code "99999" from the output for PLACE.
- PHSafe Table Shell names were adjusted. (Ex. p17 is now PH1)

### Fixed
- Fixed a bug in phsafe output validation that caused spurious validation failures when different outputs were assigned different budgets.
- Fix PH1_denom tabulations so they do not require calculating PH1_num.
- Adjusted Geography Preprocessing to remove group quarters only geographies. 

## 0.4.1 - 2022-09-09
### Fixed
- Updated tmlt.core and tmlt.analytics to fix a bug where queries failed to evaluate. 

## 0.4.0 - 2022-07-27
### Changed
- Changed phsafe to use the new `groupby` function instead of `groupby_domains`.
- Update list of valid FIPS codes for a run in `state_filter` check. Ensure PR and US runs are independent.
- Added noise scale and noise mechanism to every statistic output by phsafe private algorithm.
- Add DP program output validation logic and `--validate-private-output` flag to `private mode` in CLI.

### Fixed
- Fixed pattern for COUNT column in output validation config to include negative values.
- Fixed type of *_DATA_CELL column in private and non-private outputs

## 0.3.0 - 2022-05-05
### Changed
- Updated truncation method for persons in households to hash.
- Updated public join to new `QueryBuilder.join_public` interface.
- Private tabulations updated to use single query evaluate of the Safetables `Session` interface rather than the deprecated multi-query variant.
- Added privacy budget checks to avoid building queries for tabulations disabled in the config.
- Updated uses of `PrivacyBudget` to the new interface.
- Updated uses of `QueryBuilder.groupby` to use the newly-named `QueryBuilder.groupby_domains`.
- Added support for additional geos. PHSafe now supports following geos: USA, State, County, Tract, Block_Group, Place and AIANNH.
- Removed session-level noise param usage and specified the mechanism used in each query.
- Updated dependencies to use the renamed `tmlt.analytics` package.
- Changed some silent asserts to human-readable errors
- Changed `p37_num` and `h12_num` to be computed as post-processing.
- Moved examples and benchmarks into `examples/` and `benchmark/` directories, respectively, from `phsafe_examples` and `phsafe_benchmark`.
- Updated private joins to use new `TruncationStrategy` types from Analytics.
- Updated dependencies to use the renamed `tmlt.core` package.
- Updated Tumult library name (was: "SafeTables", now: "Analytics")
- Updated the private and non-private algorithm to collapse cell 11 into cell 2 for P31 table shell.
- Added ability to assign privacy budget to each population group (iteration level/geo level).
- Add repartition to write output as single part csv than multiple part files per tabulation.

### Fixed
- FINAL_POP input domains for input validation updated to [0, 100000] so that 99999 is included.

### Documentation
- Enabled building documentation for previous releases of the package.
- Future documentation will include any exceptions defined in this library.

## 0.2.0 - 2021-09-10
### Added
- Private and nonprivate tabulations for P36, P37, P40, H11, and H12 tables.
- Added `privacy_defn` flag for switching between PureDP and Rho zCDP to the config.

### Changed
- Namespaced packages as tmlt and relocated resources folder inside package.
- Epsilons in config are changed to privacy_budgets.
- Updated private tabulations to run on Safetables `Session` interface.

### Fixed
- MAFID input domains for input validation updated to [100000001, 900000000] so that 899999999 is included.

## 0.1.1 - 2021-04-27
### Fixed
- Add certain properties to spark config files.
- Add tearDown step to some tests.

## 0.1.0 - 2021-03-31
### Added
- A private mode prototype of the PHSafe executor for P17, P30, and P31 tabulations
  in local and cluster mode.

## 0.1.0-alpha - 2021-03-05
### Added
- PHSafe command line interface to validate input files and produce non-private P17, P30 and P31 tabulations in local and cluster mode.
