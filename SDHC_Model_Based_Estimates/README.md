Ratio Models for S-DHC Tables PH1 - PH8
===

This repository generates model-based estimates of complex household relationships from differentially private noisy measurements.
Models are implemented with R statistical software.

Estimates for all tables are generated in parallel with the following command.

> `$ source run_models.sh`

The `run_models.sh` script starts by checking that all required R packages are installed.
If certain packages are missing, the script will install them automatically.
Model-based estimates for each table are then simulated and saved in the `estimates` directory corresponding to the table name.
A log for each model is generated and saved in the log directory.
Model configuration parameters common to all tables are specified in the file `config_global.json`.
Configuration parameters specific to a particular table are specified by a table-specific file, e.g., `PH1/config.json`.

Let's take table PH1 as an example.
When `run_models.sh` is invoked, the log file `log/ph1.out` is created and captures all messages from the script `PH1/tnormal_ph1.R`.
Messages in the log file show what percentage of the PH1 table cells have been estimated for several geographies: all 50 states and the District of Columbia, Puerto Rico, and the nation as a whole.
Files with the model-based estimates for each geography is saved in `PH1/estimates`.

Model-based estimates can be generated for a single table can be generated for PH1 (or any other table) by navigating to the PH1 directory and the following command.

> `PH1$ Rscript tnormal_ph1.R`

This command will write model-based estimates to `PH1/estimates` and write logging messages to the user's screen as standard output.

Project Organization
--------
```
├── config_global.json          <- Configuration settings common to all tables
├── data                        <- Input data generated for use across one or more models
├── log                         <- All log files
├── PH[1-8]                     <- Table-specific functions, estimates, and configuration
│   ├── config.json             <- Configuration settings for individual tables
│   ├── estimates               <- Directory of estimates for states+DC, Puerto Rico, and Nation
│   └── tnormal_ph[1-8].R       <- Codes to generate model-based estimates for each table
├── README.md                   <- Top-level README for project
├── run_models.sh               <- Script to run all models in parallel
├── src                         <- Directory containing scripts used by all tables
│   ├── package_check.R         <- Automated package check and install
│   └── state_fips.R            <- Generate fips codes to define geographies used in modeling
├── summary                     <- Directory of scripts to compile summary of model results
│   ├── out_count.tex           <- Table file generated by `summarize_results.R`
│   ├── out_ratio.tex           <- Table file generated by `summarize_results.R`
│   ├── summarize_results.R     <- Script to compile model performance summary
│   └── summary.tex             <- Report with model specification and performance summary
└── unit_test                   <- Directory of scripts to complete unit tests
    ├── individual_table.R      <- Complete unit tests for individual entries table-by-table
    └── summary_table.R         <- Complete unit tests for summary-level metrics
```
--------
