#!/usr/bin/env bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $SCRIPT_DIR/../tmlt/phsafe/

# Run PHSafe error report
spark-submit \
    --properties-file resources/spark_configs/spark_local_properties.conf \
    accuracy_report.py \
        "resources/toy_dataset/config_zcdp.json" \
        "resources/toy_dataset/" \
        "example_error_report/"
