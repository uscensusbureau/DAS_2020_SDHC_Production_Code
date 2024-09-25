#!/usr/bin/env bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $SCRIPT_DIR/../tmlt/phsafe/

# Run PHSafe in non-private mode on toy_dataset with validation.

spark-submit \
    --properties-file resources/spark_configs/spark_local_properties.conf \
    ph-safe.py non-private resources/toy_dataset/config_puredp.json \
    resources/toy_dataset example_output/non_private -l \
    example_output/non_private/phsafe_toydataset.log --validate
