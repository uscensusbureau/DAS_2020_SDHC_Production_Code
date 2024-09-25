#!/usr/bin/env bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $SCRIPT_DIR/../tmlt/phsafe/

# Run PHSafe in validate mode on toy_dataset.

spark-submit \
    --properties-file resources/spark_configs/spark_local_properties.conf \
    ph-safe.py validate resources/toy_dataset/config_puredp.json \
    resources/toy_dataset -l \
    example_output/validation/phsafe_toydataset.log