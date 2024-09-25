#!/usr/bin/env bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $SCRIPT_DIR/../tmlt/phsafe/

# Run PHSafe in private mode using zcdp on toy_dataset without validation.

spark-submit \
    --properties-file resources/spark_configs/spark_local_properties.conf \
    ph-safe.py private resources/toy_dataset/config_zcdp.json \
    resources/toy_dataset example_output/private_zcdp -l \
    example_output/private_zcdp/phsafe_toydataset.log
    