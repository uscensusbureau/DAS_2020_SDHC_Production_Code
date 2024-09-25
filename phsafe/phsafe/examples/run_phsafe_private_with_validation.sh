#!/usr/bin/env bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $SCRIPT_DIR/../tmlt/phsafe/

# Run PHSafe in private mode using puredp on toy_dataset with input  and output validation.

spark-submit \
    --properties-file resources/spark_configs/spark_local_properties.conf \
    ph-safe.py private resources/toy_dataset/config_puredp.json \
    resources/toy_dataset example_output/private_with_validation -l \
    example_output/private_with_validation/phsafe_toydataset.log --validate --validate-private-output
