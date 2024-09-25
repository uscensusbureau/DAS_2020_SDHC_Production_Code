#!/usr/bin/env bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $SCRIPT_DIR/../../../../tmlt/phsafe/

spark-submit \
    --properties-file resources/spark_configs/spark_local_properties.conf \
    ph-safe.py non-private \
    $SCRIPT_DIR/config.json \
    $SCRIPT_DIR \
    $SCRIPT_DIR/outputs \
    --log $SCRIPT_DIR/test_log.log
