#!/bin/bash/
##############################################################################
# Summary: This script fits models to state- and national-level noisy
# measurements of quantities from the Census Demographic and Housing
# Characteristics File (S-DHC) quantities. "The S-DHC tables reflect
# especially complex relationships between the characteristics about
# households and the people living in them" (census.gov, About 2020 Census
# Data Products). Running this script uses the configuration settings in
# config_global.json and PH[1-8]/config.json to generate model-based
# estimates of S-DHC quantities. Results are saved in each respective
# PH[1-8]/estimates directory.
#
# Usage: source run_models.sh
# Return: NULL
#
# Author: ${DEVELOPER} 
# Last Updated: 10 July 2023
#
# Modified:
#    14 July 2023, ${DEVELOPER} 
#    Line 33, PH${i} to ph${i} in order to match subdirectory names
##############################################################################

echo "Make log subdirectory, if it doesn't exist."
if [ ! -d ./log ] ; then
        mkdir ./log
        echo "./log created"
fi

echo "Check for missing packages, and install them."
Rscript src/package_check.R > log/packages.out 2>&1

echo "Make data subdirectory, if it doesn't exist."
if [ ! -d ./data ] ; then
                mkdir ./data
                        echo "./data created"
fi

echo "Specifying FIPS codes"
Rscript src/state_fips.R > log/fips.out 2>&1

echo "Starting model fitting"
for i in {1..8}
do
    mkdir -p PH${i}/estimates
    Rscript PH${i}/tnormal_ph${i}.R > log/ph${i}.out 2>&1 &
done
# wait until all models complete
wait

# Grab the logs and direct them into the primary stdout
echo '*********START OF FIPS LOGS*********'
cat log/fips.out
wait
echo '**********END OF FIPS LOGS**********'
echo '*******START OF PACKAGES LOGS*******'
cat log/packages.out
wait
echo '********END OF PACKAGES LOGS********'

# Check if any of the primary logs have errors as well
all_passed=0

for i in {1..8}
do
    echo "********START OF PH${i} LOGS********"
    cat log/ph${i}.out
    wait
    echo "*********END OF PH${i} LOGS*********"
    if ! grep -q "Completed all geographies" log/ph${i}.out; then
        all_passed=1
    fi
done
echo '**********END OF FIPS LOGS**********'


# Generate summary of model fit performance
echo "Generating model fit summary"
Rscript summary/summarize_results.R > log/summary.out 2>&1
echo '********START OF SUMMARY LOGS********'
cat log/summary.out
echo '*********END OF SUMMARY LOGS*********'

# Fail the program if an error was detected
if [ $all_passed -eq 1 ]; then
    echo "Errors detected in one or more R scripts!"
    exit 1
fi

echo "Model fitting complete"

echo "Validating Modeling"
Rscript validate_modeling.R
echo "Modeling Validation Complete"
