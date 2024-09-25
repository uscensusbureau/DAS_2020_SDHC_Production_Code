#!/bin/bash

# zip the code
ZIPFILE=das_phsafe_cef_reader.zip
export ZIPFILE

# make zip
zip -r -q $ZIPFILE . -i '*.py' '*.sh' '*.ini'
echo $ZIPFILE

spark_submit="spark-submit
	--conf spark.local.dir=/mnt/tmp/
    	--conf spark.eventLog.enabled=true
    	--conf spark.eventLog.dir=/mnt/tmp/logs/
    	--conf spark.submit.deployMode=client
	--conf spark.network.timeout=3000
	--conf spark.driver.extraJavaOptions=-Dlog4j.configuration=file:log4j.configuration.xml
	--conf spark.executor.extraJavaOptions=-Dlog4j.configuration=file:log4j.configuration.xml
	--conf spark.driver.extraJavaOptions=-Dlog4j.configuration=file:log4j.properties
	--conf spark.executor.extraJavaOptions=-Dlog4j.configuration=file:log4j.properties
	--files log4j.configuration.xml
	--files log4j.properties
	--master yarn
	--py-files das_phsafe_cef_reader.zip
	--name $JBID:convert_measurements"
python_cmd="safetab_h_cef_reader.py"
echo
echo
echo $spark_submit $python_cmd
echo
extra_java_options="-XX:+PrintGCTimeStamps -XX:+PrintGCDetails -verbose:gc"

# Send the code to spark submit and call convert_pickled.python
$spark_submit --conf "spark.executor.extraJavaOptions=$extra_java_options" $python_cmd
