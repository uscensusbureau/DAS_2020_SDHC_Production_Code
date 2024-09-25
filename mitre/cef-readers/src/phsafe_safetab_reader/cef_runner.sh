#!/bin/bash

# zip the code
ZIPFILE=das_decennial.zip
export ZIPFILE

# cd up into das_decennial root
cd ..

# make zip in das_decennial
zip -r -q $ZIPFILE . -i '*.py' '*.sh' '*.ini'
echo $ZIPFILE

# cd back into reader directory
cd -

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
	--files "../log4j.configuration.xml"
	--files "../log4j.properties"
	--master yarn
	--py-files "../das_decennial.zip"
	--name $JBID:convert_measurements"
python_cmd="cef_reader.py"
echo
echo
echo $spark_submit $python_cmd
echo
extra_java_options="-XX:+PrintGCTimeStamps -XX:+PrintGCDetails -verbose:gc"

# Send the code to spark submit and call convert_pickled.python
$spark_submit --conf "spark.executor.extraJavaOptions=$extra_java_options" $python_cmd
