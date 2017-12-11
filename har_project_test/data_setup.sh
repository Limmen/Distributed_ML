#!/bin/bash

#export MASTER=spark://limmen:7077
export MASTER=spark://limmen-MS-7823:7077
$SPARK_HOME/bin/spark-submit \
--master $MASTER \
/media/limmen/HDD/workspace/python/har_project_test/har_data_setup.py \
--cluster