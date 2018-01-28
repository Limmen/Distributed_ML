#!/bin/bash

export MASTER=spark://limmen:7077
export SPARK_WORKER_INSTANCES=1
export CORES_PER_WORKER=1
export TOTAL_CORES=4
export SPARK_EXECUTOR_MEMORY=15360
export SPARK_DRIVER_MEMORY=15360

$SPARK_HOME/bin/spark-submit \
--master $MASTER \
--driver-memory 10G \
--executor-memory 10G \
--conf spark.driver.maxResultSize=9G \
--total-executor-cores 2 \
/home/kim/workspace/python/har_test/data_setup.py