#!/bin/bash

$SPARK_HOME/sbin/start-master.sh

#export MASTER=spark://limmen:7077
export MASTER=spark://limmen-MS-7823:7077
export SPARK_WORKER_INSTANCES=4
export CORES_PER_WORKER=2
export TOTAL_CORES=8

$SPARK_HOME/sbin/start-slave.sh $MASTER --cores 2 --memory 3g

firefox http://127.0.0.1:8080 &