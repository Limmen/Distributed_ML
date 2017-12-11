#!/bin/bash

#export MASTER=spark://limmen:7077
export MASTER=spark://limmen-MS-7823:7077
export SPARK_WORKER_INSTANCES=4
export CORES_PER_WORKER=2
export TOTAL_CORES=8

$SPARK_HOME/bin/spark-submit \
--master $MASTER \
--py-files /media/limmen/HDD/workspace/python/har_project_test/har_dist_model.py \
--conf spark.cores.max=$TOTAL_CORES \
--conf spark.task.cpus=$CORES_PER_WORKER \
--conf spark.executorEnv.JAVA_HOME="$JAVA_HOME" \
/media/limmen/HDD/workspace/python/har_project_test/har_spark_setup.py \
--cluster_size $SPARK_WORKER_INSTANCES \
--features cleaned_data/train/features \
--labels cleaned_data/train/labels \
--mode train \
--model saved_model \
--tensorboard

# --steps 10000000000
#--epochs 1000000 \
#--batch_size 100 \
