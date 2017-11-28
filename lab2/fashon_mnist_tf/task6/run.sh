#!/bin/bash

$SPARK_HOME/sbin/start-master.sh

export MASTER=spark://ThinkPad-W510:7077
export SPARK_WORKER_INSTANCES=4
export CORES_PER_WORKER=2

$SPARK_HOME/sbin/start-slave.sh spark://ThinkPad-W510:7077 --cores 2 --memory 4g

firefox http://127.0.0.1:8080 &

$SPARK_HOME/bin/spark-submit \
--master spark://ThinkPad-W510:7077 \
--py-files /home/limmen/workspace/scala/ID2223-Scalable-ML/lab2/fashon_mnist_tf/task6/task6.py \
--conf spark.cores.max=8 \
--conf spark.task.cpus=2 \
--conf spark.executorEnv.JAVA_HOME="$JAVA_HOME" \
/home/limmen/workspace/scala/ID2223-Scalable-ML/lab2/fashon_mnist_tf/task6/task6.py

