
import argparse
import pyspark
from pyspark.sql.functions import udf
from pyspark.sql.types import *
import tensorflow as tf
import pandas as pd
import numpy as np

SEQ_LABELS_TRAIN = "data/y_train.csv"
SEQ_FEATURES_TRAIN = "data/x_train.csv"
SEQ_LABELS_TEST = "data/y_test.csv"
SEQ_FEATURES_TEST = "data/x_test.csv"

conf = pyspark.SparkConf()
conf = conf.setAppName("har_data_setup").set("spark.hadoop.validateOutputSpecs", "false")

sc = pyspark.SparkContext(conf=conf)
sql = pyspark.SQLContext(sc)

CLEANED_DATA_PATH = "./cleaned_data"

def read_raw_data(sql):
    seq_features_train_raw = sql.read.format("com.databricks.spark.csv").options(header="false").load(SEQ_FEATURES_TRAIN)
    seq_labels_train_raw = sql.read.format("com.databricks.spark.csv").options(header="false").load(SEQ_LABELS_TRAIN)
    seq_features_test_raw = sql.read.format("com.databricks.spark.csv").options(header="false").load(SEQ_FEATURES_TEST)
    seq_labels_test_raw = sql.read.format("com.databricks.spark.csv").options(header="false").load(SEQ_LABELS_TEST)
    return seq_features_train_raw,seq_labels_train_raw, seq_features_test_raw, seq_labels_test_raw

seq_features_train_raw, seq_labels_train_raw,seq_features_test_raw,seq_labels_test_raw = read_raw_data(sql)
features_train_size = seq_features_train_raw.count()
labels_train_size = seq_labels_train_raw.count()
features_test_size = seq_features_test_raw.count()
labels_test_size = seq_labels_test_raw.count()

print("train feat size: {0}, train label size: {1}, test feat size {2}, test label size {3}".format(features_train_size, labels_train_size, features_test_size, labels_test_size))

seq_labels_test_raw.printSchema

classes = seq_labels_train_raw.unionAll(seq_labels_test_raw).select("_c0").distinct().rdd.map(lambda row: row._c0).zipWithIndex().collectAsMap()
seq_labels_train_clean = seq_labels_train_raw.select("_c0").rdd.map(lambda row: classes[row._c0])
seq_labels_test_clean = seq_labels_test_raw.select("_c0").rdd.map(lambda row: classes[row._c0])

labels_train_np = seq_labels_train_clean.collect()
labels_test_np = seq_labels_test_clean.collect()

np.savetxt(CLEANED_DATA_PATH + "/train/labels/y_train.csv", np.array(labels_train_np).astype(int), fmt='%i', delimiter=",")
np.savetxt(CLEANED_DATA_PATH + "/test/labels/y_test.csv", np.array(labels_test_np).astype(int), fmt='%i', delimiter=",")
np.savetxt(CLEANED_DATA_PATH + "/classes/classes.csv", np.array([[k,v] for k,v in classes.items()]),fmt="%s", delimiter=",")
np.savetxt(CLEANED_DATA_PATH + "/size/sizes.csv", np.array([["features_train_size", features_train_size], ["labels_train_size", labels_train_size], ["features_test_size", features_test_size], ["labels_test_size", labels_test_size]]), fmt="%s", delimiter=",")

