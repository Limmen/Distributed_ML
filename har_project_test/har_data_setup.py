import pyspark
import csv
import argparse
import logging

PA_DATA = "data/activity_data/Phones_accelerometer.csv"
PG_DATA = "data/activity_data/Phones_gyroscope.csv"
WA_DATA = "data/activity_data/Watch_accelerometer.csv"
WG_DATA = "data/activity_data/Watch_gyroscope.csv"

def sparkConf(cluster):
    # overwrite the output directory in spark  set("spark.hadoop.validateOutputSpecs", "false")
    conf = pyspark.SparkConf()
    if not cluster:
        return conf \
            .setMaster("local") \
            .setAppName("har_data_setup") \
            .set("spark.hadoop.validateOutputSpecs", "false")
    else:
        return conf \
            .setAppName("har_data_setup") \
            .set("spark.hadoop.validateOutputSpecs", "false")

def read_raw_data(sql, num_partitions):
    paRaw = sql.read.format("com.databricks.spark.csv").options(header="true", numPartitons=num_partitions).load(PA_DATA)
    gaRaw = sql.read.format("com.databricks.spark.csv").options(header="true", numPartitons=num_partitions).load(PG_DATA)
    waRaw = sql.read.format("com.databricks.spark.csv").options(header="true", numPartitons=num_partitions).load(WA_DATA)
    wgRaw = sql.read.format("com.databricks.spark.csv").options(header="true", numPartitons=num_partitions).load(WG_DATA)
    return paRaw.unionAll(gaRaw).unionAll(waRaw).unionAll(wgRaw)


def main(args):
    output_path = args.output
    sc = pyspark.SparkContext(conf=sparkConf(args.cluster))
    sql = pyspark.SQLContext(sc)
    raw_data = read_raw_data(sql)
    raw_data = raw_data.drop("Index")
    raw_data_size = raw_data.count()
    #raw_data = raw_data.limit(13000000)
    train_raw, test_raw = raw_data.randomSplit([0.8, 0.2])
    train_data_size = train_raw.count()
    test_data_size = test_raw.count()
    train_raw = train_raw.limit(train_data_size - (train_data_size % 1000))
    train_data_size = train_data_size - (train_data_size % 1000)
    test_raw = test_raw.limit(test_data_size - (test_data_size % 1000))
    test_data_size = test_data_size - (test_data_size % 1000)
    train_raw = train_raw.repartition(a)
    test_raw = test_raw.repartition(args.num_partitions)
    classes = raw_data.select("gt").distinct().rdd.map(lambda row: row.gt).zipWithIndex().collectAsMap()
    devices = raw_data.select("Device").distinct().rdd.map(lambda row: row.Device).zipWithIndex().collectAsMap()
    models = raw_data.select("Model").distinct().rdd.map(lambda row: row.Model).zipWithIndex().collectAsMap()
    users = raw_data.select("User").distinct().rdd.map(lambda row: row.User).zipWithIndex().collectAsMap()
    labels_train = train_raw.select("gt").rdd.map(lambda row: classes[row.gt])
    features_train = train_raw.drop("gt").rdd.map(lambda row: (row.Arrival_Time,
                                                        row.Creation_Time,
                                                        row.x,
                                                        row.y,
                                                        row.z,
                                                        users[row.User],
                                                        models[row.Model],
                                                        devices[row.Device]))
    labels_test = test_raw.select("gt").rdd.map(lambda row: classes[row.gt])
    features_test = test_raw.drop("gt").rdd.map(lambda row: (row.Arrival_Time,
                                                               row.Creation_Time,
                                                               row.x,
                                                               row.y,
                                                               row.z,
                                                               users[row.User],
                                                               models[row.Model],
                                                               devices[row.Device]))
    features_train.repartition(args.num_partitions).map(lambda row: ','.join([str(i) for i in list(row)])).saveAsTextFile(output_path + "/train/features")
    labels_train.repartition(args.num_partitions).map(lambda row: ','.join(str(row))).saveAsTextFile(output_path + "/train/labels")
    write_dict(classes, output_path + "/classes")
    write_dict(devices, output_path + "/devices")
    write_dict(models, output_path + "/models")
    write_dict(users, output_path + "/users")
    write_dict({"raw_data_size": raw_data_size, "train_data_size": train_data_size, "test_data_size": test_data_size}, output_path + "/size")
    features_test.repartition(args.num_partitions).map(lambda row: ','.join([str(i) for i in list(row)])).saveAsTextFile(output_path + "/test/features")
    labels_test.repartition(args.num_partitions).map(lambda row: ','.join(str(row))).saveAsTextFile(output_path + "/test/labels")


def write_dict(dict, filename):
    with open(filename, 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in dict.items():
            writer.writerow([key, value])

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num-partitions", help="Number of output partitions", type=int, default=10)
    parser.add_argument("-o", "--output", help="HDFS directory to save examples in parallelized format", default="cleaned_data")
    parser.add_argument("-c", "--cluster", help="run on cluster master or local master", action="store_true")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    logging.info("args:", args)
    main(args)
