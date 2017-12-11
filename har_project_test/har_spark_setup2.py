import pyspark
import argparse
from tensorflowonspark import TFCluster
import har_dist_model2
import logging

def sparkConf():
    # overwrite the output directory in spark  set("spark.hadoop.validateOutputSpecs", "false")
    conf = pyspark.SparkConf()
    return conf \
        .setAppName("har_data_training_inference") \
        .set("spark.hadoop.validateOutputSpecs", "false")

def read_data(sc, featuresFile, labelsFile):
    labels = sc.textFile(labelsFile).map(lambda ln: [int(x) for x in ln.split(',')])
    features = sc.textFile(featuresFile).map(lambda ln: [float(x) for x in ln.split(',')]).repartition(labels.getNumPartitions())
    dataRDD = features.zip(labels)
    return dataRDD

def main():
    sc = pyspark.SparkContext(conf=sparkConf())
    executors = sc._conf.get("spark.executor.instances")
    num_executors = int(executors) if executors is not None else 1
    num_ps = 1
    args = parse_args(num_executors)
    #dataRDD = read_data(sc, args.features, args.labels)
    cluster = TFCluster.run(sc, har_dist_model2.map_fun, args, args.cluster_size, num_ps, args.tensorboard, TFCluster.InputMode.TENSORFLOW, log_dir=args.model)
    #if args.mode == "train":
    #    logging.info("servers for training")
    #    cluster.train(dataRDD, args.epochs) #num_epochs: number of times to repeat the dataset during training.
    #else:
    #    logging.info("servers for inference")
    #    labelRDD = cluster.inference(dataRDD)
    #    logging.info("--------------------------------received labelRDD-------------------------")
    #    labelRDD.saveAsTextFile(args.output)
    cluster.shutdown()
    logging.info("Finnished, cluster shutdown")

def parse_args(num_executors):
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', "--cluster", action='store_true', default=False)
    parser.add_argument("-n", "--cluster_size", help="number of nodes in the cluster", type=int, default=num_executors)
    parser.add_argument("-tb", "--tensorboard", help="launch tensorboard process", action="store_true")
    parser.add_argument("-X", "--mode", help="train|inference", default="train")
    parser.add_argument("-f", "--features", help="HDFS path to features in parallelized format")
    parser.add_argument("-l", "--labels", help="HDFS path to labels in parallelized format")
    parser.add_argument("-m", "--model", help="HDFS path to save/load model during train/inference", default="saved_model")
    parser.add_argument("-r", "--rdma", help="use rdma connection", default=False)
    parser.add_argument("-o", "--output", help="HDFS path to save test/inference output", default="predictions")
    parser.add_argument("-s", "--steps", help="maximum number of steps", type=int, default=1000)
    parser.add_argument("-b", "--batch_size", help="number of records per batch", type=int, default=100)
    parser.add_argument("-e", "--epochs", help="number of epochs", type=int, default=1)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    logging.info("Setting up spark cluster for training/inference")
    main()