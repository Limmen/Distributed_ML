import tensorflow as tf
from tensorflowonspark import TFNode
import numpy as np
import time
import logging

NUM_FEATURES = 8
NUM_CLASSES = 7


def map_fun(args, ctx):
    NUM_FEATURES = 8
    NUM_CLASSES = 7

    def print_log(worker_num, arg):
        print("%d: " % worker_num)
        print(arg)

    from tensorflowonspark import TFNode
    from datetime import datetime
    import getpass
    import math
    import numpy
    import os
    import signal
    import tensorflow as tf
    import time
    # Used to get TensorBoard logdir for TensorBoard that show up in HopsWorks
    #from hops import tensorboard

    worker_num = ctx.worker_num
    job_name = ctx.job_name
    task_index = ctx.task_index
    cluster_spec = ctx.cluster_spec
    print_log(worker_num, "task_index: {0}, job_name {1}, cluster_spec: {2}".format(task_index, job_name, cluster_spec))
    num_workers = len(cluster_spec['worker'])

    # Delay PS nodes a bit, since workers seem to reserve GPUs more quickly/reliably (w/o conflict)
    if job_name == "ps":
        time.sleep((worker_num + 1) * 5)

    batch_size = args.batch_size
    print_log(worker_num, "batch_size: {0}".format(batch_size))

    # Get TF cluster and server instances
    cluster, server = TFNode.start_cluster_server(ctx, 1, args.rdma)

    def feed_dict(batch):
        # Convert from [(features, labels)] to two numpy arrays of the proper type
        features = []
        labels = []
        for item in batch:
            features.append(item[0])
            labels.append(item[1])
        xs = np.array(features)
        xs = xs.astype(np.float32)
        ys = np.array(labels)
        ys = ys.reshape(len(ys))
        ys = ys.astype(np.uint8)
        return (xs, ys)

    def read_csv_examples(feature_dir, label_dir, batch_size=100, num_epochs=None, task_index=None, num_workers=None):
        print_log(worker_num, "num_epochs: {0}".format(num_epochs))
        # Setup queue of csv feature filenames
        tf_record_pattern = os.path.join(feature_dir, 'part-*')
        features = tf.gfile.Glob(tf_record_pattern)
        print_log(worker_num, "features: {0}".format(features))
        feature_queue = tf.train.string_input_producer(features, shuffle=False, capacity=1000, num_epochs=num_epochs,
                                                       name="feature_queue")

        # Setup queue of csv label filenames
        tf_record_pattern = os.path.join(label_dir, 'part-*')
        labels = tf.gfile.Glob(tf_record_pattern)
        print_log(worker_num, "labels: {0}".format(labels))
        label_queue = tf.train.string_input_producer(labels, shuffle=False, capacity=1000, num_epochs=num_epochs,
                                                     name="label_queue")

        # Setup reader for feature queue
        feature_reader = tf.TextLineReader(name="feature_reader")
        _, feat_csv = feature_reader.read(feature_queue)
        feature_defaults = [[1.0] for col in range(NUM_FEATURES)]
        feat = tf.stack(tf.decode_csv(feat_csv, feature_defaults))
        # Normalize values to [0,1]
        #norm = tf.constant(255, dtype=tf.float32, shape=(NUM_FEATURES,))
        #feature = tf.div(feat, norm)
        feature = feat
        print_log(worker_num, "feature: {0}".format(feature))

        # Setup reader for label queue
        label_reader = tf.TextLineReader(name="label_reader")
        _, label_csv = label_reader.read(label_queue)
        #label_defaults = [[1.0] for col in range(NUM_CLASSES)]
        label_defaults = [tf.constant([], dtype=tf.int64)]
        label = tf.stack(tf.decode_csv(label_csv, label_defaults))
        print_log(worker_num, tf.shape(label))
        #label2 = tf.reshape(label, [-1])
        print_log(worker_num, "label: {0}".format(label))

        # Return a batch of examples
        return tf.train.batch([feature, label], batch_size, num_threads=10, name="batch_csv")

    if job_name == "ps":
        print_log(worker_num, "Parameter Server Joining")
        server.join()

    elif job_name == "worker":
        print_log(worker_num, "worker starting")

        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % task_index,
                cluster=cluster)):

            def define_placeholders():
                x = tf.placeholder(tf.float32, [None, NUM_FEATURES])
                y = tf.placeholder(tf.int64, [None])
                return x, y

            def build_graph(x):
                # The model
                W = tf.Variable(tf.zeros([NUM_FEATURES, NUM_CLASSES]))
                tf.summary.histogram("hidden_weights", W)
                b = tf.Variable(tf.zeros([NUM_CLASSES]))
                logits = tf.matmul(x, W) + b
                return logits

            def define_optimizer(logits, labels):
                global_step = tf.Variable(0)
                # Define loss and optimizer
                cross_entropy = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(labels, [-1]), logits=logits))

                tf.summary.scalar("loss", cross_entropy)

                train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy, global_step=global_step)

                prediction = tf.argmax(tf.nn.softmax(logits), 1, name="prediction")
                # Test trained model
                correct_prediction = tf.equal(tf.argmax(logits, 1), labels)
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                tf.summary.scalar("acc", accuracy)
                return train_step, accuracy, cross_entropy, global_step, prediction

            # Placeholders or QueueRunner/Readers for input data
            num_epochs = 1 if args.mode == "inference" else None if args.epochs == 0 else args.epochs
            index = task_index if args.mode == "inference" else None
            workers = num_workers if args.mode == "inference" else None

            features = TFNode.hdfs_path(ctx, args.features)
            labels = TFNode.hdfs_path(ctx, args.labels)
            x, y = read_csv_examples(features, labels, 100, num_epochs, index, workers)

            # x, y = define_placeholders()
            logits = build_graph(x)
            training_step, accuracy, cross_entropy_loss, global_step, pred = define_optimizer(logits, y)

            saver = tf.train.Saver()
            summary_op = tf.summary.merge_all()
            init_op = tf.global_variables_initializer()

            logdir = TFNode.hdfs_path(ctx, args.model)
            #logdir = tensorboard.logdir()
            print_log(worker_num, "tensorflow model path: {0}".format(logdir))

            if job_name == "worker" and task_index == 0:
                summary_writer = tf.summary.FileWriter(logdir, graph=tf.get_default_graph())

            if args.mode == "train":
                sv = tf.train.Supervisor(is_chief=(task_index == 0),
                                         logdir=logdir,
                                         init_op=init_op,
                                         summary_op=None,
                                         summary_writer=None,
                                         saver=saver,
                                         global_step=global_step,
                                         stop_grace_secs=300,
                                         save_model_secs=10)
            else:
                sv = tf.train.Supervisor(is_chief=(task_index == 0),
                                         logdir=logdir,
                                         summary_op=None,
                                         saver=saver,
                                         global_step=global_step,
                                         stop_grace_secs=300,
                                         save_model_secs=0)

            output_dir = TFNode.hdfs_path(ctx, args.output)
            output_file = tf.gfile.Open("{0}/part-{1:05d}".format(output_dir, worker_num), mode='w')
            # The supervisor takes care of session initialization, restoring from
            # a checkpoint, and closing when done or an error occurs.
            with sv.managed_session(server.target) as sess:
                print_log(worker_num, "session ready, starting training")

                # Loop until the supervisor shuts down or maximum steps have completed.
                step = 0
                count = 0
                # TFNode.DataFeed handles SPARK input mode, will convert the RDD into TF formats
                # tf_feed = TFNode.DataFeed(ctx.mgr, args.mode == "train")
                while not sv.should_stop() and step < args.steps:
                    # Run a training step asynchronously.
                    # See `tf.train.SyncReplicasOptimizer` for additional details on how to
                    # perform *synchronous* training.

                    # using feed_dict
                    # batch_xs, batch_ys = feed_dict(tf_feed.next_batch(batch_size))
                    # feed = {x: batch_xs, y: batch_ys}

                    if args.mode == "train":
                        _, summary, step = sess.run([training_step, summary_op, global_step])
                        # logging.info accuracy and save model checkpoint to HDFS every 100 steps
                        if (step % 100 == 0):
                            acc = sess.run(accuracy)
                            print_log(worker_num, "step: {0}, acc: {1}".format(step, acc))

                    if sv.is_chief:
                        summary_writer.add_summary(summary, step)
                    else:  # args.mode == "inference"
                        # if(len(batch_ys == batch_size)):
                        label, preds, acc = sess.run([labels, pred, accuracy])
                        # acc, loss = sess.run([accuracy, cross_entropy_loss], feed_dict=feed)
                        #results = ["Label: {0}, Prediction: {1}".format(label, pred) for label, pred in
                        #           zip(batch_ys, preds)]
                        #print_log(worker_num, "len_results: {0}".format(len(results)))
                        #tf_feed.batch_results(results)
                        #print_log(worker_num, "acc: {0}".format(acc))

                        #labels, pred, acc = sess.run([label, prediction, accuracy])
                        # print("label: {0}, pred: {1}".format(labels, pred))
                        #print("acc: {0}".format(acc))
                        for i in range(len(label)):
                            count += 1
                            output_file.write("{0} {1}\n".format(label[i], pred[i]))
                        print("count: {0}".format(count))

            if args.mode == "inference":
                output_file.close()
            # Delay chief worker from shutting down supervisor during inference, since it can load model, start session,
            # run inference and request stop before the other workers even start/sync their sessions.
            if task_index == 0:
                time.sleep(60)

            # Ask for all the services to stop.
            print("{0} stopping supervisor".format(datetime.now().isoformat()))
            sv.stop()



