import tensorflow as tf
from tensorflowonspark import TFNode
import numpy as np
import time
import logging

NUM_FEATURES = 8
NUM_CLASSES = 7


def map_fun(args, ctx):
    worker_num = ctx.worker_num
    job_name = ctx.job_name
    task_index = ctx.task_index
    cluster_spec = ctx.cluster_spec
    logging.info(f" worker_num: {worker_num}, task_index: {task_index}, job_name: {job_name}, cluster_spec: {cluster_spec}")

    # Delay PS nodes a bit, since workers seem to reserve GPUs more quickly/reliably (w/o conflict)
    if job_name == "ps":
        time.sleep((worker_num + 1) * 5)

    batch_size = args.batch_size
    logging.info(f"batchsize: {batch_size}")
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

    if job_name == "ps":
        logging.info("Parameter Server joining")
        server.join()

    elif job_name == "worker":
        logging.info("worker starting")
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
                    tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))

                tf.summary.scalar("loss", cross_entropy)

                train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy, global_step=global_step)

                prediction = tf.argmax(tf.nn.softmax(logits), 1, name="prediction")
                # Test trained model
                correct_prediction = tf.equal(tf.argmax(logits, 1), labels)
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                tf.summary.scalar("acc", accuracy)
                return train_step, accuracy, cross_entropy, global_step, prediction

            x, y = define_placeholders()
            logits = build_graph(x)
            training_step, accuracy, cross_entropy_loss, global_step, pred = define_optimizer(logits, y)

            saver = tf.train.Saver()
            summary_op = tf.summary.merge_all()
            init_op = tf.global_variables_initializer()

            logdir = TFNode.hdfs_path(ctx, args.model)
            logging.info("tensorflow model path: {0}".format(logdir))

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
            # The supervisor takes care of session initialization, restoring from
            # a checkpoint, and closing when done or an error occurs.
            with sv.managed_session(server.target) as sess:
                logging.info("session ready, starting training")

                # Loop until the supervisor shuts down or maximum steps have completed.
                step = 0
                # TFNode.DataFeed handles SPARK input mode, will convert the RDD into TF formats
                tf_feed = TFNode.DataFeed(ctx.mgr, args.mode == "train")
                while not sv.should_stop() and not tf_feed.should_stop() and step < args.steps:
                    # Run a training step asynchronously.
                    # See `tf.train.SyncReplicasOptimizer` for additional details on how to
                    # perform *synchronous* training.

                    # using feed_dict
                    batch_xs, batch_ys = feed_dict(tf_feed.next_batch(batch_size))
                    feed = {x: batch_xs, y: batch_ys}

                    if len(batch_xs) > 0:
                        if args.mode == "train":
                            _, summary, step = sess.run([training_step, summary_op, global_step], feed_dict=feed)
                            # logging.info accuracy and save model checkpoint to HDFS every 100 steps
                            if (step % 100 == 0):
                                acc = sess.run(accuracy, {x: batch_xs, y: batch_ys})
                                logging.info(f"step: {step}, acc: {acc}")

                            if sv.is_chief:
                                summary_writer.add_summary(summary, step)
                        else:  # args.mode == "inference"
                            if(len(batch_ys == batch_size)):
                                preds, acc = sess.run([pred, accuracy], feed_dict=feed)
                                #acc, loss = sess.run([accuracy, cross_entropy_loss], feed_dict=feed)
                                results = ["Label: {0}, Prediction: {1}".format(label, pred) for label, pred in
                                           zip(batch_ys, preds)]
                                logging.info(f"len results: {len(results)}")
                                tf_feed.batch_results(results)
                                logging.info("acc: {0}".format(acc))
                            else:
                                logging.info("Skipping last batch because it is not complete")

            if sv.should_stop() or step >= args.steps:
                logging.info("terminating")
                tf_feed.terminate()

        # Ask for all the services to stop.
        logging.info("stopping supervisor")
        sv.stop()



