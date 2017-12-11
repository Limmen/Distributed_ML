import tensorflow as tf
# import tensorflowonspark as tfos
import pandas as pd
import numpy as np

CLASSES = {"bike": 0, "sit": 1, "stand": 2, "walk": 3, "stairsup": 4, "stairsdown": 5}
PA_DATA = "data/activity_data/Phones_accelerometer.csv"
CSV_COLUMNS = ["Index", "Arrival_Time", "Creation_Time", "x", "y", "z", "User", "Model", "Device", "gt"]
NUM_FEATURES = 8
NUM_CLASSES = 7
NUM_TRAINING_ITER = 100
NUM_EPOCH_SIZE = 10
pd.options.mode.chained_assignment = None


def convert_label(str):
    if CLASSES.get(str):
        return CLASSES.get(str)
    else:
        return 6

def convert_model(str):
    return 0


def convert_device(str):
    return 0


def convert_user(str):
    return 0


def pre_process():
    ""
    raw_data = pd.read_csv("data/activity_data/Phones_accelerometer.csv")
    raw_data = raw_data.drop(["Index"], axis=1)
    features = raw_data[["Arrival_Time", "Creation_Time", "x", "y", "z", "User", "Model", "Device"]]
    features["Model"] = features["Model"].apply(lambda x: convert_model(x))
    features["Device"] = features["Device"].apply(lambda x: convert_device(x))
    features["User"] = features["User"].apply(lambda x: convert_user(x))
    labels = raw_data[["gt"]]
    labels = labels.applymap(lambda x: convert_label(x))
    return labels.values, features.values


def define_placeholders():
    # input
    x = tf.placeholder(tf.float32, [None, NUM_FEATURES])
    y = tf.placeholder(tf.int64, [None])
    return x, y


def build_graph(x):
    # The model
    W = tf.Variable(tf.zeros([NUM_FEATURES, NUM_CLASSES]))
    b = tf.Variable(tf.zeros([NUM_CLASSES]))
    logits = tf.matmul(x, W) + b
    return logits


def define_optimizer(logits, labels):
    # Define loss and optimizer
    cross_entropy = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
    train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)
    # Test trained model
    correct_prediction = tf.equal(tf.argmax(logits, 1), labels)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return train_step, accuracy, cross_entropy


def init_graph():
    # initialize and run start operation
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    return sess


def main(x_data, y_data):
    train_accuracy = []
    train_cost = []
    test_accuracy = []
    test_cost = []

    x, y = define_placeholders()
    logits = build_graph(x)
    training_step, accuracy, cross_entropy_loss = define_optimizer(logits, y)
    sess = init_graph()
    for i in range(NUM_TRAINING_ITER):
        print(f"iter: {i}")
        if (i - 1 > 0):
            print(f"accuracy: {train_accuracy[i-1]}")
        print(len(train_accuracy))
        a, c, ta, tc = training_step_fun(x, y, x_data, y_data, sess, training_step, accuracy, cross_entropy_loss)
        train_accuracy += a
        train_cost += c
        test_accuracy += ta
        test_cost += tc

    sess.close()
    return test_accuracy


# Function representing a single iteration during training.
# Returns a tuple of accuracy and loss statistics.
def training_step_fun(x, y, x_batch, y_batch, sess, training_step, accuracy, cross_entropy_loss):
    # the backpropagation training step
    sess.run(training_step, feed_dict={x: x_batch, y: y_batch})

    # evaluating model performance for printing purposes
    # evaluation used to later visualize how well you did at a particular time in the training
    train_a = []  # Array of training-accuracy for a single iteration
    train_c = []  # Array of training-cost for a single iteration
    test_a = []  # Array of test-accuracy for a single iteration
    test_c = []  # Array of test-cost for a single iteration

    train_acc, train_cos = sess.run([accuracy, cross_entropy_loss], feed_dict={x: x_batch, y: y_batch})
    train_a.append(train_acc)
    train_c.append(train_cos)

    return train_a, train_c, test_a, test_c


if __name__ == '__main__':
    y_data, x_data = pre_process()
    print(x_data[0])
    print(y_data[0])
    tf.convert_to_tensor(y_data)
    print(y_data.shape)
    print(x_data.shape)
    print(y_data.reshape(len(y_data)).shape)
    test_accuracy = main(x_data, y_data.reshape(len(y_data)))
