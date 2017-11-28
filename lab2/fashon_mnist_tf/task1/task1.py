# Task 1 - Softmax Regression without regularization
# See results in stats.txt
# Results were about 0.82, the plots show a little bit of overfitting but not too bad.
# The model is not so powerful so that is probably why it did not overfit despite noy using any regularization.
# It performed a little bit better with Adam which adapts the learning rate (momentum can increase convergence and
# adaptive learning rate can adjust to not overshoot mimimas).
# Adam needed much smaller learning rate than GD to not oscillate, this is probably due to momentum increasing the
# learning rate to fast in the beginning if the initial learning rate is too high
from __future__ import print_function
# all tensorflow api is accessible through this
import tensorflow as tf
# to visualize the results
import matplotlib.pyplot as plt
# 70k mnist dataset that comes with the tensorflow container
from tensorflow.examples.tutorials.mnist import input_data

# Enable deterministic comparisons between executions
tf.set_random_seed(0)

# constants
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
NUM_CLASSES = 10
GD_LEARNING_RATE = 0.5
ADAM_LEARNING_RATE = 0.005
BATCH_SIZE = 100

# load data
mnist = input_data.read_data_sets('../data/fashion', one_hot=True)

print('Number of train examples in dataset ' + str(len(mnist.train.labels)))
print('Number of test examples in dataset ' + str(len(mnist.test.labels)))

# Define placeholders for input data and for input truth labels
X = tf.placeholder(tf.float32,
                   [None, IMAGE_SIZE, IMAGE_SIZE, 1])  # training examples (just one color channel, i.e grayscale)
y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])  # correct answers(labels)

## Define variables for the parameters of the model: Weights and biases, zero-initializaton
W = tf.Variable(tf.zeros([IMAGE_PIXELS, NUM_CLASSES]))  # weights W[784, 10], 784=28*28
b = tf.Variable(tf.zeros([NUM_CLASSES]))  # biases b[10]

# XX is the design matrix [TRAINING_SAMPLES, FEATUREVECTOR]
XX = tf.reshape(X, [-1, IMAGE_PIXELS])  # flatten the images into a single vector of pixels (1D input, not 2D)

# 2. Define the model - compute predictions
# Softmax is the activation function --> softmax regression (generalization of logistic regression to multi-class classification)
# softmax(X*W + b)
y = tf.nn.softmax(tf.matmul(XX, W) + b)

# 3. Define the loss function
# Cross entropy - generalization of log-loss to multi-class classification.
# Negative sum of truth label y_ times log of prediction y
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# 4. Define the accuracy
# Correct prediction is black/white, either the classification is correct or not
# Accuracy is the ratio of correct predictions over wrong predictions
correct_predictions = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

# 5. Train with an Optimizer
#train_step = tf.train.GradientDescentOptimizer(GD_LEARNING_RATE).minimize(cross_entropy)
train_step = tf.train.AdamOptimizer(ADAM_LEARNING_RATE).minimize(cross_entropy)
#train_step = tf.train.AdamOptimizer(0.0005).minimize(cross_entropy)

# initialize variables, init is a starting operation that consumes tensors as input.
init = tf.global_variables_initializer()
# session encapsulates the environment in which Operation objects are executed
sess = tf.Session()
# run the init operation
sess.run(init)


# Function representing a single iteration during training.
# Returns a tuple of accuracy and loss statistics.
def training_step(i, update_test_data, update_train_data):
    # actual learning
    # reading batches of 100 images with 100 labels
    batch_X, batch_Y = mnist.train.next_batch(BATCH_SIZE)
    # the backpropagation training step, feeds in the batch
    sess.run(train_step, feed_dict={XX: batch_X, y_: batch_Y})

    # evaluating model performance for printing purposes
    # evaluation used to later visualize how well you did at a particular time in the training
    train_a = []  # Array of training-accuracy for a single iteration
    train_c = []  # Array of training-cost for a single iteration
    test_a = []  # Array of test-accuracy for a single iteration
    test_c = []  # Array of test-cost for a single iteration

    # If stats for train-data should be updates, compute loss and accuracy for the batch and store it
    if update_train_data:
        a, c = sess.run([accuracy, cross_entropy], feed_dict={XX: batch_X, y_: batch_Y})
        train_a.append(a)
        train_c.append(c)

    # If stats for test-data should be updates, compute loss and accuracy for the batch and store it
    if update_test_data:
        a, c = sess.run([accuracy, cross_entropy], feed_dict={XX: mnist.test.images, y_: mnist.test.labels})
        test_a.append(a)
        test_c.append(c)

    return train_a, train_c, test_a, test_c


# 6. Train and test the model, store the accuracy and loss per iteration

train_a = []  # Array of training-accuracy for each epoch
train_c = []  # Array of training-cost for each epoch
test_a = []  # Array of test-accuracy for each epoch
test_c = []  # Array of test-cost for each epoch

NUM_TRAINING_ITER = 10000
NUM_EPOCH_SIZE = 100

for i in range(NUM_TRAINING_ITER):
    test = False
    if i % NUM_EPOCH_SIZE == 0:
        test = True
        print("iter: " + str(i))
    a, c, ta, tc = training_step(i, test, test)  # Get the statistics for this training step
    # Update the stats with stats for this training step
    train_a += a
    train_c += c
    test_a += ta
    test_c += tc

# 7. Plot and visualise the accuracy and loss

print('Final test accuracy ' + str(test_a[-1]))
print('Final test loss ' + str(test_c[-1]))

# accuracy training vs testing dataset
plt.plot(train_a, label='Train data')
plt.xlabel('Epoch')
plt.plot(test_a, label='Test data')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()
plt.title('Accuracy per epoch train vs test')
plt.show()

# loss training vs testing dataset
plt.plot(train_c, label='Train data')
plt.plot(test_c, label='Test data')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss per epoch, train vs test')
plt.grid(True)
plt.show()

# # Zoom in on the tail of the plots
# zoom_point = 50
# x_range = range(zoom_point, int(NUM_TRAINING_ITER / NUM_EPOCH_SIZE))
# plt.plot(x_range, train_a[zoom_point:])
# plt.plot(x_range, test_a[zoom_point:])
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.title('Accuracy per epoch train vs test')
# plt.legend()
# plt.grid(True)
# plt.show()
#
# plt.plot(train_c[zoom_point:])
# plt.plot(test_c[zoom_point:])
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Loss per epoch train vs test')
# plt.legend()
# plt.grid(True)
# plt.show()
