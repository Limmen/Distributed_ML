# Task 4 - Feed-Forward NN with 4 layers
# 1. See results in stats.txt

from __future__ import print_function

# all tensorflow api is accessible through this
import tensorflow as tf
# to visualize the resutls
import matplotlib.pyplot as plt
# 70k mnist dataset that comes with the tensorflow container
from tensorflow.examples.tutorials.mnist import input_data

# Enable deterministic comparisons between executions
tf.set_random_seed(0)

#constants
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
NUM_CLASSES = 10
NUM_HIDDEN_1 = 200
NUM_HIDDEN_2 = 100
NUM_HIDDEN_3 = 60
NUM_HIDDEN_4 = 30
DROPOUT_RATE = 0.1
GD_LEARNING_RATE = 0.5
ADAM_LEARNING_RATE = 0.005

# load data
mnist = input_data.read_data_sets('../data/fashion', one_hot=True, validation_size=0)

print('Number of train examples in dataset ' + str(len(mnist.train.labels)))
print('Number of test examples in dataset ' + str(len(mnist.test.labels)))

# Define placeholders for input data and for input truth labels
x = tf.reshape(tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, 1]), [-1, IMAGE_PIXELS]) # training examples (just one color channel, i.e grayscale)
y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES]) # correct answers(labels)

# Define variables for the parameters of the model: Weights and biases, random-initialization with gaussian dist.
# For each convolutional layer specify patch size, input channels and output channels
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv_layer(x, W, stride):
    return tf.nn.conv2d(x, W, strides=stride, padding='SAME')

x_image = tf.reshape(x, [-1, 28, 28, 1])

W_conv1 = weight_variable([5, 5, 1, 4])
b_conv1 = bias_variable([4])

W_conv2 = weight_variable([5, 5, 4, 8])
b_conv2 = weight_variable([8])

W_conv3 = weight_variable([4, 4, 8, 12])
b_conv3 = weight_variable([12])

W_fc = weight_variable([7*7*12, 200])
b_fc = weight_variable([200])

W_fc2 = weight_variable([200, NUM_CLASSES])
b_fc2 = bias_variable([NUM_CLASSES])

# 2. Define the model - compute predicitions

# hidden layers with relu, 3 convolutional layer and 1 FC layer
h_conv1 = tf.nn.relu(conv_layer(x_image, W_conv1, [1, 1, 1, 1]) + b_conv1)
h_conv2 = tf.nn.relu(conv_layer(h_conv1, W_conv2, [1, 2, 2, 1]) + b_conv2)
h_conv3 = tf.nn.relu(conv_layer(h_conv2, W_conv3, [1, 2, 2, 1]) + b_conv3)
h_conv3_flat = tf.reshape(h_conv3, [-1, 7*7*12])

h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc) + b_fc)

#task 4
#h_fc1 = tf.nn.dropout(h_fc1, keep_prob=0.5)

# Compute the logits, aka the inverse of the sigmoid/softmax outputs
logits = tf.matmul(h_fc1, W_fc2) + b_fc2


# Readout/Softmax layer
# Define the loss, which is the loss between softmax of logits and the labels
# Tensorflow performs softmax (the output activation) as part of the loss for efficiency
cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=logits, name='xentropy'))

# 4. Define the accuracy
# Correct prediction is black/white, either the classification is correct or not
# Accuracy is the ratio of correct predictions over wrong predictions
correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

# Exponential decay of learning rate
global_step = tf.Variable(0, trainable=False)
starter_learning_rate = ADAM_LEARNING_RATE
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 500, 0.96, staircase=True)

# 5. Train with an Optimizer
# task 1
#train_step = tf.train.GradientDescentOptimizer(GD_LEARNING_RATE).minimize(cross_entropy_loss)

# task 2
#train_step = tf.train.AdamOptimizer(ADAM_LEARNING_RATE).minimize(cross_entropy_loss)

# task 3
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss, global_step=global_step)

# initialize and run start operation
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Function representing a single iteration during training.
# Returns a tuple of accuracy and loss statistics.
def training_step(i, update_test_data, update_train_data):
    # actual learning
    # reading batches of 100 images with 100 labels
    batch_X, batch_Y = mnist.train.next_batch(100)
    # the backpropagation training step
    sess.run(train_step, feed_dict={x: batch_X, y_: batch_Y})

    # evaluating model performance for printing purposes
    # evaluation used to later visualize how well you did at a particular time in the training
    train_a = []  # Array of training-accuracy for a single iteration
    train_c = []  # Array of training-cost for a single iteration
    test_a = []  # Array of test-accuracy for a single iteration
    test_c = []  # Array of test-cost for a single iteration

    # If stats for train-data should be updates, compute loss and accuracy for the batch and store it
    if update_train_data:
        train_acc, train_cos = sess.run([accuracy, cross_entropy_loss], feed_dict={x: batch_X, y_: batch_Y})
        train_a.append(train_acc)
        train_c.append(train_cos)

    # If stats for test-data should be updates, compute loss and accuracy for the batch and store it
    if update_test_data:
        test_acc, test_cos = sess.run([accuracy, cross_entropy_loss],
                                      feed_dict={x: mnist.test.images, y_: mnist.test.labels})
        test_a.append(test_acc)
        test_c.append(test_cos)

    return train_a, train_c, test_a, test_c



# 6. Train and test the model, store the accuracy and loss per iteration

train_accuracy = []
train_cost = []
test_accuracy = []
test_cost = []

NUM_TRAINING_ITER = 10000
NUM_EPOCH_SIZE = 100
for i in range(NUM_TRAINING_ITER):
    test = False
    if i % NUM_EPOCH_SIZE == 0:
        test = True
        print("iter: " + str(i))
    a, c, ta, tc = training_step(i, test, test)
    train_accuracy += a
    train_cost += c
    test_accuracy += ta
    test_cost += tc

# 7. Plot and visualise the accuracy and loss

print('Final test accuracy ' + str(test_accuracy[-1]))
print('Final test loss ' + str(test_cost[-1]))

# accuracy training vs testing dataset
plt.plot(train_accuracy, label='Train data')
plt.xlabel('Epoch')
plt.plot(test_accuracy, label='Test data')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()
plt.title('Accuracy per epoch train vs test')
plt.show()

# loss training vs testing dataset
plt.plot(train_cost, label='Train data')
plt.plot(test_cost, label='Test data')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss per epoch, train vs test')
plt.grid(True)
plt.show()

# Zoom in on the tail of the plots
zoom_point = 50
x_range = range(zoom_point, int(NUM_TRAINING_ITER / NUM_EPOCH_SIZE))
plt.plot(x_range, train_accuracy[zoom_point:])
plt.plot(x_range, test_accuracy[zoom_point:])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy per epoch train vs test')
plt.legend()
plt.grid(True)
plt.show()

plt.plot(train_cost[zoom_point:])
plt.plot(test_cost[zoom_point:])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss per epoch train vs test')
plt.legend()
plt.grid(True)
plt.show()