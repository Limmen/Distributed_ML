# Task 3 - Regularization and Tuning of Hyperparams (FF-NN-4)
# 1. See results in stats.txt.
# We did not do any heavy tuning of the hyperparameters but found a
# good setting for learning rate decay that improved the results for the gradient descent optimizer,
# furthermore we found that dropout did not help much, or we did not find good values for dropout,
# we should use grid search of random search to find better values because dropout in this case actually
# reduced the accuracy, which indicates underfitting.
# 2. Learning rate decay can counter gradient oscillations.
# the effect is that the learning takes large steps in beginning towards the minimum,
# and if the gradients are a bit noisy it will avoid wandering far off the minima since
# the steps will get smaller and the gradient will oscillate in a smaller area around the minimum.
# If the learning rate is static there is a risk that if the gradient is noisy, the learning will
# move far away from the minima in some iterations. Adam/Adagrad and similar optimizers performs automatic
# learning rate decay and adaption. In this task we use our own decay instead
# 3. Dropout means that during training some neurons are randomly set to 0 for each training example
# The intuition behind dropout as a regularization
# technique is that it hinders neurons in the network from adapting too strongly
# on the output of the prior layer, as those networks will not always be present during training


from __future__ import print_function
# all tensorflow api is accessible through this
import tensorflow as tf
# to visualize the resutls
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
NUM_HIDDEN_1 = 200
NUM_HIDDEN_2 = 100
NUM_HIDDEN_3 = 60
NUM_HIDDEN_4 = 30
DROPOUT_RATE = 0.1

# load data
mnist = input_data.read_data_sets('../data/fashion', one_hot=True, validation_size=0)

print('Number of train examples in dataset ' + str(len(mnist.train.labels)))
print('Number of test examples in dataset ' + str(len(mnist.test.labels)))

# Define placeholders for input data and for input truth labels
x = tf.placeholder(tf.float32,
                   [None, IMAGE_SIZE, IMAGE_SIZE, 1])  # training examples (just one color channel, i.e grayscale)
y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])  # correct answers(labels)

# Define variables for the parameters of the model: Weights and biases, random-initialization with gaussian dist.
w_1 = tf.Variable((tf.truncated_normal([784, NUM_HIDDEN_1], stddev=0.1)))
b_1 = tf.Variable(tf.zeros([NUM_HIDDEN_1]))

w_2 = tf.Variable((tf.truncated_normal([NUM_HIDDEN_1, NUM_HIDDEN_2], stddev=0.1)))
b_2 = tf.Variable(tf.zeros([NUM_HIDDEN_2]))

w_3 = tf.Variable((tf.truncated_normal([NUM_HIDDEN_2, NUM_HIDDEN_3], stddev=0.1)))
b_3 = tf.Variable(tf.zeros([NUM_HIDDEN_3]))

w_4 = tf.Variable((tf.truncated_normal([NUM_HIDDEN_3, NUM_HIDDEN_4], stddev=0.1)))
b_4 = tf.Variable(tf.zeros([NUM_HIDDEN_4]))

w_5 = tf.Variable((tf.truncated_normal([NUM_HIDDEN_4, NUM_CLASSES], stddev=0.1)))
b_5 = tf.Variable(tf.zeros([NUM_CLASSES]))

# 2. Define the model - compute predicitions
xx = tf.reshape(x, [-1, IMAGE_PIXELS])  # flatten the images into a single vector of pixels (1D input, not 2D)

# Hidden unit activations : ReLU with dropout
hidden1 = tf.nn.relu(tf.matmul(xx, w_1) + b_1)
#hidden1 = tf.nn.dropout(hidden1, keep_prob=DROPOUT_RATE)
hidden2 = tf.nn.relu(tf.matmul(hidden1, w_2) + b_2)
#hidden2 = tf.nn.dropout(hidden2, keep_prob=DROPOUT_RATE)
hidden3 = tf.nn.relu(tf.matmul(hidden2, w_3) + b_3)
# hidden3 = tf.nn.dropout(hidden3, keep_prob=DROPOUT_RATE)
hidden4 = tf.nn.relu(tf.matmul(hidden3, w_4) + b_4)
# hidden4 = tf.nn.dropout(hidden4, keep_prob=DROPOUT_RATE)

# Compute the logits, aka the inverse of the sigmoid/softmax outputs
logits = tf.matmul(hidden4, w_5) + b_5

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
starter_learning_rate = GD_LEARNING_RATE
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 50, 0.96, staircase=True)

# 5. Train with an Optimizer
#train_step = tf.train.GradientDescentOptimizer(GD_LEARNING_RATE).minimize(cross_entropy_loss)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy_loss, global_step=global_step)

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
    sess.run(train_step, feed_dict={xx: batch_X, y_: batch_Y})

    # evaluating model performance for printing purposes
    # evaluation used to later visualize how well you did at a particular time in the training
    train_a = []  # Array of training-accuracy for a single iteration
    train_c = []  # Array of training-cost for a single iteration
    test_a = []  # Array of test-accuracy for a single iteration
    test_c = []  # Array of test-cost for a single iteration

    # If stats for train-data should be updates, compute loss and accuracy for the batch and store it
    if update_train_data:
        train_acc, train_cos = sess.run([accuracy, cross_entropy_loss], feed_dict={xx: batch_X, y_: batch_Y})
        train_a.append(train_acc)
        train_c.append(train_cos)

    # If stats for test-data should be updates, compute loss and accuracy for the batch and store it
    if update_test_data:
        test_acc, test_cos = sess.run([accuracy, cross_entropy_loss],
                                      feed_dict={xx: mnist.test.images, y_: mnist.test.labels})
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
    a, c, ta, tc = training_step(i, test, test)  # Get the statistics for this training step
    # Update the stats with stats for this training step
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
