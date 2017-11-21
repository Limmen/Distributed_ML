# all tensorflow api is accessible through this
import tensorflow as tf
# to visualize the resutls
import matplotlib.pyplot as plt
# 70k mnist dataset that comes with the tensorflow container
from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(0)

IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
NUM_CLASSES = 10


NUM_HIDDEN_1 = 200
NUM_HIDDEN_2 = 100
NUM_HIDDEN_3 = 60
NUM_HIDDEN_4 = 30

DROPOUT_RATE = 0.1

# load data
mnist = input_data.read_data_sets('../data/fashion', one_hot=True, validation_size=0)

x = tf.reshape(tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, 1]), [-1, IMAGE_PIXELS])
y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES]) # correct answers(labels)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=stride, padding='SAME')

# 2. Define the model - compute predicitions

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

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1, [1,1,1,1]) + b_conv1)
h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2, [1,2,2,1]) + b_conv2)
h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, [1,2,2,1]) + b_conv3)
h_conv3_flat = tf.reshape(h_conv3, [-1, 7*7*12])
h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc) + b_fc)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob=0.5)
logits = tf.matmul(h_fc1, W_fc2) + b_fc2


cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=logits, name='xentropy')
loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')


# 4. Define the accuracy
correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
# 5. Train with an Optimizer
train_step = tf.train.AdamOptimizer().minimize(loss)


# initialize
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


def training_step(i, update_test_data, update_train_data):

    print 'Iter ' + str(i)
    ####### actual learning
    # reading batches of 100 images with 100 labels
    batch_X, batch_Y = mnist.train.next_batch(100)
    # the backpropagation training step
    sess.run(train_step, feed_dict={x: batch_X, y_: batch_Y})

    ####### evaluating model performance for printing purposes
    # evaluation used to later visualize how well you did at a particular time in the training
    train_a = []
    train_c = []
    test_a = []
    test_c = []
    if update_train_data:
        train_acc, train_cos = sess.run([accuracy, loss], feed_dict={x: batch_X, y_: batch_Y})
        train_a.append(train_acc)
        train_c.append(train_cos)

    if update_test_data:
        test_acc, test_cos = sess.run([accuracy, loss], feed_dict={x: mnist.test.images, y_: mnist.test.labels})
        test_a.append(test_acc)
        test_c.append(test_cos)

    return train_a, train_c, test_a, test_c


# 6. Train and test the model, store the accuracy and loss per iteration

train_accuracy = []
train_cost = []
test_accuracy = []
test_cost = []

training_iter = 1000
epoch_size = 100
for i in range(training_iter):
    test = False
    if i % epoch_size == 0:
        test = True
    a, c, ta, tc = training_step(i, test, test)
    train_accuracy += a
    train_cost += c
    test_accuracy += ta
    test_cost += tc

# 7. Plot and visualise the accuracy and loss

print 'Final test accuracy ' + str(test_accuracy[-1])
print 'Final test loss ' + str(test_cost[-1])

# accuracy training vs testing dataset
plt.plot(train_accuracy, label='Train data')
plt.xlabel('Epoch')
plt.plot(test_accuracy, label='Test data')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()
plt.title('Accuracy train vs test')
plt.show()

# loss training vs testing dataset
plt.plot(train_cost, label='Train data')
plt.plot(test_cost, label='Test data')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()

# # Zoom in on the tail of the plots
# zoom_point = 50
# x_range = range(zoom_point,int(training_iter/epoch_size))
# plt.plot(x_range, train_accuracy[zoom_point:])
# plt.plot(x_range, test_accuracy[zoom_point:])
# plt.grid(True)
# plt.show()
#
# plt.plot(train_cost[zoom_point:])
# plt.plot(test_cost[zoom_point:])
# plt.grid(True)
# plt.show()