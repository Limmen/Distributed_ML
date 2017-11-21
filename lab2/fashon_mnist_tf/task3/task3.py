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

x = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, 1])
y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES]) # correct answers(labels)

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
xx = tf.reshape(x, [-1, IMAGE_PIXELS])

hidden1 = tf.nn.relu(tf.matmul(xx, w_1) + b_1)
#hidden1 = tf.nn.dropout(hidden1, keep_prob=DROPOUT_RATE)
hidden2 = tf.nn.relu(tf.matmul(hidden1, w_2) + b_2)
hidden2 = tf.nn.dropout(hidden2, keep_prob=DROPOUT_RATE)
hidden3 = tf.nn.relu(tf.matmul(hidden2, w_3) + b_3)
#hidden3 = tf.nn.dropout(hidden3, keep_prob=DROPOUT_RATE)
hidden4 = tf.nn.relu(tf.matmul(hidden3, w_4) + b_4)
#hidden4 = tf.nn.dropout(hidden4, keep_prob=DROPOUT_RATE)
logits = tf.matmul(hidden4, w_5) + b_5


#labels = tf.to_int64(y_)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=logits, name='xentropy')
loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
#softmax = tf.nn.softmax(logits)

# 4. Define the accuracy
correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
# Exponential learning rate
global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 0.5
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 1000, 0.96, staircase=True)
# 5. Train with an Optimizer
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)


# initialize
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


def training_step(i, update_test_data, update_train_data):

    #print 'Iter ' + str(i)
    ####### actual learning
    # reading batches of 100 images with 100 labels
    batch_X, batch_Y = mnist.train.next_batch(100)
    # the backpropagation training step
    sess.run(train_step, feed_dict={xx: batch_X, y_: batch_Y})

    ####### evaluating model performance for printing purposes
    # evaluation used to later visualize how well you did at a particular time in the training
    train_a = []
    train_c = []
    test_a = []
    test_c = []
    if update_train_data:
        train_acc, train_cos = sess.run([accuracy, loss], feed_dict={xx: batch_X, y_: batch_Y})
        train_a.append(train_acc)
        train_c.append(train_cos)

    if update_test_data:
        test_acc, test_cos = sess.run([accuracy, loss], feed_dict={xx: mnist.test.images, y_: mnist.test.labels})
        test_a.append(test_acc)
        test_c.append(test_cos)

    return train_a, train_c, test_a, test_c


# 6. Train and test the model, store the accuracy and loss per iteration

train_accuracy = []
train_cost = []
test_accuracy = []
test_cost = []

training_iter = 10000
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