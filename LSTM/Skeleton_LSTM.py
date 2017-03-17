
import tensorflow as tf
from tensorflow.python.ops import rnn #, rnn_cell
from tensorflow.contrib.rnn import RNNCell
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import os
import re
import random
import PrepareData
from collections import deque

tf.reset_default_graph()
#  matplotlib inline

# Import MINST data
work_train = "/home/darid/ResearchWork/DataCollection/Data/Train"
work_test ="/home/darid/ResearchWork/DataCollection/Data/Test"
num = 0
# circle queue
ls_dir = os.listdir(work_train)
next_fn = deque(ls_dir, len(ls_dir))



# read all the data in a list : train_data
# train_data = []
# train_label = []
# for i in range(0, len(ls_dir)):
#      train_data.append(PrepareData.importData(work_train+'/'+ next_fn[i])[0])
#      train_label.append(PrepareData.importData(work_train+'/'+ next_fn[i])[1])
#
#
#
# next_train_data = deque(train_data, len(train_data))

def Next_data():
    # next_batch = next_train_data.rotate(-75*70)
    # return next_batch
    next_fn.rotate(-1)
    return PrepareData.importData(work_train+'/'+ next_fn[0])






# Parameters
learning_rate = 0.000026
training_iters = 10000
batch_size = 1
display_step = 100

print learning_rate
# Network Parameters
n_input = 25 # data input  shape: 25(joints)*3(dimension)*80(frame)
n_steps = 60 # timesteps
n_hidden = 128 # hidden layer num of features
n_classes = 8 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None,n_steps, n_input])
y = tf.placeholder("float", [None,n_classes])

print x,y

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def RNN(x, weights, biases):
    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Permuting batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, n_input)
    # print 'darid'

    x = tf.reshape(x, [-1, n_input])
    # print tf.shape(x)
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(x, n_steps, 0)
    # print tf.shape(x)
    # Define a lstm cell with tensorflow
    lstm_cell = tf.contrib.rnn.LSTMCell(n_hidden,forget_bias=1.0)
    # lstm_cell = RNNCell.BasicLSTMCell(n_hidden, forget_bias=1.0)

    # lsmt_layers = tf.nn.rnn_cell.MultiRNNCell([lstm_cell]*2)
    lsmt_layers = tf.contrib.rnn.MultiRNNCell([lstm_cell]*2)
    # Get lstm cell output
    # outputs, states = rnn.rnn(lsmt_layers, x, dtype=tf.float32)
    outputs, states = tf.contrib.rnn.static_rnn(lsmt_layers, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']


pred = RNN(x, weights, biases)

# Define loss and optimizer
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)) #xutao 3/13/2017
# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
optimizer = tf.train.RMSPropOptimizer(learning_rate, 0.9).minimize(cost)
# optimizer = tf.train.MomentumOptimizer(learning_rate,0.9).minimize(cost)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
# Evaluate model
results = tf.argmax(pred,1)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()
#
#
# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations


        # batch_x, batch_y = mnist.train.next_batch(batch_size)
        # # Darid Input Data
        # batch_x = batch_x.reshape((batch_size, n_steps, n_input))
        # # Run optimization op (backprop)
    pn = 0
    i = 0
    while step * batch_size < training_iters:

        batch_x, batch_y = Next_data()

        # input data stream
        # if i >= len(train_data):
        #     i=0
        #
        # batch_x = train_data[i]
        #
        # batch_y = train_label[i]


        batch_x = np.asarray(batch_x)
        # Reshape data to get 28 seq of 28 elements
        batch_x = batch_x.reshape((batch_size, n_steps, n_input))
        # Run optimization op (backprop)
        test = sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        pr = sess.run(correct_pred, feed_dict={x: batch_x, y: batch_y})
        pr_s = str(pr[0])
        if pr_s=="True":
           pn = pn+1
        if step % display_step == 0:
            # Calculate batch accuracy

            # acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})

            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            ac = float(pn) / display_step
            # print ac,pn
            print "Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(ac)
            pn = 0
        step += 1
        i += 1
    print "Optimization Finished!"
# #
#     # Calculate accuracy for 128 mnist test images
#     test_len = 128
#     test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
#     test_label = mnist.test.labels[:test_len]
#     print "Testing Accuracy:", \
#         sess.run(accuracy, feed_dict={x: test_data, y: test_label})


    pn = 0
    tn = 0
    for t_f in os.listdir(work_test):


        test_x, test_y = PrepareData.importData(work_test+'/'+t_f)

        test_x = np.asarray(test_x)

        test_x = test_x.reshape((batch_size, n_steps, n_input))
        # Run optimization op (backprop)
        # test = sess.run(optimizer, feed_dict={x: test_x, y: test_y})
        re = sess.run(results,feed_dict={x: test_x})
        print re
        print test_y
        pr = sess.run(correct_pred, feed_dict={x: test_x, y: test_y})
        print pr
        pr_s = str(pr[0])
        if pr_s == "True":
            pn = pn + 1

        tn = tn+1

    print float(pn)/tn
