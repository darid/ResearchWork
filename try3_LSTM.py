import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np


learning_rate = 0.01
training_iters = 10000
batch_size = 1
display_step = 10

n_input = 2
n_steps = 8
n_hidden = 16
n_output = 1


int2binary = {}
binary_dim = 8

large_number = pow(2,binary_dim)
binary = np.unpackbits(np.array([range(large_number)],dtype= np.uint8 ).T,axis =1)
print binary

x = tf.placeholder(tf.float32,[None,n_input,n_steps])
y = tf.placeholder(tf.float32,[None,n_output])

weights = {'out':tf.Variable(tf.random_normal([n_hidden,n_output]))}

biases = {'out':tf.Variable(tf.random_normal([n_output]))}

def RNN(x,weight, biases):

    # x = tf.transpose(x,)
    x = tf.transpose(x, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, n_input)

    x = tf.reshape(x, [-1, n_input])

    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(0, n_steps, x)

    lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']


pred = RNN(x, weights,biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()


# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step  < training_iters:

        a_int = np.random.randint(large_number/2)
        a = binary[a_int]
        b_int = np.random.randint(large_number/2)
        b = binary[a_int]


        c_int = a_int+b_int

        c = binary[c_int]


        x_input = np.array([[a,b]])
        y_output = np.array([c]).T

        # # Reshape data to get 28 seq of 28 elements
        # batch_x = batch_x.reshape((batch_size, n_steps, n_input))
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: x_input, y: y_output})
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: x_input, y: y_output})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: x_input, y: y_output})
            print "Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc)
        step += 1
    print "Optimization Finished!"
    #
    # # Calculate accuracy for 128 mnist test images
    # test_len = 128
    # test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    # test_label = mnist.test.labels[:test_len]
    # print "Testing Accuracy:", \
    #     sess.run(accuracy, feed_dict={x: test_data, y: test_label})
