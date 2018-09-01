#!/usr/bin/python3

from __future__ import print_function

from generate_sample import generate_sample

import numpy as np
# noinspection PyUnresolvedReferences
import seaborn as sns

import tensorflow as tf
from tensorflow.contrib import rnn



"""
Copyright (c) <2018> <Mohammadreza Baharani, Mehrdad Biglarbeigian>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


"""
A Recurrent Neural Network (LSTM) implementation example using TensorFlow library.
Inspired by
    https://github.com/aymericdamien/TensorFlow-Examples/
    and
    http://mourafiq.com/2016/05/15/predicting-sequences-using-rnn-in-tensorflow.html
    and
    https://github.com/sunsided/tensorflow-lstm-sin
"""


# Parameters
data_file =  "RoIFor5Devs.mat"
batch_size = 4  # because we have four devices to learn, and one device to test
learning_rate = 0.003
training_iters = 1000
training_iter_step_down_every = 250000
display_step = 100
updating_plot = 10

# Network Parameters
n_input = 1  # Delta{R}
n_steps = 20  # time steps
n_hidden = 32  # Num of features
n_outputs = 104  # output is a series of Delta{R}+
n_layers = 4  # number of stacked LSTM layers
save_movie = False
save_res_as_file = True
reach_to_test_error = 5e-10


'''
    Computation graph
'''

with tf.name_scope("INPUTs"):
    # tf Graph input
    lr = tf.placeholder(tf.float32, [])
    x = tf.placeholder(tf.float32, [None, n_steps, n_input])
    y = tf.placeholder(tf.float32, [None, n_outputs])

# Define weights
weights = {
    'out': tf.Variable(tf.truncated_normal([n_hidden, n_outputs], stddev=1.0))
}

biases = {
    'out': tf.Variable(tf.truncated_normal([n_outputs], stddev=0.1))
}

# Define the GRU cells
#with tf.name_scope("GRU_CELL"):
#    gru_cells = [rnn.GRUCell(n_hidden) for _ in range(n_layers)]
#with tf.name_scope("GRU_NETWORK"):
#    stacked_lstm = rnn.MultiRNNCell(gru_cells)

# Define the LSTM cells
with tf.name_scope("LSTM_CELL"):
    lstm_cells = [rnn.LSTMCell(n_hidden, forget_bias=1.) for _ in range(n_layers)]
#with tf.name_scope("LSTM"):
    stacked_lstm = rnn.MultiRNNCell(lstm_cells)

with tf.name_scope("OUTPUTs"):
    outputs, states = tf.nn.dynamic_rnn(stacked_lstm, inputs=x, dtype=tf.float32, time_major=False)
    h = tf.transpose(outputs, [1, 0, 2])
    pred = tf.nn.bias_add(tf.matmul(h[-1], weights['out']), biases['out'])


'''
We don't need this part for inference
'''
# Define loss (Euclidean distance) and optimizer
#individual_losses = tf.reduce_sum(tf.squared_difference(pred, y), reduction_indices=1)
#with tf.name_scope("Loss"):
#    loss = tf.reduce_mean(individual_losses)
#    tf.summary.scalar("loss", loss)

#optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

# Initializing the variables
init = tf.global_variables_initializer()

merged = tf.summary.merge_all()

load_model = True
model_file_name = './model_dev12.ckpt'
saver = tf.train.Saver()

test_device = [1] # 0 -> dev36, 1 -> dev12, 2 -> dev38, 3 -> dev11, 4 -> dev9
TX2_Board_Power = False

config = tf.ConfigProto(
        device_count = {'GPU': 1}
    )

if __name__ == "__main__":

    with tf.Session() as sess:

        if load_model:
            # Restore variables from disk.
            saver.restore(sess, model_file_name)
            print('Model restored.')

        sess.run(init)
        from datetime import datetime


        _, _, _, _, l = generate_sample(filename=data_file, batch_size=1, samples=n_steps, predict=n_outputs, test=True, test_set=test_device)
        total_dev_len = l[0]
        how_many_seg = int(total_dev_len / (n_steps + n_outputs))

        avg_Time=.0
        
        if TX2_Board_Power:
            import socket
            # Create a TCP/IP socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

            # Connect the socket to the port where the server is listening
            server_address = ('localhost', 8080)
            print('connecting to {} port {}'.format(*server_address))
            sock.connect(server_address)
        
            sent_before = False
            message = b'START\n'
            print('sending {!r}'.format(message))
            sock.sendall(message)

        for i in range(how_many_seg):
                tstart = datetime.now()
                t, y, next_t, expected_y,_ = generate_sample(filename=data_file,
                                                           batch_size=1, samples=n_steps, predict=n_outputs,
                                                           start_from=i * (n_steps + n_outputs), test=True, test_set=test_device)
                test_input = y.reshape((1, n_steps, n_input))
                prediction = sess.run(pred, feed_dict={x: test_input})
                tend = datetime.now()
                
                if TX2_Board_Power:
                    if not sent_before:
                        message = b'STOP\n'
                        print('sending {!r}'.format(message))
                        sock.sendall(message)
                        sent_before = True

                # remove the batch size dimensions
                t = t.squeeze()
                y = y.squeeze()
                delta = tend - tstart
                avg_Time += (delta.total_seconds())*1000
                print('Next loop:'+'{}'.format(int(i*100/(how_many_seg-1))), end='\r')

        
        print('Avg elapsed time for predicting one minute: '+ '{}'.format(int(avg_Time/how_many_seg)))
        
        if TX2_Board_Power:
            print('closing socket')
            sock.close()
