from __future__ import print_function

from generate_sample import generate_sample

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
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


fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)  # Plots train and test losses.
ax2 = fig.add_subplot(1, 2, 2)  # Plots prediction
plt.tight_layout(pad=1.5, w_pad=3.0, h_pad=1.0)


# Parameters
learning_rate = 0.003
training_iters = 2500
training_iter_step_down_every = 250000
batch_size = 4  # because we have four devices to learn, and one device to test
display_step = 100
updating_plot = 10

# Network Parameters
n_input = 1  # Delta{R}
n_steps = 20  # time steps
n_hidden = 64  # Num of features
n_outputs = 104  # output is a series of Delta{R}+
n_layers = 4  # number of stacked LSTM layers
save_movie = False
save_res_as_file = True
reach_to_test_error = 6e-5


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

# Define loss (Euclidean distance) and optimizer
individual_losses = tf.reduce_sum(tf.squared_difference(pred, y), reduction_indices=1)
with tf.name_scope("Loss"):
    loss = tf.reduce_mean(individual_losses)
    tf.summary.scalar("loss", loss)

optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

# Initializing the variables
init = tf.global_variables_initializer()

merged = tf.summary.merge_all()


def handle_close(evt):
    if save_res_as_file:
        print("Writing to 'res.txt' file...")
        total_dev36 = 21179
        how_many_seg = int(total_dev36 / (n_steps + n_outputs))

        pred_lst = np.array([])
        for i in range(how_many_seg):
            t, y, next_t, expected_y = generate_sample(filename="RoIFor5Devs.mat",
                                                       batch_size=1, samples=n_steps, predict=n_outputs,
                                                       start_from=i * (n_steps + n_outputs), consider_dev36=True)
            test_input = y.reshape((1, n_steps, n_input))
            prediction = sess.run(pred, feed_dict={x: test_input})
            # remove the batch size dimensions
            pred_lst = np.hstack((pred_lst, y[0]))  # Input Seq
            pred_lst = np.hstack((pred_lst, prediction[0]))  # Prediction

        pred_nump = np.array(pred_lst)
        np.savetxt('res.txt', pred_nump, fmt="%f", newline='\r\n')


fig.canvas.mpl_connect('close_event', handle_close)


def animate(k):

    global step_global

    train(step_global+1)
    if step_global % (updating_plot) == 0:
        ax1.clear()
        ax1.plot(loss_train, color='blue', linestyle='-',
             label='Train', zorder=1)
        ax1.plot(loss_test, color='red', linestyle='-',
             label='Test', zorder=0)
        ax1.legend(loc='upper right')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('MSE')
        ax1.set_ylim(0, 1)
        ax1.set_xlim(0, training_iters)

    if step_global % (updating_plot) == 0:
        ax2.clear()
        total_dev36 = 21179
        how_many_seg = int(total_dev36 / (n_steps + n_outputs))

        for i in range(how_many_seg):
            t, y, next_t, expected_y = generate_sample(filename="RoIFor5Devs.mat",
                                                       batch_size=1, samples=n_steps, predict=n_outputs,
                                                       start_from=i * (n_steps + n_outputs), consider_dev36=True)
            test_input = y.reshape((1, n_steps, n_input))
            prediction = sess.run(pred, feed_dict={x: test_input})
            # remove the batch size dimensions
            t = t.squeeze()
            y = y.squeeze()
            next_t = next_t.squeeze()
            prediction = prediction.squeeze()
            if i == (how_many_seg - 1):
                ax2.plot(t, y, color='black', label='Input')
                ax2.plot(np.append(t[-1], next_t), np.append(y[-1], expected_y), color='green', linestyle='-.',
                            label='Real')
                ax2.plot(np.append(t[-1], next_t), np.append(y[-1], prediction), color='red', linestyle=':',
                            label='Predicted')
                # plt.ylim([-1, 1])
                ax2.legend(loc='upper left')
                ax2.set_xlabel('Samples')
                ax2.set_ylabel('$\Delta R$')
                ax2.set_ylim(-0.003, 0.0550)
            else:
                ax2.plot(t, y, color='black')
                ax2.plot(np.append(t[-1], next_t), np.append(y[-1], expected_y), color='green', linestyle='-.')
                ax2.plot(np.append(t[-1], next_t), np.append(y[-1], prediction), color='red', linestyle=':')
                ax2.set_ylim(-0.003, 0.0550)

    step_global += 1


loss_test = []
loss_train = []


def train(step):
    current_learning_rate = learning_rate
    current_learning_rate *= 0.1 ** ((step * batch_size) // training_iter_step_down_every)

    _, batch_x, __, batch_y = generate_sample(filename="RoIFor5Devs.mat",
                                              batch_size=batch_size, samples=n_steps, predict=n_outputs)

    batch_x = batch_x.reshape((batch_size, n_steps, n_input))
    batch_y = batch_y.reshape((batch_size, n_outputs))

    # Run optimization op (backprop)
    _, loss_value, summary = sess.run([optimizer, loss, merged], feed_dict={x: batch_x, y: batch_y,
                                                                            lr: current_learning_rate})
    writer.add_summary(summary, step)

    _, batch_x_test, __, batch_y_test = generate_sample(filename="RoIFor5Devs.mat",
                                                        batch_size=1, samples=n_steps, predict=n_outputs,
                                                        consider_dev36=True)

    batch_x_test = batch_x_test.reshape((1, n_steps, n_input))
    batch_y_test = batch_y_test.reshape((1, n_outputs))

    loss_value_test, summary = sess.run([loss, merged], feed_dict={x: batch_x_test, y: batch_y_test})

    loss_test.append(loss_value_test)
    loss_train.append(loss_value)

    if step % display_step == 0:
        print("Epoch:" + "{}".format(step) + " Iteration " + str(step * batch_size) + ", Minibatch Loss= " +
              "{:.6f}".format(loss_value) + ", Minibatch Loss Test= " +
              "{:.6f}".format(loss_value_test))

    if not save_movie:
        if loss_value <= reach_to_test_error:
            plt.close('all')  # End the optimization and save result into file if it is activated.


# Launch the graph
step_global = 0
ani = None  # to make it global

if __name__ == "__main__":
    with tf.Session() as sess:

        sess.run(init)

        loss_value = float('+Inf')
        loss_value_test = float('+Inf')

        writer = tf.summary.FileWriter("./output", sess.graph)
        how_many_frame = int(training_iters)+1
        ani = animation.FuncAnimation(fig, animate, range(1, how_many_frame), interval=16, blit=False, repeat=False)
        writer.close()

        if save_movie:
            mywriter = animation.FFMpegWriter(fps=24, codec='libx264',
                                              extra_args=['-pix_fmt', 'yuv420p', '-profile:v', 'high', '-tune', 'animation',
                                                          '-crf', '18'])
            ani.save("./deltaR_video.mp4", writer=mywriter)
        else:
            plt.show()
