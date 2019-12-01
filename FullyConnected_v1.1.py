#!/usr/bin/env python
# __author__ = 'neevsamuel'
import tensorflow as tf
import numpy as np
import time as tm

total_size_of_data = 20000
size_of_x = 20
size_of_received_vector_y = 30
B = 1000
train_iter = 1000000
test_iter = 1000
low_snr_db_train = 7.0
high_snr_db_train = 14.0
low_snr_db_test = 8.0
high_snr_db_test = 13.0
num_snr = 6
fc_size = 200
num_of_hidden_layers = 4
startingLearningRate = 0.0003
decay_factor = 0.97
decay_step_size = 1000
H = np.genfromtxt('Top06_30_20.csv', dtype=None, delimiter=',')

low_snr_train = 10.0 ** (low_snr_db_train / 10.0)
high_snr_train = 10.0 ** (high_snr_db_train / 10.0)
low_snr_test = 10.0 ** (low_snr_db_test / 10.0)
high_snr_test = 10.0 ** (high_snr_db_test / 10.0)
bers = np.zeros((1, num_snr))


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.05)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def Generate_data(total_size_of_data, size_of_x, size_of_received_vector_y, snr_low, snr_high, H_org):
    W_ = np.zeros([total_size_of_data, size_of_x, size_of_x])

    #rand Create an array of the given shape and populate it with random samples from a uniform distribution over [0, 1)
    #The sign function returns -1 if x < 0, 0 if x==0, 1 if x > 0
    x_ = np.sign(np.random.rand(total_size_of_data, size_of_x) - 0.5)
    y_ = np.zeros([total_size_of_data, size_of_received_vector_y])

    #randn return a sample (or samples) from the “standard normal” distribution
    w = np.random.randn(total_size_of_data, size_of_received_vector_y)
    Hy_ = x_ * 0
    H_ = np.zeros([total_size_of_data, size_of_received_vector_y, size_of_x])
    HH_ = np.zeros([total_size_of_data, size_of_x, size_of_x])
    SNR_ = np.zeros([total_size_of_data])

    for i in range(total_size_of_data):
        SNR = np.random.uniform(low=snr_low, high=snr_high)
        H = H_org

        tmp_snr = np.trace(np.dot(np.transpose(H),H)) / size_of_x
        H_[i, :, :] = H

        y_[i, :] = np.dot(H,x_[i, :]) + (w[i, :] * np.sqrt(tmp_snr) / np.sqrt(SNR))

        Hy_[i, :] = np.dot(np.transpose(H),y_[i, :])

        HH_[i, :, :] = np.dot(np.transpose(H),H_[i, :, :])

        SNR_[i] = SNR
    #return y_, H_, Hy_, HH_, x_, SNR
    return y_,x_

y_, x_ = Generate_data(total_size_of_data, size_of_x, size_of_received_vector_y, low_snr_train, high_snr_train, H)

NNinput = tf.placeholder(tf.float32, shape=[None, size_of_received_vector_y], name='input')
org_signal = tf.placeholder(tf.float32, shape=[None, size_of_x], name='org_siganl')
batchSize = tf.placeholder(tf.int32)

##### The network ######
### Initializing network
h_fc = []
W_fc_input = weight_variable([size_of_received_vector_y, fc_size])
b_fc_input = bias_variable([fc_size])
h_fc.append(tf.nn.relu(tf.matmul(NNinput, W_fc_input) + b_fc_input))

### Hidden layers

def affine_layer(x, input_size, output_size, Layer_num):
    W = tf.Variable(tf.random_normal([input_size, output_size], stddev=0.01))
    w = tf.Variable(tf.random_normal([1, output_size], stddev=0.01))
    y = tf.matmul(x, W) + w
    return y

def relu_layer(x, input_size, output_size, Layer_num):
    y = tf.nn.relu(affine_layer(x, input_size, output_size, Layer_num))
    return y

### Where did we get h_fc[i-1]
for i in range(1,num_of_hidden_layers):
    print('I am i:',i)
    h_fc.append(relu_layer(h_fc[i - 1], fc_size, fc_size, 'relu' + str(i)))

### Last layer

def Neural_network(x):
    affine_layer()
    relu_layer()

    W_fc_final = weight_variable([fc_size, size_of_x])
    b_final = bias_variable([size_of_x])
    estimated_signal = tf.matmul(h_fc[i], W_fc_final) + b_final

    return estimated_signal

##### Error estimation
estimated_signal = Neural_network
estimated_error = tf.reduce_sum(tf.square(org_signal - estimated_signal))

##### Optimization
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(startingLearningRate, global_step, decay_step_size, decay_factor,
                                           staircase=True)
training_step = tf.train.AdamOptimizer(learning_rate).minimize(estimated_error)

##### Calculation of accuracy
val = tf.reshape(org_signal, tf.stack([size_of_x * batchSize]))
final = tf.reshape(estimated_signal, tf.stack([size_of_x * batchSize]))
rounded = tf.sign(final)
eq = tf.equal(rounded, val)
accuracy = tf.reduce_sum(tf.cast(eq, tf.int32))


##### Initializing tensorflow session #####
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

##### training phase og the network #####
for i in range(train_iter):
    batch_Y, batch_H, batch_HY, batch_HH, batch_X, SNR1 = Generate_data(B, size_of_x, size_of_received_vector_y, low_snr_train, high_snr_train, H)

    if i % 100 == 0:
        training_accuracy = accuracy.eval(feed_dict={NNinput: batch_Y, org_signal: batch_X, batchSize: B})

        current_estimated_error = estimated_error.eval(feed_dict={NNinput: batch_Y, org_signal: batch_X, batchSize: B})

        print("step %d, loss is %g, number of correct bits %d" % (i, current_estimated_error, training_accuracy))

    training_step.run(feed_dict={NNinput: batch_Y, org_signal: batch_X, batchSize: B})


##### Testing the Network #####
tmp_bers = np.zeros((1, test_iter))
tmp_times = np.zeros((1, test_iter))
times = np.zeros((1, 1))
testHitCount = 0

snr_list_db = np.linspace(low_snr_db_test, high_snr_db_test, num_snr)
snr_list = 10.0 ** (snr_list_db / 10.0)

for i_snr in range(num_snr):
    Cur_SNR = snr_list[i_snr]
    print('cur snr')
    print(Cur_SNR)
    for i in range(test_iter):
        batch_Y, batch_H, batch_HY, batch_HH, batch_X, SNR1 = Generate_data(B, size_of_x, size_of_received_vector_y, Cur_SNR, Cur_SNR, H)
        tic = tm.time()
        tmp_bers[0][i] = accuracy.eval(feed_dict={NNinput: batch_Y, org_signal: batch_X, batchSize: B})
        toc = tm.time()
        tmp_times[0][i] = toc - tic
        if i % 100 == 0:
            accuracy.eval(feed_dict={NNinput: batch_Y, org_signal: batch_X, batchSize: B})
            current_estimated_error = accuracy.eval(feed_dict={NNinput: batch_Y, org_signal: batch_X, batchSize: B})
            print("test accuracy %g" % accuracy.eval(feed_dict={NNinput: batch_Y, org_signal: batch_X, batchSize: B}))

    bers[0][i_snr] = np.mean(tmp_bers[0])

times[0][0] = np.mean(tmp_times[0]) / B
print('Average time to detect a single K bit signal is:')
print(times)
bers = bers / (size_of_x * B)
bers = 1 - bers
snrdb_list = np.linspace(low_snr_db_test, high_snr_db_test, num_snr)
print('snrdb_list')
print(snrdb_list)
print('Bit error rates are is:')
print(bers)
