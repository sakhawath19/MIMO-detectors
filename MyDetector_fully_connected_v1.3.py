# !/usr/bin/env python

"""
Fully Connected network has been updated here
Last layer is splitted into user size and used softmax to classify signal
Second last layer also splitted into four equal parts
"""
"""
Parameters
K - size of x
N - size of y
snrdb_low - the lower bound of noise db used during training
snr_high - the higher bound of noise db used during training
L - number of layers in DetNet
v_size = size of auxiliary variable at each layer
hl_size - size of hidden layer at each DetNet layer (the dimention the layers input are increased to
startingLearningRate - the initial step size of the gradient descent algorithm
decay_factor & decay_step_size - each decay_step_size steps the learning rate decay by decay_factor
train_iter - number of train iterations
train_batch_size - batch size during training phase
test_iter - number of test iterations
test_batch_size  - batch size during testing phase
LOG_LOSS - equal 1 if loss of each layer should be sumed in proportion to the layer depth, otherwise all losses have the 
same weight 
res_alpha- the proportion of the previuos layer output to be added to the current layers output (view ResNet article)
snrdb_low_test & snrdb_high_test & num_snr - when testing, num_snr different SNR values will be tested, uniformly spread 
between snrdb_low_test and snrdb_high_test 
"""

"""
Data generation for train and test phases
In this example, both functions are the same.
This duplication is in order to easily allow testing cases where the test is over different distributions of data than 
in the training phase.
e.g. training over gaussian i.i.d. channels and testing over a specific constant channel.
currently both test and train are over i.i.d gaussian channel.
"""


import tensorflow as tf
import numpy as np
import time as tm
import matplotlib.pyplot as plt


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.05)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def constellation_alphabet(mod):
    if mod == 'BPSK':
        return np.array([[-1, 1]])
    elif mod == 'QPSK':
        return np.array([[-1, 1]])
    elif mod == '16QAM':
        return np.array([[-3, -1, 1, 3]])
    elif mod == '64QAM':
        return np.array([[-7, -5, -3, -1, 1, 3, 5, 7]])


def generate_data_train(B, K, N, snr_low, snr_high, H_org):
    # H_ = np.random.randn(B, N, K)
    # W_ = np.zeros([B, K, K])
    rand_symbol_ind = (np.random.randint(low = 0, high = CONS_ALPHABET.shape[1], size = (B*K, 1))).flatten()
    transmitted_symbol = (CONS_ALPHABET[:, rand_symbol_ind])

    x_ = transmitted_symbol.reshape(B, K)

    length_one_hot_vector = CONS_ALPHABET.shape[1]
    x_one_hot = generate_one_hot(x_, B, K, length_one_hot_vector)
    y_ = np.zeros([B, N])
    w = np.random.randn(B, N)
    H_ = np.zeros([B, N, K])
    Hy_ = x_ * 0
    HH_ = np.zeros([B, K, K])
    SNR_ = np.zeros([B])
    for i in range(B):
        SNR = np.random.uniform(low=snr_low, high=snr_high)
        H = H_org
        #H = H_[i, :, :]
        tmp_snr = (H.T.dot(H)).trace() / K
        H_[i, :, :] = H
        y_[i, :] = (H.dot(x_[i, :]) + w[i, :] * np.sqrt(tmp_snr) / np.sqrt(SNR))
        Hy_[i, :] = H.T.dot(y_[i, :])
        HH_[i, :, :] = H.T.dot(H_[i, :, :])
        SNR_[i] = SNR
    return y_, H_, Hy_, HH_, x_, SNR_, x_one_hot


def generate_data_iid_test(B, K, N, snr_low,snr_high):
    H_=np.random.randn(B, N, K)
    # W_=np.zeros([B,K,K])
    x_= np.sign(np.random.rand(B, K)-0.5)
    y_= np.zeros([B,N])
    w = np.random.randn(B,N)
    Hy_ = x_ * 0
    HH_ = np.zeros([B, K, K])
    SNR_ = np.zeros([B])

    for i in range(B):
        SNR = np.random.uniform(low=snr_low,high=snr_high)
        H = H_[i,:,:]
        tmp_snr =(H.T.dot(H)).trace()/K
        H_[i,:,:] = H
        y_[i,:] = (H.dot(x_[i,:])+w[i,:]*np.sqrt(tmp_snr)/np.sqrt(SNR))
        Hy_[i,:] = H.T.dot(y_[i,:])
        HH_[i,:,:] = H.T.dot( H_[i,:,:])
        SNR_[i] = SNR
        print(y_)
    return y_,H_,Hy_,HH_,x_,SNR_


def generate_one_hot(symbol, B, K, length_one_hot):
    depth_one_hot_vector = CONS_ALPHABET.shape[1]
    reset_symbol = symbol + np.multiply(np.ones([B, K]), (abs(np.amin(CONS_ALPHABET, 1)) + 1))
    one_hot_vector = tf.one_hot(reset_symbol, depth_one_hot_vector, on_value=1.0, off_value=0.0, axis=-1)
    one_hot_arr = sess.run(one_hot_vector)
    one_hot_vector_reshaped = one_hot_arr.reshape(B, K, length_one_hot)
    return one_hot_vector_reshaped


def hidden_layer(x,input_size,output_size,Layer_num):
    W = tf.Variable(tf.random_normal([input_size, output_size], stddev=0.01))
    w = tf.Variable(tf.random_normal([1, output_size], stddev=0.01))
    y = tf.matmul(x, W)+w
    return y


def activation_fn(x,input_size,output_size,Layer_num):
    y = tf.nn.relu(hidden_layer(x,input_size,output_size,Layer_num))
    return y

'''..................................................................................................................'''
sess = tf.InteractiveSession()

# parameters
user = 20 # 8  # 20
antenna = 30 # 16  # 30

num_snr = 6
low_snr_db_train = 7.0
high_snr_db_train = 14.0
low_snr_db_test = 7.0  # 8.0
high_snr_db_test = 14.0  # 13.0

low_snr_train = 10.0 ** (low_snr_db_train/10.0)
high_snr_train = 10.0 ** (high_snr_db_train/10.0)
low_snr_test = 10.0 ** (low_snr_db_test/10.0)
high_snr_test = 10.0 ** (high_snr_db_test/10.0)

batch_size = 100  # 1000
train_iter = 10000  # 1000000
test_iter = 1000
fc_size = 200  # user * user * user # 200
num_of_hidden_layers = 5  # user
startingLearningRate = .0003  # 0.0003
decay_factor = 0.97  # 0.97
decay_step_size = 1000

H = np.genfromtxt('Top06_30_20.csv', dtype=None, delimiter=',')

CONS_ALPHABET = constellation_alphabet('QPSK')
length_one_hot_vector = CONS_ALPHABET.shape[1]


received_sig = tf.placeholder(tf.float32, shape=[None, user], name='input')
#received_sig = tf.placeholder(tf.float32, shape=[None, antenna], name='input')
transmitted_sig = tf.placeholder(tf.float32, shape=[None, user], name='org_siganl')
batchSize = tf.placeholder(tf.int32)
batch_x_one_hot = tf.placeholder(tf.float32, shape=[None, user, length_one_hot_vector], name='one_hot_org_siganl')

# The network
h_fc = []
W_fc_input = weight_variable([user, fc_size])
#W_fc_input = weight_variable([antenna, fc_size])
b_fc_input = bias_variable([fc_size])
h_fc.append(tf.nn.relu(tf.matmul(received_sig, W_fc_input) + b_fc_input))

for i in range(num_of_hidden_layers):
    h_fc.append(activation_fn(h_fc[i - 1], fc_size, fc_size, 'relu' + str(i)))

W_fc_1_last_final_1 = weight_variable([fc_size, fc_size])
W_fc_1_last_final_2 = weight_variable([fc_size, fc_size])
W_fc_1_last_final_3 = weight_variable([fc_size, fc_size])
W_fc_1_last_final_4 = weight_variable([fc_size, fc_size])

b_1_last_final_1 = bias_variable([fc_size])
b_1_last_final_2= bias_variable([fc_size])
b_1_last_final_3 = bias_variable([fc_size])
b_1_last_final_4 = bias_variable([fc_size])

h_1_last_final_1 = tf.matmul(h_fc[i], W_fc_1_last_final_1) + b_1_last_final_1
h_1_last_final_2 = tf.matmul(h_fc[i], W_fc_1_last_final_2) + b_1_last_final_2
h_1_last_final_3 = tf.matmul(h_fc[i], W_fc_1_last_final_3) + b_1_last_final_3
h_1_last_final_4 = tf.matmul(h_fc[i], W_fc_1_last_final_4) + b_1_last_final_4

fc_1_final_1 = tf.concat([h_1_last_final_1, h_1_last_final_2], 1)
fc_1_final_2 = tf.concat([fc_1_final_1, h_1_last_final_3], 1)
fc_1_final = tf.concat([fc_1_final_2, h_1_last_final_4], 1)


W_fc_2_last_final_1 = weight_variable([fc_size * 4, fc_size])
W_fc_2_last_final_2 = weight_variable([fc_size * 4, fc_size])

b_2_last_final_1 = bias_variable([fc_size])
b_2_last_final_2= bias_variable([fc_size])

h_2_last_final_1 = tf.matmul(fc_1_final, W_fc_2_last_final_1) + b_2_last_final_1
h_2_last_final_2 = tf.matmul(fc_1_final, W_fc_2_last_final_2) + b_2_last_final_2

fc_2_final = tf.concat([h_2_last_final_1, h_2_last_final_2], 1)


W_fc_3_last_final_1 = weight_variable([fc_size * 2, fc_size])
b_3_last_final_1 = bias_variable([fc_size])
fc_3_final = tf.matmul(fc_2_final, W_fc_3_last_final_1) + b_3_last_final_1


W_fc_4_last_final_1 = weight_variable([fc_size, length_one_hot_vector * user])
b_4_last_final_1 = bias_variable([length_one_hot_vector * user])
fc_4_final = tf.matmul(fc_3_final, W_fc_4_last_final_1) + b_4_last_final_1

'''
W_fc_second_last_final = weight_variable([fc_size, length_one_hot_vector * user])
b_second_last_final = bias_variable([length_one_hot_vector * user])
h_second_last_final = tf.matmul(h_fc[i], W_fc_second_last_final) + b_second_last_final
'''

h_final = tf.reshape(fc_4_final, [batchSize, user, length_one_hot_vector])

final_output = tf.nn.softmax(h_final, axis=2)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=final_output, labels=batch_x_one_hot))
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(startingLearningRate, global_step, decay_step_size, decay_factor,
                                           staircase=True)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

com_received_transmitted_sig = tf.equal(tf.argmax(final_output, 2), tf.argmax(batch_x_one_hot, 2))
accuracy = tf.reduce_sum(tf.cast(com_received_transmitted_sig, 'float')) / (user * batch_size)

cost_accumulated = []
accuracy_accumulated = []
sess.run(tf.global_variables_initializer())

# train the network

for i in range(train_iter):
    batch_Y, batch_H, batch_HY, batch_HH, batch_X, SNR1, one_hot = generate_data_train(batch_size, user, antenna, low_snr_train, high_snr_train, H)
    _, c = sess.run([train_step, loss], feed_dict={received_sig: batch_HY, transmitted_sig: batch_X, batchSize: batch_size, batch_x_one_hot: one_hot})
    ac = sess.run([accuracy], feed_dict={received_sig: batch_HY, transmitted_sig: batch_X, batchSize: batch_size, batch_x_one_hot: one_hot})

    if i % 100 == 0:
        print('Training teration', i, '|    ', '    Accuracy:', ac[0],'|    ' '     Loss:', c)
        cost_accumulated.extend([c])
        accuracy_accumulated.extend([ac])


# test the network
bers = np.zeros((1, num_snr))
tmp_bers = np.zeros((1, test_iter))
tmp_times = np.zeros((1, test_iter))
times = np.zeros((1, 1))
test_accuracy_accumulated = []

snr_list_db = np.linspace(low_snr_db_test, high_snr_db_test, num_snr)
snr_list = 10.0 ** (snr_list_db / 10.0)

for i_snr in range(num_snr):
    cur_SNR = snr_list[i_snr]
    print('Current SNR', cur_SNR)
    for i in range(test_iter):
        batch_Y, batch_H, batch_HY, batch_HH, batch_X, SNR1, one_hot = generate_data_train(batch_size, user, antenna, low_snr_test, high_snr_test, H)
        tic = tm.time()
        test_ac = sess.run([accuracy], feed_dict={received_sig: batch_Y, transmitted_sig: batch_X, batchSize: batch_size, batch_x_one_hot: one_hot})
        tmp_bers[0][i] = test_ac[0]
        toc = tm.time()
        tmp_times[0][i] = toc - tic
        if i % 100 == 0:
            test_ac = sess.run([accuracy], feed_dict={received_sig: batch_Y, transmitted_sig: batch_X, batchSize: batch_size, batch_x_one_hot: one_hot})
            print("Test accuracy", test_ac)

    bers[0][i_snr] = np.mean(tmp_bers[0])

bers = 1 - bers
times[0][0] = np.mean(tmp_times[0]) / batch_size
snrdb_list = np.linspace(low_snr_db_test, high_snr_db_test, num_snr)

print('Average time to detect a single K bit signal is:', times)
print('snrdb_list:', snrdb_list)
print('Bit error rates are is:', bers)

fig2 = plt.figure('BPSK, TxR=16x32, Iteration=1000,SNR=7-14')
#fig2 = plt.figure('Python, 64QAM, TxR=16x32, Iteration=100000,SNR=10-3-38')
#fig2 = plt.figure('Python, 64QAM, TxR=16x32, Iteration=100000,SNR=-2-2-16, WRONG RESULT')
ax2 = fig2.add_subplot(111)
ax2.plot(snrdb_list.reshape(-1), bers.reshape(-1), color='black', marker='*', linestyle='-', linewidth=1, markersize=6, label='FC')
#ax2.plot(snr_db_list, bers_mmse, color='black', marker='*', linestyle='-', linewidth=1, markersize=6, label='MMSE')
#ax2.plot(snr_db_list, bers_zf, color='blue', marker='d', linestyle='-', linewidth=1, markersize=5, label='ZF')
#ax2.plot(snr_db_list, bers_mf, color='red', marker='x', linestyle='-', linewidth=1, markersize=6, label='MF')
#ax2.plot(snr_db_list, bers_sd, color='magenta', marker='s', linestyle='-', linewidth=1, markersize=6, label='SD')
ax2.set_title('BER vs SNR')
ax2.set_xlabel('SNR(dB)')
ax2.set_ylabel('BER')
ax2.set_yscale('log')
ax2.set_ylim(0.0001, 1)
ax2.set_xlim(8, 13)
plt.grid(b=True, which='major', color='#666666', linestyle='--')
plt.legend(title='Detectors:')
plt.show()