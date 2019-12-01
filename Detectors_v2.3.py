#!/usr/bin/env python
import tensorflow as tf
import numpy as np
import time as tm
import math
import sys
import pickle as pkl
import matplotlib.pyplot as plt
import matplotlib.path as mpath


sess = tf.InteractiveSession()

# parameters
K = 16#20
USER = K
N = 32#30
snrdb_low = 7.0
snrdb_high = 14.0
snr_low = 10.0 ** (snrdb_low / 10.0)
snr_high = 10.0 ** (snrdb_high / 10.0)
L = 10  # 90
v_size = 2 * K
hl_size = 8 * K
startingLearningRate = 0.0001
decay_factor = 0.97
decay_step_size = 1000
train_iter = 10#1000  # 10000#20000
train_batch_size = 1000  # 5000
test_iter = 10 #200  # 10000#200
test_batch_size = 10#100
LOG_LOSS = 1
res_alpha = 0.9
num_snr = 6
snrdb_low_test = 8.0
snrdb_high_test = 13.0



def generate_data_iid_test(B, K, N, snr_low, snr_high):
    H_ = np.random.randn(B, N, K)
    W_ = np.zeros([B, K, K])
    x_ = np.sign(np.random.rand(B, K) - 0.5)
    y_ = np.zeros([B, N])
    w = np.random.randn(B, N)
    Hy_ = x_ * 0
    HH_ = np.zeros([B, K, K])
    SNR_ = np.zeros([B])

    for i in range(B):
        SNR = np.random.uniform(low=snr_low, high=snr_high)
        H = H_[i, :, :]
        tmp_snr = (H.T.dot(H)).trace() / K
        H_[i, :, :] = H
        y_[i, :] = (H.dot(x_[i, :]) + w[i, :] * np.sqrt(tmp_snr) / np.sqrt(SNR))
        Hy_[i, :] = H.T.dot(y_[i, :])
        HH_[i, :, :] = H.T.dot(H_[i, :, :])
        SNR_[i] = SNR
        print(y_)
    return y_, H_, Hy_, HH_, x_, SNR_


def generate_data_train(B, K, N, snr_low, snr_high):
    H_ = np.random.randn(B, N, K)
    W_ = np.zeros([B, K, K])
    x_ = np.sign(np.random.rand(B, K) - 0.5)
    y_ = np.zeros([B, N])
    w = np.random.randn(B, N)
    Hy_ = x_ * 0
    HH_ = np.zeros([B, K, K])
    SNR_ = np.zeros([B])
    for i in range(B):
        SNR = np.random.uniform(low=snr_low, high=snr_high)
        H = H_[i, :, :]
        tmp_snr = (H.T.dot(H)).trace() / K
        H_[i, :, :] = H
        y_[i, :] = (H.dot(x_[i, :]) + w[i, :] * np.sqrt(tmp_snr) / np.sqrt(SNR))
        Hy_[i, :] = H.T.dot(y_[i, :])
        HH_[i, :, :] = H.T.dot(H_[i, :, :])
        SNR_[i] = SNR
    return y_, H_, Hy_, HH_, x_, SNR_


def piecewise_linear_soft_sign(x):
    t = tf.Variable(0.1)
    y = -1 + tf.nn.relu(x + t) / (tf.abs(t) + 0.00001) - tf.nn.relu(x - t) / (tf.abs(t) + 0.00001)
    return y


def affine_layer(x, input_size, output_size, Layer_num):
    W = tf.Variable(tf.random_normal([input_size, output_size], stddev=0.01))
    w = tf.Variable(tf.random_normal([1, output_size], stddev=0.01))
    y = tf.matmul(x, W) + w
    return y


def relu_layer(x, input_size, output_size, Layer_num):
    y = tf.nn.relu(affine_layer(x, input_size, output_size, Layer_num))
    return y


def sign_layer(x, input_size, output_size, Layer_num):
    y = piecewise_linear_soft_sign(affine_layer(x, input_size, output_size, Layer_num))
    return y


# tensorflow placeholders, the input given to the model in order to train and test the network
HY = tf.placeholder(tf.float32, shape=[None, K])
X = tf.placeholder(tf.float32, shape=[None, K])
HH = tf.placeholder(tf.float32, shape=[None, K, K])

batch_size = tf.shape(HY)[0]
X_LS = tf.matmul(tf.expand_dims(HY, 1), tf.matrix_inverse(HH))
X_LS = tf.squeeze(X_LS, 1)
loss_LS = tf.reduce_mean(tf.square(X - X_LS))
ber_LS = tf.reduce_mean(tf.cast(tf.not_equal(X, tf.sign(X_LS)), tf.float32))

S = []
S.append(tf.zeros([batch_size, K]))
V = []
V.append(tf.zeros([batch_size, v_size]))
LOSS = []
LOSS.append(tf.zeros([]))
BER = []
BER.append(tf.zeros([]))

# The architecture of DetNet
for i in range(1, L):
    temp1 = tf.matmul(tf.expand_dims(S[-1], 1), HH)
    temp1 = tf.squeeze(temp1, 1)
    Z = tf.concat([HY, S[-1], temp1, V[-1]], 1)
    ZZ = relu_layer(Z, 3 * K + v_size, hl_size, 'relu' + str(i))
    S.append(sign_layer(ZZ, hl_size, K, 'sign' + str(i)))
    S[i] = (1 - res_alpha) * S[i] + res_alpha * S[i - 1]
    V.append(affine_layer(ZZ, hl_size, v_size, 'aff' + str(i)))
    V[i] = (1 - res_alpha) * V[i] + res_alpha * V[i - 1]
    if LOG_LOSS == 1:
        LOSS.append(np.log(i) * tf.reduce_mean(
            tf.reduce_mean(tf.square(X - S[-1]), 1) / tf.reduce_mean(tf.square(X - X_LS), 1)))
    else:
        LOSS.append(tf.reduce_mean(tf.reduce_mean(tf.square(X - S[-1]), 1) / tf.reduce_mean(tf.square(X - X_LS), 1)))
    BER.append(tf.reduce_mean(tf.cast(tf.not_equal(X, tf.sign(S[-1])), tf.float32)))

TOTAL_LOSS = tf.add_n(LOSS)

global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(startingLearningRate, global_step, decay_step_size, decay_factor,
                                           staircase=True)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(TOTAL_LOSS)
init_op = tf.global_variables_initializer()

sess.run(init_op)
# Training DetNet
for i in range(train_iter):  # num of train iter
    batch_Y, batch_H, batch_HY, batch_HH, batch_X, SNR1 = generate_data_train(train_batch_size, K, N, snr_low, snr_high)
    train_step.run(feed_dict={HY: batch_HY, HH: batch_HH, X: batch_X})
    if i % 100 == 0:
        batch_Y, batch_H, batch_HY, batch_HH, batch_X, SNR1 = generate_data_iid_test(train_batch_size, K, N, snr_low,
                                                                                     snr_high)
        results = sess.run([loss_LS, LOSS[L - 1], ber_LS, BER[L - 1]], {HY: batch_HY, HH: batch_HH, X: batch_X})
        print_string = [i] + results
        print('Training iteration:', i)
        # this print line need to rewrite
        # print (' ').join('%s' % x for x in print_string)



def constellation_alphabet(mod):
    if mod == 'BPSK':
        return np.array([[-1, 1]], np.complex)
    elif mod == 'QPSK':
        return np.array([[-1-1j, -1+1j, 1-1j, 1+1j]], np.complex)
    elif mod == '16QAM':
        return np.array([[-3 - 3j, -3 - 1j, -3 + 3j, -3 + 1j, - 1 - 3j, -1 - 1j, -1 + 3j, -1 + 1j, + 3 - 3j, +3 - 1j,
                          +3 + 3j, +3 + 1j, + 1 - 3j, +1 - 1j, +1 + 3j, +1 + 1j]], np.complex)
    elif mod == '64QAM':
        return np.array([[-7 - 7j, -7 - 5j, -7 - 1j, -7 - 3j, -7 + 7j, -7 + 5j, -7 + 1j, -7 + 3j, - 5 - 7j, -5 - 5j,
                          -5 - 1j, -5 - 3j, -5 + 7j, -5 + 5j, -5 + 1j, -5 + 3j,- 1 - 7j, -1 - 5j, -1 - 1j, -1 - 3j,
                          -1 + 7j, -1 + 5j, -1 + 1j, -1 + 3j, - 3 - 7j, -3 - 5j, -3 - 1j, -3 - 3j, -3 + 7j, -3 + 5j,
                          -3 + 1j, -3 + 3j, + 7 - 7j, +7 - 5j, +7 - 1j, +7 - 3j, +7 + 7j, +7 + 5j, +7 + 1j, +7 + 3j,
                          + 5 - 7j, +5 - 5j, +5 - 1j, +5 - 3j, +5 + 7j, +5 + 5j, +5 + 1j, +5 + 3j, + 1 - 7j, +1 - 5j,
                          +1 - 1j, +1 - 3j, +1 + 7j, +1 + 5j, +1 + 1j, +1 + 3j, + 3 - 7j, +3 - 5j, +3 - 1j, +3 - 3j,
                          +3 + 7j, +3 + 5j, +3 + 1j, +3 + 3j]], np.complex)


'''CHANGE YOUR PARAMETERS HERE'''
ITERATION = 1#100000
USER = 16
RECEIVER = 32

snr_db_list = []
for snr in range(10, 38, 3):
    snr_db_list.append(snr)

CONS_ALPHABET = constellation_alphabet('BPSK')   # choose anything from here BPSK,QPSK,16QAM,64QAM
'''CHANGEABLE PARAMETERS ARE ABOVE THE LINE'''

signal_energy_avg = np.mean(np.square(np.abs(CONS_ALPHABET)))

snrdb_list = np.linspace(snrdb_low_test, snrdb_high_test, num_snr)
bers_mmse_in_iter = np.zeros([len(snrdb_list), test_iter * test_batch_size])
#bers_mmse_in_iter = np.zeros([len(snr_db_list), ITERATION])

'''Testing the model'''

# Testing the trained model
#snrdb_list = np.linspace(snrdb_low_test, snrdb_high_test, num_snr)
snr_list = 10.0 ** (snrdb_list / 10.0)
bers = np.zeros((1, num_snr))
times = np.zeros((1, num_snr))
tmp_bers = np.zeros((1, test_iter))
tmp_times = np.zeros((1, test_iter))
for j in range(num_snr):
    for jj in range(test_iter):
        print('snr:')
        print(snrdb_list[j])
        print('test iteration:')
        print(jj)
        batch_Y, batch_H, batch_HY, batch_HH, batch_X, SNR1 = generate_data_iid_test(test_batch_size, K, N, snr_list[j],
                                                                                     snr_list[j])
        tic = tm.time()
        tmp_bers[:, jj] = np.array(sess.run(BER[L - 1], {HY: batch_HY, HH: batch_HH, X: batch_X}))
        toc = tm.time()
        tmp_times[0][jj] = toc - tic

        for jjj in range(test_batch_size):
         received_signal = batch_Y[jjj, :].reshape(N, 1)
         transmitted_symbol = batch_X[jjj, :].reshape(K, 1)
         channel = batch_H[jjj, :].reshape(N, K)
         noise_variance = 1 * K / snrdb_list[j]
         rerun = test_iter * test_batch_size - 1
         bers_mmse_in_iter[j][rerun] = mmse(transmitted_symbol, noise_variance, channel, received_signal)

    bers[0][j] = np.mean(tmp_bers, 1)
    times[0][j] = np.mean(tmp_times[0]) / test_batch_size


bers_mmse = np.mean(bers_mmse_in_iter, axis=1)
print('mmse error rate', bers_mmse)
bits_per_symbol = math.log2(CONS_ALPHABET.shape[1])
bers_mmse = np.divide(bers_mmse,bits_per_symbol)
print('mmse detector error rate after divided by bits_per_symbol', bers_mmse)

print('snrdb_list')
print(snrdb_list)
print('bers')
print(bers)
print('times')
print(times)

plt.figure('Bit Error Rate')
plt.subplot(111)
plt.semilogy(snrdb_list, bers_mmse, color='black', marker='*', linestyle='-', linewidth=1, markersize=6, label='MMSE')
plt.semilogy(snrdb_list, bers[0], color='green', marker='^', linestyle='-', linewidth=1, markersize=6, label='DetNet')
plt.title('BER vs SNR')
plt.xlabel('average SNR(dB) per receive antenna')
plt.xlim(8, 13)
plt.xscale('linear')
plt.ylabel('BER')
plt.ylim(0.0001, 1)
plt.grid(b=True, which='major', color='#666666', linestyle='--')
plt.legend(title='Detectors:')
plt.show()