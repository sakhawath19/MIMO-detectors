#!/usr/bin/env python
import tensorflow as tf
import numpy as np
import time as tm
import math
import matplotlib.pyplot as plt
from scipy import linalg


###start here
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
LOG_LOSS - equal 1 if loss of each layer should be sumed in proportion to the layer depth, otherwise all losses have the same weight 
res_alpha- the proportion of the previuos layer output to be added to the current layers output (view ResNet article)
snrdb_low_test & snrdb_high_test & num_snr - when testing, num_snr different SNR values will be tested, uniformly spread between snrdb_low_test and snrdb_high_test 

"""
sess = tf.InteractiveSession()

# parameters
ITERATION = 1 #100000
USER = 16
RECEIVER = 32
K = 16  # 20
N = 32  # 30
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
train_iter = 10  # 1000  # 10000#20000
train_batch_size = 10  # 5000
test_iter = 10  # 200  # 10000 #200
test_batch_size = 10  # 100
LOG_LOSS = 1
res_alpha = 0.9
num_snr = 6
snrdb_low_test = 8.0
snrdb_high_test = 13.0

"""Data generation for train and test phases
In this example, both functions are the same.
This duplication is in order to easily allow testing cases where the test is over different distributions of data than in the training phase.
e.g. training over gaussian i.i.d. channels and testing over a specific constant channel.
currently both test and train are over i.i.d gaussian channel.
"""


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
        #print(y_)
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
# init_op=tf.initialize_all_variables()
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


'''Code from Detectors_1.6'''
def mmse(s, n0, h, y):
    p1 = np.matmul(np.matrix.getH(h), y)
    p2 = np.matmul(np.matrix.getH(h), h) + (n0 / signal_energy_avg) * np.identity(USER)
    xhat = np.matmul(np.linalg.inv(p2), p1)

    v1 = np.matmul(xhat, np.ones([1, CONS_ALPHABET.size]))
    v2 = np.matmul(np.ones([USER, 1]), CONS_ALPHABET)
    idxhat = np.argmin(np.square(np.abs(v1 - v2)), axis=1)

    estimated_symbol = CONS_ALPHABET[:, idxhat]
    accuracy_mmse = np.equal(estimated_symbol.flatten(), np.transpose(s))

    error_mmse = 1 - (np.sum(accuracy_mmse) / USER)
    #error_mmse = 1 - (np.sum(accuracy_mmse) / (USER * CONS_ALPHABET.shape[1]))

    return error_mmse

def zero_forcing(s, h, y):
    p1 = np.matmul(np.matrix.getH(h), h)
    p2 = np.matmul(np.matrix.getH(h), y)
    xhat = np.matmul(np.linalg.inv(p1), p2)

    v1 = np.matmul(xhat, np.ones([1, CONS_ALPHABET.size]))
    v2 = np.matmul(np.ones([USER, 1]), CONS_ALPHABET)
    idxhat = np.argmin(np.square(np.abs(v1 - v2)), axis=1)

    idx = CONS_ALPHABET[:, idxhat]
    accuracy_zf = np.equal(idx.flatten(), np.transpose(s))

    error_zf = 1 - (np.sum(accuracy_zf) / USER)
    #error_zf = 1 - (np.sum(accuracy_zf) / (USER * CONS_ALPHABET.shape[1]))

    return error_zf


def matched_filter(s, h, y):
    p1 = np.matmul(np.matrix.getH(h), y)

    h_flat = h.reshape(h.shape[0] * h.shape[1], 1)
    _, singular_value, _ = np.linalg.svd(h_flat)
    p2 = singular_value[np.argmax(singular_value)] #norm

    xhat = p1 * (1/p2)

    v1 = np.matmul(xhat, np.ones([1, CONS_ALPHABET.size]))
    v2 = np.matmul(np.ones([USER, 1]), CONS_ALPHABET)
    idxhat = np.argmin(np.square(np.abs(v1 - v2)), axis=1)

    idx = CONS_ALPHABET[:, idxhat]
    accuracy_mf = np.equal(idx.flatten(), np.transpose(s))

    error_mf = 1 - (np.sum(accuracy_mf) / USER)
    #error_mf = 1 - (np.sum(accuracy_mf) / (USER * CONS_ALPHABET.shape[1]))

    return error_mf

def sphere_detector(s,H,y):
    # Initialization
    Radius = np.inf
    PA = np.zeros([USER, 1],dtype=int)
    ST = np.zeros([USER, CONS_ALPHABET.size])

    # Preprocessing
    Q, R = linalg.qr(H, mode='economic')
    y_hat = np.matmul(np.matrix.getH(Q),y)

    # Add root node to stack
    level = USER - 1
    sub1 = y_hat[level]
    sub2 = R[level, level]
    sub3 = sub2 * CONS_ALPHABET.T
    ST[level, :] = (np.square(np.abs(sub1 - sub3))).T
    path_flag = 1

    # Sphere detector begin
    while(level <= USER-1):
        minPED = np.amin(ST[level, :])
        idx = np.argmin(ST[level, :])

        # Proceed only if list is not empty
        if(minPED < np.inf):
            ST[level, idx] = np.inf

            if (path_flag <= 1 ):
                NewPath = idx

            else:
                new_path_t = PA[level + 1: None, 0]
                NewPath = np.hstack((idx,new_path_t))

            path_flag = path_flag + 1

            # Search child
            if(minPED < Radius):
                if(level > 0):

                    PA[level:None,0] = NewPath.reshape(-1)

                    level = level - 1
                    PA_t = PA[level + 1: None, 0]
                    PA_t_inv = PA_t.reshape(PA_t.size,1)

                    R_t = R[level,level+1:None]
                    R_t_shape = R_t.reshape(1,R_t.size)

                    DF_t_2 = CONS_ALPHABET[0,PA_t_inv]
                    DF_t_2_inv = DF_t_2.reshape(DF_t_2.size,1)

                    DF = np.matmul(R_t_shape, DF_t_2_inv)


                    tub1 = y_hat[level]
                    tub2 = R[level, level]
                    tub3 = tub2 * CONS_ALPHABET.T

                    ST[level,:]  = minPED + (np.square(np.abs(tub1 - tub3 - DF))).T

                    #print('debug')
                else:
                    idxhat = NewPath.reshape(NewPath.size,1)
                    idx = CONS_ALPHABET[:, idxhat]
                    Radius = minPED

        else:
            level = level + 1
    print('Done')

    accuracy_sd = np.equal(idx.flatten(), np.transpose(s))

    error_sd = 1 - (np.sum(accuracy_sd) / USER)
    return error_sd



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


snr_db_list = []
for snr in range(10, 38, 3):
    snr_db_list.append(snr)

CONS_ALPHABET = constellation_alphabet('BPSK')   # choose anything from here BPSK,QPSK,16QAM,64QAM
'''CHANGEABLE PARAMETERS ARE ABOVE THE LINE'''

signal_energy_avg = np.mean(np.square(np.abs(CONS_ALPHABET)))

snrdb_list = np.linspace(snrdb_low_test, snrdb_high_test, num_snr)
bers_mmse_in_iter = np.zeros([len(snrdb_list), test_iter * test_batch_size])
bers_zf_in_iter = np.zeros([len(snrdb_list), test_iter * test_batch_size])
bers_mf_in_iter = np.zeros([len(snrdb_list), test_iter * test_batch_size])
bers_sd_in_iter = np.zeros([len(snrdb_list), test_iter * test_batch_size])


'''Testing the model'''
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
         bers_zf_in_iter[j][rerun] = zero_forcing(transmitted_symbol, channel, received_signal)
         bers_mf_in_iter[j][rerun] = matched_filter(transmitted_symbol, channel, received_signal)
         bers_sd_in_iter[j][rerun] = sphere_detector(transmitted_symbol, channel, received_signal)

    bers[0][j] = np.mean(tmp_bers, 1)
    times[0][j] = np.mean(tmp_times[0]) / test_batch_size


bers_mmse = np.mean(bers_mmse_in_iter, axis=1)
bers_zf = np.mean(bers_zf_in_iter, axis=1)
bers_mf = np.mean(bers_mf_in_iter, axis=1)
bers_sd = np.mean(bers_sd_in_iter, axis=1)


bits_per_symbol = math.log2(CONS_ALPHABET.shape[1])

bers_mmse = np.divide(bers_mmse, bits_per_symbol)
bers_zf = np.divide(bers_zf, bits_per_symbol)
bers_mf = np.divide(bers_mf, bits_per_symbol)
bers_sd = np.divide(bers_sd, bits_per_symbol)

print('SNR list')
print(snrdb_list)

print('mmse detector error rate', bers_mmse)
print('zero forcing detector error rate', bers_zf)
print('matched filter error rate', bers_mf)
print('sphere detector error rate', bers_sd)
print('DetNet erro rate', bers)
print('times')
print(times)
print(bers[0])

plt.figure('Bit Error Rate')
plt.subplot(111)
plt.semilogy(snrdb_list, bers[0], color='green', marker='^', linestyle='-', linewidth=1, markersize=6, label='DetNet')
plt.semilogy(snrdb_list, bers_mmse, color='black', marker='*', linestyle='-', linewidth=1, markersize=6, label='MMSE')
plt.semilogy(snrdb_list, bers_zf, color='blue', marker='d', linestyle='-', linewidth=1, markersize=5, label='ZF')
plt.semilogy(snrdb_list, bers_mf, color='red', marker='x', linestyle='-', linewidth=1, markersize=6, label='MF')
plt.semilogy(snrdb_list, bers_sd, color='pink', marker='o', linestyle='-', linewidth=1, markersize=6, label='SD')

plt.title('BER vs SNR')
plt.xlabel('average SNR(dB) per receive antenna')
plt.xlim(8, 13)
plt.xscale('linear')
plt.ylabel('BER')
plt.ylim(0.0001, 1)
plt.grid(b=True, which='major', color='#666666', linestyle='--')
plt.legend(title='Detectors:')
plt.show()