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

def generate_data_train(B, K, N, snr_low, snr_high):
    H_ = np.random.randn(B, N, K)
    W_ = np.zeros([B, K, K])
    #x_ = np.sign(np.random.rand(B, K) - 0.5)

    rand_symbol_ind = (np.random.randint(low=0, high=CONS_ALPHABET.shape[1], size=(B * 2 * USER, 1))).flatten()

    transmitted_symbol = (CONS_ALPHABET[:, rand_symbol_ind])
    x_ = transmitted_symbol.reshape(B,2*USER)

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

def constellation_alphabet_real(mod):
    if mod == 'BPSK':
        return np.array([[-1, 1]], np.int)
    elif mod == 'QPSK':
        return np.array([[-1, 1]], np.int)
    elif mod == '16QAM':
        return np.array([[-1, 1, -3, 3]], np.int)
    elif mod == '64QAM':
        return np.array([[-1, 1, -3, 3, -5, 5, -7, 7]], np.int)


CONS_ALPHABET = constellation_alphabet_real('64QAM')   # choose anything from here BPSK,QPSK,16QAM,64QAM


y_g, H_g, Hy_g, HH_g, x_g, SNR_g = generate_data_train(10, 2*K, 2*N, snr_low, snr_high)

print('Stop')