# !/usr/bin/env python

"""
Fully Connected network has been updated here
Decision is taken based on least square error
2 network is cascaded
first networks predict the symbol and next network is fed by eliminating the effect of already predicted symbols impact
"""

import tensorflow as tf
import numpy as np
import time as tm
import matplotlib.pyplot as plt

# parameters
user = 20  # 20  # 8  # 20
antenna = 30  # 16  # 30

num_snr = 6
low_snr_db_train = 7.0
high_snr_db_train = 12.0  # 14.0
low_snr_db_test = 7.0  # 8.0
high_snr_db_test = 12.0  # 13.0

low_snr_train = 10.0 ** (low_snr_db_train/10.0)
high_snr_train = 10.0 ** (high_snr_db_train/10.0)
low_snr_test = 10.0 ** (low_snr_db_test/10.0)
high_snr_test = 10.0 ** (high_snr_db_test/10.0)

batch_size = 1000  # 1000
train_iter = 8000  # 1000000
test_iter = 1000
fc_size = 200  # user * user * user # 200
num_of_hidden_layers = 5  # user
startingLearningRate = .0003  # 0.0003
decay_factor = 0.97  # 0.97
decay_step_size = 1000

bers = np.zeros((1, num_snr))
bers_2 = np.zeros((1, num_snr))

H = np.genfromtxt('Top06_30_20.csv', dtype=None, delimiter=',')


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.05)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def constellation_alphabet(mod):
    if mod == 'BPSK':
        return np.array([[-1, 1]], np.float)
    elif mod == 'QPSK':
        return np.array([[-1, 1]], np.float)
    elif mod == '16QAM':
        return np.array([[-3, -1, 1, 3]], np.float)
    elif mod == '64QAM':
        return np.array([[-7, -5, -3, -1, 1, 3, 5, 7]], np.float)


def generate_data_train(B, K, N, snr_low, snr_high, H_org):
    # H_org = np.random.randn(B, N, K)
    # W_ = np.zeros([B, K, K])
    rand_symbol_ind = (np.random.randint(low=0, high=CONS_ALPHABET.shape[1], size=(B*K, 1))).flatten()
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
    H_ = np.random.randn(B, N, K)
    # W_=np.zeros([B,K,K])
    x_ = np.sign(np.random.rand(B, K)-0.5)
    y_ = np.zeros([B,N])
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


def generate_intermediate_input(transmitted_symbol, estimated_symbol):
    trasmitted_symbol_stacked = transmitted_symbol
    estimated_symbol_stacked = estimated_symbol

    v1 = np.matmul(estimated_symbol_stacked, np.ones([1, CONS_ALPHABET.size]))
    v2 = np.matmul(np.ones([user * batch_size, 1]), CONS_ALPHABET)

    idxhat = np.argmin(np.square(np.abs(v1 - v2)), axis=1)

    idx = CONS_ALPHABET[:, idxhat]
    accuracy_zf = np.equal(idx.flatten(), np.transpose(trasmitted_symbol_stacked))

    detection_in_one_and_zero = np.multiply(np.array(accuracy_zf), 1)
    detection_in_one_and_zero_inverted = 1 - detection_in_one_and_zero

    falsely_detected_symbol = np.multiply(np.transpose(detection_in_one_and_zero_inverted), estimated_symbol_stacked)

    falsely_detected_symbol_reshaped = np.reshape(falsely_detected_symbol, [batch_size, user])

    return falsely_detected_symbol_reshaped


def exclude_impact_of_detected_signal(received_sig_e, estimated_symbol_e, batch_e, user_e, antenna_e,
                                      channel_e):
    y_new_e = list()

    for m in range(batch_e):
        a_1 = tf.reshape(estimated_symbol_e[m, :], [user_e, 1])
        d1 = tf.reshape(tf.transpose(received_sig_e[m, :]), [antenna_e, 1])
        d2 = tf.matmul(tf.dtypes.cast(channel_e, tf.float32), a_1)

        d3 = tf.transpose(d1 - d2)

        y_new_e.append(d3)

    hshi = tf.stack(y_new_e)
    hshi = tf.reshape(hshi, [batch_e, antenna_e])

    return hshi


def least_square_error_rate(transmitted_symbol, estimated_symbol):
    trasmitted_symbol_stacked = transmitted_symbol
    estimated_symbol_stacked = estimated_symbol

    v1 = np.matmul(estimated_symbol_stacked, np.ones([1, CONS_ALPHABET.size]))
    v2 = np.matmul(np.ones([user * batch_size, 1]), CONS_ALPHABET)

    idxhat = np.argmin(np.square(np.abs(v1 - v2)), axis=1)

    idx = CONS_ALPHABET[:, idxhat]
    accuracy_zf = np.equal(idx.flatten(), np.transpose(trasmitted_symbol_stacked))
    total_detected_symbol = np.sum(accuracy_zf)
    error_ls = 1 - (total_detected_symbol / (user * batch_size))

    return error_ls, total_detected_symbol

'''..................................................................................................................'''
sess = tf.InteractiveSession()

CONS_ALPHABET = constellation_alphabet('BPSK')
length_one_hot_vector = CONS_ALPHABET.shape[1]

received_sig = tf.placeholder(tf.float32, shape=[None, user], name='received_sig')
received_sig_y = tf.placeholder(tf.float32, shape=[None, antenna], name='input')
transmitted_sig = tf.placeholder(tf.float32, shape=[None, user], name='org_siganl')
intermediate_sig = tf.placeholder(tf.float32, shape=[None, user], name='intermediate_sig')
net_2_input_sig = tf.placeholder(tf.float32, shape=[None, antenna], name='net_2_input_sig')
batchSize = tf.placeholder(tf.int32)
batch_x_one_hot = tf.placeholder(tf.float32, shape=[None, user, length_one_hot_vector], name='one_hot_org_siganl')

# The network
h_fc = []
W_fc_input = weight_variable([user, fc_size])
b_fc_input = bias_variable([fc_size])
h_fc.append(tf.nn.relu(tf.matmul(received_sig, W_fc_input) + b_fc_input))

for i in range(num_of_hidden_layers):
    h_fc.append(activation_fn(h_fc[i - 1], fc_size, fc_size, 'relu' + str(i)))

W_fc_final = weight_variable([fc_size, user])
b_final = bias_variable([user])
h_final = tf.matmul(h_fc[i], W_fc_final) + b_final
'''................'''

'''................'''
y_new = exclude_impact_of_detected_signal(received_sig_y, h_final, batch_size, user, antenna, H)
'''................'''

'''...............'''
h_fc_2 = []
W_fc_input_2 = weight_variable([antenna, fc_size])
b_fc_input_2 = bias_variable([fc_size])
h_fc_2.append(tf.nn.relu(tf.matmul(y_new, W_fc_input_2) + b_fc_input_2))

for j in range(num_of_hidden_layers):
    h_fc_2.append(activation_fn(h_fc_2[j-1], fc_size, fc_size, 'relu' + str(j)))

W_fc_final_2 = weight_variable([fc_size, user])
b_final_2 = bias_variable([user])
h_final_2 = tf.matmul(h_fc_2[j], W_fc_final_2) + b_final_2
'''........................'''

'''........................'''
ssd = tf.reduce_sum(tf.square(transmitted_sig - h_final_2))

global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(startingLearningRate, global_step, decay_step_size, decay_factor,
                                           staircase=True)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(ssd)
'''........................'''

'''........................'''
transmitted_sym = tf.reshape(transmitted_sig, tf.stack([user * batchSize]))
transmitted_sym_reshaped = tf.reshape(transmitted_sig, [user * batch_size, 1])
output_of_final_network = tf.reshape(h_final_2, [user * batch_size, 1])

final_out = tf.reshape(h_final_2, tf.stack([user * batchSize]))
rounded_output = tf.sign(final_out)
equality_check = tf.equal(rounded_output, transmitted_sym)
total_detected_symbols = tf.reduce_sum(tf.cast(equality_check, tf.int32))

loss = ssd

sess.run(tf.global_variables_initializer())
'''........................'''

'''........................'''
"""
training phase of the network
"""
for i in range(train_iter):
    batch_Y, batch_H, batch_HY, batch_HH, batch_X, SNR1, one_hot = generate_data_train(batch_size, user, antenna, low_snr_train, high_snr_train, H)
    train_step.run(feed_dict={received_sig_y: batch_Y, received_sig: batch_HY, transmitted_sig: batch_X, batchSize: batch_size})

    if i % 100 == 0:
        correct_bits = total_detected_symbols.eval(feed_dict={received_sig_y: batch_Y, received_sig: batch_HY, transmitted_sig: batch_X, batchSize: batch_size})
        training_loss = loss.eval(feed_dict={received_sig_y: batch_Y, received_sig: batch_HY, transmitted_sig: batch_X, batchSize: batch_size})
        val_3, final_3 = sess.run([transmitted_sym_reshaped, output_of_final_network], feed_dict={received_sig_y: batch_Y, received_sig: batch_HY, transmitted_sig: batch_X, batchSize: batch_size})
        current_training_error, correct_detection = least_square_error_rate(val_3, final_3)
        print("iteration: %d, training loss: %g, number of correct bits from previous method: %d" % (i, training_loss, correct_bits))
        print("current_training_error: %g" % current_training_error, "|| " "currently_detected_symbol: %d" % correct_detection)
        if correct_bits > 19875:
            break

"""
Testing phase of the network
"""
tmp_bers = np.zeros((1, test_iter))
tmp_bers_2 = np.zeros((1, test_iter))
tmp_times = np.zeros((1, test_iter))
times = np.zeros((1, 1))
testHitCount = 0

snr_list_db = np.linspace(low_snr_db_test, high_snr_db_test, num_snr)
snr_list = 10.0 ** (snr_list_db / 10.0)

for i_snr in range(num_snr):
    Cur_SNR = snr_list[i_snr]
    print('Currrent SNR: %g' % Cur_SNR)

    for i in range(test_iter):
        batch_Y, batch_H, batch_HY, batch_HH, batch_X, SNR1, one_hot = generate_data_train(batch_size, user, antenna, Cur_SNR, Cur_SNR, H)
        # tic = tm.time()
        tmp_bers[0][i] = total_detected_symbols.eval(feed_dict={received_sig_y: batch_Y, received_sig: batch_HY, transmitted_sig: batch_X, batchSize: batch_size})
        val_3, final_3 = sess.run([transmitted_sym_reshaped, output_of_final_network], feed_dict={received_sig_y: batch_Y, received_sig: batch_HY, transmitted_sig: batch_X, batchSize: batch_size})
        tmp_bers_2[0][i], detected_symbol = least_square_error_rate(val_3, final_3)
        # toc = tm.time()
        # tmp_times[0][i] = toc - tic
        if i % 100 == 0:
            print("Correctly detected symbol using previous method: %g" % tmp_bers[0][i])
            print("Correctly detected symbol using least square: %d" % detected_symbol)
            print("Current bit error rate: %g" % tmp_bers_2[0][i])

    bers[0][i_snr] = np.mean(tmp_bers[0])
    bers_2[0][i_snr] = np.mean(tmp_bers_2[0])

# times[0][0] = np.mean(tmp_times[0]) / batch_size
# print('Average time to detect a single K bit signal is:', times)

print('snr_list_db', snr_list_db)

bers = 1 - bers / (user * batch_size)
print('Bit error rates from previous method:', bers)

print('Bit error rates calculated using least square error:', bers_2)

fig1 = plt.figure('BPSK, TxR:20x30, Iteration:8000, SNR:7-12, Batch size:1000')
ax1 = fig1.add_subplot(111)
ax1.plot(snr_list_db.reshape(-1), bers.reshape(-1), color='black', marker='*', linestyle='-', linewidth=1, markersize=6, label='FC')
ax1.plot(snr_list_db.reshape(-1), bers_2.reshape(-1), color='blue', marker='d', linestyle='-', linewidth=1, markersize=6, label='FC_LS')
ax1.set_title('BER vs SNR')
ax1.set_xlabel('SNR(dB)')
ax1.set_ylabel('BER')
ax1.set_yscale('log')
ax1.set_ylim(0.0001, 1)
ax1.set_xlim(7, 12)
plt.grid(b=True, which='major', color='#666666', linestyle='--')
plt.legend(title='Detector')
plt.show()