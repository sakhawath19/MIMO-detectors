
'''................................................................................'''
import tensorflow as tf
import numpy as np
import pandas as pd
import time as tm
#from tensorflow.contrib.data import Iterator
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor
'''..................................................................................'''
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

n_classes = 2 * user

H = np.genfromtxt('Top06_30_20.csv', dtype=None, delimiter=',')

'''..................................................................................'''

sess = tf.InteractiveSession()

def constellation_alphabet(mod):
    if mod == 'BPSK':
        return np.array([[-1, 1]], np.float)
    elif mod == 'QPSK':
        return np.array([[-1, 1]], np.float)
    elif mod == '16QAM':
        return np.array([[-3, -1, 1, 3]], np.float)
    elif mod == '64QAM':
        return np.array([[-7, -5, -3, -1, 1, 3, 5, 7]], np.float)

CONS_ALPHABET = constellation_alphabet('QPSK')
length_one_hot_vector = CONS_ALPHABET.shape[1]


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

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.05)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


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



received_sig = tf.placeholder(tf.float32, shape=[None, 1, antenna, 1], name='input')
#received_sig = tf.placeholder(tf.float32, shape=[None, user], name='input')
#received_sig = tf.placeholder(tf.float32, shape=[None, antenna], name='input')
transmitted_sig = tf.placeholder(tf.float32, shape=[None, user], name='org_siganl')
batchSize = tf.placeholder(tf.int32)
batch_x_one_hot = tf.placeholder(tf.float32, shape=[None, user, length_one_hot_vector], name='one_hot_org_siganl')
'''...............................................................................'''

#x = tf.placeholder('float32', [None, 1, samples_num_in_segments, 1])
y = tf.placeholder('int32')

keep_rate = 0.75
keep_prob = tf.placeholder(tf.float32)

def error_rate(transmitted_symbol, estimated_symbol):
    trasmitted_symbol_stacked = transmitted_symbol
    #trasmitted_symbol_stacked = tf.reshape(transmitted_symbol, tf.stack([K * batchSize]))
    estimated_symbol_stacked = estimated_symbol
    #estimated_symbol_stacked = tf.reshape(estimated_symbol, tf.stack([K * batchSize]))
    #t = np.ones([1, CONS_ALPHABET.size])
    #p1 = np.matmul(estimated_symbol_stacked, t)
    v1 = np.matmul(estimated_symbol_stacked, np.ones([1, CONS_ALPHABET.size]))
    v2 = np.matmul(np.ones([user * batch_size, 1]), CONS_ALPHABET)

    idxhat = np.argmin(np.square(np.abs(v1 - v2)), axis=1)

    idx = CONS_ALPHABET[:, idxhat]
    accuracy_zf = np.equal(idx.flatten(), np.transpose(trasmitted_symbol_stacked))

    error_zf = 1 - (np.sum(accuracy_zf) / (user * batch_size))

    return error_zf


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')


def maxpool2d(x):
    #                        size of window         movement of window
    return tf.nn.max_pool(x, ksize=[1, 1, 5, 1], strides=[1, 1, 1, 1], padding='SAME')
    #return tf.nn.max_pool(x, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='VALID')


def convolutional_neural_network(x):

    weights = {'W_conv1': tf.Variable(tf.random_normal([1, 5, 1, 8])), # 8 filters
               'W_conv2': tf.Variable(tf.random_normal([1, 5, 8, 16])),
               'W_conv3': tf.Variable(tf.random_normal([1, 5, 16, 32])),
               'W_conv4': tf.Variable(tf.random_normal([1, 5, 32, 32])),


               'W_fc': tf.Variable(tf.random_normal([14 * 32, 200])),  #([7* 32 * 41, 256])) # no of wavelet components is 7
               #'W_fc': tf.Variable(tf.random_normal([7 * 32 * 41, 256])),  #([7* 32 * 41, 256])) # no of wavelet components is 7
               'out': tf.Variable(tf.random_normal([200, 200]))}
               #'out': tf.Variable(tf.random_normal([200, n_classes]))}

    biases = {'b_conv1': tf.Variable(tf.random_normal([8])),
              'b_conv2': tf.Variable(tf.random_normal([16])),
              'b_conv3': tf.Variable(tf.random_normal([32])),
              'b_conv4': tf.Variable(tf.random_normal([32])),

              'b_fc': tf.Variable(tf.random_normal([200])),
              'out': tf.Variable(tf.random_normal([200]))}  # n times 32 will be the b_fc, input dimension
              #'out': tf.Variable(tf.random_normal([n_classes]))}  # n times 32 will be the b_fc, input dimension

    # NETWORK 1
    conv_layer_1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
    conv_max_pool_layer_1 = maxpool2d(conv_layer_1)

    conv_layer_2 = tf.nn.relu(conv2d(conv_max_pool_layer_1, weights['W_conv2']) + biases['b_conv2'])
    conv_max_pool_layer_2 = maxpool2d(conv_layer_2)

    conv_layer_3 = tf.nn.relu(conv2d(conv_max_pool_layer_2, weights['W_conv3']) + biases['b_conv3'])
    conv_max_pool_layer_3 = maxpool2d(conv_layer_3)

    conv_layer_4 = tf.nn.relu(conv2d(conv_max_pool_layer_3, weights['W_conv4']) + biases['b_conv4'])
    net_1 = maxpool2d(conv_layer_4)


    # Fully connected layer
    fc = tf.reshape(net_1, [-1, 14* 32])  # change 7 as a variable of no of components
    fc = tf.tanh(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])
    fc = tf.nn.dropout(fc, keep_rate)

    output = tf.matmul(fc, weights['out']) + biases['out']

    return output

conv_out = convolutional_neural_network(received_sig)

W_fc_second_last_final = weight_variable([fc_size, length_one_hot_vector * user])
b_second_last_final = bias_variable([length_one_hot_vector * user])
h_second_last_final = tf.matmul(conv_out, W_fc_second_last_final) + b_second_last_final

h_final = tf.reshape(h_second_last_final, [batchSize, user, length_one_hot_vector])

final_output = tf.nn.softmax(h_final, axis=2)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=final_output, labels=batch_x_one_hot))
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(startingLearningRate, global_step, decay_step_size, decay_factor,
                                           staircase=True)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

com_received_transmitted_sig = tf.equal(tf.argmax(final_output, 2), tf.argmax(batch_x_one_hot, 2))
accuracy = tf.reduce_sum(tf.cast(com_received_transmitted_sig, 'float')) / (user * batch_size)


val = tf.reshape(transmitted_sig, tf.stack([user * batchSize]))
val_2 = tf.reshape(transmitted_sig, [user * batch_size, 1])
'''
final_2 = tf.reshape(h_final, [user * batch_size, 1])
final = tf.reshape(h_final, tf.stack([user * batchSize]))
rounded = tf.sign(final)
eq = tf.equal(rounded, val)
eq2 = tf.reduce_sum(tf.cast(eq, tf.int32))
'''
cost_accumulated = []
accuracy_accumulated = []
sess.run(tf.global_variables_initializer())

# train the network

for i in range(train_iter):
    batch_Y, batch_H, batch_HY, batch_HH, batch_X, SNR1, one_hot = generate_data_train(batch_size, user, antenna, low_snr_train, high_snr_train, H)
    _, c = sess.run([train_step, loss], feed_dict={received_sig: batch_Y.reshape([-1, 1, antenna, 1]), transmitted_sig: batch_X, batchSize: batch_size, batch_x_one_hot: one_hot})
    #_, c = sess.run([train_step, loss], feed_dict={received_sig: batch_Y, transmitted_sig: batch_X, batchSize: batch_size, batch_x_one_hot: one_hot})
    ac = sess.run([accuracy], feed_dict={received_sig: batch_Y.reshape([-1, 1, antenna, 1]), transmitted_sig: batch_X, batchSize: batch_size, batch_x_one_hot: one_hot})
    '''
    val_3, final_3 = sess.run([val_2, conv_out], feed_dict={received_sig: batch_Y.reshape([-1, 1, antenna, 1]), transmitted_sig: batch_X, batchSize: batch_size})
    eq3 = error_rate(val_3, final_3)
    print('Error_', eq3)
    '''
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
        test_ac = sess.run([accuracy], feed_dict={received_sig: batch_Y.reshape([-1, 1, antenna, 1]), transmitted_sig: batch_X, batchSize: batch_size, batch_x_one_hot: one_hot})
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

'''
plt.figure('Bit Error Rate')
plt.subplot(111)
plt.semilogy(snr_db_list, bers_mmse, color='black', marker='*', linestyle='-', linewidth=1, markersize=6, label='MMSE')
plt.semilogy(snr_db_list, bers_zf, color='blue', marker='d', linestyle='-', linewidth=1, markersize=5, label='ZF')
plt.semilogy(snr_db_list, bers_mf, color='red', marker='x', linestyle='-', linewidth=1, markersize=6, label='MF')
plt.semilogy(snr_db_list, bers_sd, color='pink', marker='o', linestyle='-', linewidth=1, markersize=6, label='SD')
plt.title('BER vs SNR')
plt.xlabel('average SNR(dB) per receive antenna')
plt.xlim(-2, 16)
plt.xscale('linear')
plt.ylabel('BER')
plt.ylim(0.0001, 1)
plt.grid(b=True, which='major', color='#666666', linestyle='--')
plt.legend(title='Detectors:')
plt.show()
'''