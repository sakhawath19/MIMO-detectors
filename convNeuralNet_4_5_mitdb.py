
'''................................................................................'''
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.contrib.data import Iterator
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor
'''..................................................................................'''
'''..................................................................................'''

row = -1
art_data = np.ones([1000, 720 * 7 + 1], np.float32)
for i in range(1, 101):
    for j in range(1, 11):
        row = row + 1
        art_data[row,:] = art_data[row,:] * i

data = np.array(art_data)
segments_num = data.shape[0]

#print(segments_num)

split_ind = int(segments_num * 0.8)

sample_train = data[:split_ind, :]
sample_test = data[split_ind:, :]
'''..................................................................................'''
'''
#input_data = pd.read_csv('input/component_s39_mitdb_test.csv', header=None)
input_data = pd.read_csv('input/component_s39_mitdb.csv', header=None)
data = np.array(input_data)
segments_num = data.shape[0]

print(segments_num)

split_ind = int(segments_num * 0.80)

sample_train = data[:split_ind, :]
sample_test = data[split_ind:, :]
'''
'''...............................................................................'''

qrs_data = sample_train[:,1:]
qrs_data = qrs_data.astype(np.float32)
samples_num_in_segments = qrs_data.shape[1]

patients_labels = sample_train[:, 0] - 1
# print(patients_labels)

n_classes = np.unique(patients_labels).shape[0]

patients_labels = convert_to_tensor(patients_labels, dtype=dtypes.int32)

one_hot = tf.one_hot(patients_labels, n_classes,dtype=dtypes.int32)
# print(one_hot)

qrs_data_labels = tf.data.Dataset.from_tensor_slices((qrs_data,one_hot))
# print(qrs_data_labels)

train_batch_size = 100
qrs_data_labels = qrs_data_labels.batch(train_batch_size)

iterator_train = Iterator.from_structure(qrs_data_labels.output_types, qrs_data_labels.output_shapes)

next_train_batch = iterator_train.get_next()

qrs_data_labels_op = iterator_train.make_initializer(qrs_data_labels)
# print(next_train_batch)

'''...............................................................................'''

test_qrs_data = sample_test[:,1:]
test_qrs_data = test_qrs_data.astype(np.float32)

test_patients_labels = sample_test[:,0] - 1
# print(test_patients_labels)

test_patients_labels = convert_to_tensor(test_patients_labels, dtype=dtypes.int32)

test_one_hot = tf.one_hot(test_patients_labels, n_classes,dtype=dtypes.int32)
# print(test_one_hot)

test_qrs_data_labels = tf.data.Dataset.from_tensor_slices((test_qrs_data,test_one_hot))
# print(test_qrs_data_labels)

test_batch_size = sample_test.shape[0]
test_qrs_data_labels = test_qrs_data_labels.batch(test_batch_size)

iterator_test = Iterator.from_structure(test_qrs_data_labels.output_types,test_qrs_data_labels.output_shapes)
next_test_batch = iterator_test.get_next()

test_qrs_data_labels_op = iterator_test.make_initializer(test_qrs_data_labels)
# print(next_test_batch)

'''...............................................................................'''

x = tf.placeholder('float32', [None, 1, samples_num_in_segments, 1])
y = tf.placeholder('int32')

keep_rate = 0.75
keep_prob = tf.placeholder(tf.float32)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')


def maxpool2d(x):
    #                        size of window         movement of window
    return tf.nn.max_pool(x, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='VALID')


def convolutional_neural_network(x):
    # c1, c2, c3, c4, c5, c6, c7  = tf.split(x,[720,720,720,720,720,720,720],2)
    c1, c2, c3, c4, c5, c6, c7  = tf.split(x,7,2)

    weights = {'W_conv1': tf.Variable(tf.random_normal([1, 5, 1, 8])), # 8 filters
               'W_conv2': tf.Variable(tf.random_normal([1, 5, 8, 16])),
               'W_conv3': tf.Variable(tf.random_normal([1, 5, 16, 32])),
               'W_conv4': tf.Variable(tf.random_normal([1, 5, 32, 32])),


               'W_fc': tf.Variable(tf.random_normal([7 * 32 * 41, 256])),  #([7* 32 * 41, 256])) # no of wavelet components is 7
               'out': tf.Variable(tf.random_normal([256, n_classes]))}

    biases = {'b_conv1': tf.Variable(tf.random_normal([8])),
              'b_conv2': tf.Variable(tf.random_normal([16])),
              'b_conv3': tf.Variable(tf.random_normal([32])),
              'b_conv4': tf.Variable(tf.random_normal([32])),

              'b_fc': tf.Variable(tf.random_normal([256])),
              'out': tf.Variable(tf.random_normal([n_classes]))}  # n times 32 will be the b_fc, input dimension

    # NETWORK 1
    conv_layer_1 = tf.tanh(conv2d(c1, weights['W_conv1']) + biases['b_conv1'])
    conv_max_pool_layer_1 = maxpool2d(conv_layer_1)

    conv_layer_2 = tf.tanh(conv2d(conv_max_pool_layer_1, weights['W_conv2']) + biases['b_conv2'])
    conv_max_pool_layer_2 = maxpool2d(conv_layer_2)

    conv_layer_3 = tf.tanh(conv2d(conv_max_pool_layer_2, weights['W_conv3']) + biases['b_conv3'])
    conv_max_pool_layer_3 = maxpool2d(conv_layer_3)

    conv_layer_4 = tf.tanh(conv2d(conv_max_pool_layer_3, weights['W_conv4']) + biases['b_conv4'])
    net_1 = maxpool2d(conv_layer_4)

    # NETWORK 2
    conv_layer_1 = tf.tanh(conv2d(c2, weights['W_conv1']) + biases['b_conv1'])
    conv_max_pool_layer_1 = maxpool2d(conv_layer_1)

    conv_layer_2 = tf.tanh(conv2d(conv_max_pool_layer_1, weights['W_conv2']) + biases['b_conv2'])
    conv_max_pool_layer_2 = maxpool2d(conv_layer_2)

    conv_layer_3 = tf.tanh(conv2d(conv_max_pool_layer_2, weights['W_conv3']) + biases['b_conv3'])
    conv_max_pool_layer_3 = maxpool2d(conv_layer_3)

    conv_layer_4 = tf.tanh(conv2d(conv_max_pool_layer_3, weights['W_conv4']) + biases['b_conv4'])
    net_2 = maxpool2d(conv_layer_4)

    # NETWORK 3
    conv_layer_1 = tf.tanh(conv2d(c3, weights['W_conv1']) + biases['b_conv1'])
    conv_max_pool_layer_1 = maxpool2d(conv_layer_1)

    conv_layer_2 = tf.tanh(conv2d(conv_max_pool_layer_1, weights['W_conv2']) + biases['b_conv2'])
    conv_max_pool_layer_2 = maxpool2d(conv_layer_2)

    conv_layer_3 = tf.tanh(conv2d(conv_max_pool_layer_2, weights['W_conv3']) + biases['b_conv3'])
    conv_max_pool_layer_3 = maxpool2d(conv_layer_3)

    conv_layer_4 = tf.tanh(conv2d(conv_max_pool_layer_3, weights['W_conv4']) + biases['b_conv4'])
    net_3 = maxpool2d(conv_layer_4)

    # NETWORK 4
    conv_layer_1 = tf.tanh(conv2d(c4, weights['W_conv1']) + biases['b_conv1'])
    conv_max_pool_layer_1 = maxpool2d(conv_layer_1)

    conv_layer_2 = tf.tanh(conv2d(conv_max_pool_layer_1, weights['W_conv2']) + biases['b_conv2'])
    conv_max_pool_layer_2 = maxpool2d(conv_layer_2)

    conv_layer_3 = tf.tanh(conv2d(conv_max_pool_layer_2, weights['W_conv3']) + biases['b_conv3'])
    conv_max_pool_layer_3 = maxpool2d(conv_layer_3)

    conv_layer_4 = tf.tanh(conv2d(conv_max_pool_layer_3, weights['W_conv4']) + biases['b_conv4'])
    net_4 = maxpool2d(conv_layer_4)

    # NETWORK 5
    conv_layer_1 = tf.tanh(conv2d(c5, weights['W_conv1']) + biases['b_conv1'])
    conv_max_pool_layer_1 = maxpool2d(conv_layer_1)

    conv_layer_2 = tf.tanh(conv2d(conv_max_pool_layer_1, weights['W_conv2']) + biases['b_conv2'])
    conv_max_pool_layer_2 = maxpool2d(conv_layer_2)

    conv_layer_3 = tf.tanh(conv2d(conv_max_pool_layer_2, weights['W_conv3']) + biases['b_conv3'])
    conv_max_pool_layer_3 = maxpool2d(conv_layer_3)

    conv_layer_4 = tf.tanh(conv2d(conv_max_pool_layer_3, weights['W_conv4']) + biases['b_conv4'])
    net_5 = maxpool2d(conv_layer_4)

    # NETWORK 6
    conv_layer_1 = tf.tanh(conv2d(c6, weights['W_conv1']) + biases['b_conv1'])
    conv_max_pool_layer_1 = maxpool2d(conv_layer_1)

    conv_layer_2 = tf.tanh(conv2d(conv_max_pool_layer_1, weights['W_conv2']) + biases['b_conv2'])
    conv_max_pool_layer_2 = maxpool2d(conv_layer_2)

    conv_layer_3 = tf.tanh(conv2d(conv_max_pool_layer_2, weights['W_conv3']) + biases['b_conv3'])
    conv_max_pool_layer_3 = maxpool2d(conv_layer_3)

    conv_layer_4 = tf.tanh(conv2d(conv_max_pool_layer_3, weights['W_conv4']) + biases['b_conv4'])
    net_6 = maxpool2d(conv_layer_4)

    # NETWORK 7
    conv_layer_1 = tf.tanh(conv2d(c7, weights['W_conv1']) + biases['b_conv1'])
    conv_max_pool_layer_1 = maxpool2d(conv_layer_1)

    conv_layer_2 = tf.tanh(conv2d(conv_max_pool_layer_1, weights['W_conv2']) + biases['b_conv2'])
    conv_max_pool_layer_2 = maxpool2d(conv_layer_2)

    conv_layer_3 = tf.tanh(conv2d(conv_max_pool_layer_2, weights['W_conv3']) + biases['b_conv3'])
    conv_max_pool_layer_3 = maxpool2d(conv_layer_3)

    conv_layer_4 = tf.tanh(conv2d(conv_max_pool_layer_3, weights['W_conv4']) + biases['b_conv4'])
    net_7 = maxpool2d(conv_layer_4)

    con_1_2 = tf.concat([net_1,net_2],2)
    con_3 = tf.concat([con_1_2,net_3],2)
    con_4 = tf.concat([con_3, net_4],2)
    con_5 = tf.concat([con_4, net_5],2)
    con_6 = tf.concat([con_5, net_6],2)
    con_7 = tf.concat([con_6, net_7],2)

    # Fully connected layer
    fc = tf.reshape(con_7, [-1, 7* 32 * 41])  # change 7 as a variable of no of components
    fc = tf.tanh(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])
    fc = tf.nn.dropout(fc, keep_rate)

    output = tf.matmul(fc, weights['out']) + biases['out']

    return output


def train_neural_network(x):
    prediction = convolutional_neural_network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    hm_epochs = 5
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(hm_epochs):
            sess.run(qrs_data_labels_op)
            epoch_loss = 0
            while True:
                try:
                    epoch_x, epoch_y = sess.run(next_train_batch)
                    _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x.reshape([-1, 1, samples_num_in_segments, 1]), y: epoch_y})
                    epoch_loss += c

                except tf.errors.OutOfRangeError:
                    print("End of training dataset.")
                    break

            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', float(epoch_loss))
            print(str(epoch_loss))

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        sess.run(test_qrs_data_labels_op)
        test_x, test_y = sess.run(next_test_batch)

        print('Accuracy:', accuracy.eval({x: test_x.reshape([-1, 1, samples_num_in_segments, 1]), y: test_y}))

train_neural_network(x)
