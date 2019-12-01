import tensorflow as tf
import numpy as np
import time as tm

#### Parameters

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


sess = tf.InteractiveSession()

NNinput = tf.placeholder(tf.float32, shape=[None, size_of_received_vector_y], name='input')
org_siganl = tf.placeholder(tf.float32, shape=[None, size_of_x], name='org_siganl')
batchSize = tf.placeholder(tf.int32)

### Data generation


### Network


### Run session for training


### Run session for testing