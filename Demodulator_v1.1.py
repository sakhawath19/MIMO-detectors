import matplotlib.pyplot as plt
import numpy as np
from numpy import array
import tensorflow as tf
import time as tm


p = array([2.15499461471988e-08,	0.0115103405335189,	0.0166323274340217,	-6.66667155493756e-08,	-0.0307565489229459,
     -0.0402860617823189,	1.50849536892277e-07,	0.0664549573975079,	0.0846757350922837,	-3.14159453806887e-07,
     -0.140100267670711,	-0.186073367717685,	7.54247908392509e-07,	0.402862042985211,	0.821638297314620,
     0.999999999998910,	0.821636686617407,	0.402859856417632,	-7.54246558238786e-07,	-0.186073636690607,
     -0.140099680893354,	3.14159076575752e-07,	0.0846758821713932,	0.0664546870342451,	-1.50849356327204e-07,
     -0.0402861281576377,	-0.0307564183899480,	6.66666177927293e-08,	0.0166323483539678,	0.0115102851636668,
     -2.15498957565380e-08])


num_values = 1
batch_size =1000
sample_size = 4 * num_values
N = sample_size
K = num_values
B = batch_size
#low_snr = 0.7
#high_snr = .14
batch_data = np.zeros([B, N])
batch_symbols = np.zeros([B, K])


def generate_data(batch_size,num_values,low_snr,high_snr):
    samples = []
    #symbols = []
    for b in range(batch_size):

        symbols = np.sign(np.random.rand(1, num_values) - 0.5)

        for i in range(num_values):

            pulse_shape = symbols[0,i] * p
            noise = np.random.uniform(low=low_snr, high=high_snr,size=len(p))
            noisy_pulse = pulse_shape + noise

            samples.extend(noisy_pulse[13:17])

        batch_data[b,:] = samples
        batch_symbols[b,:] = symbols
        samples = []
    return batch_data,batch_symbols




def affine_layer(x, input_size, output_size, Layer_num):
    W = tf.Variable(tf.random_normal([input_size, output_size], stddev=0.01))
    w = tf.Variable(tf.random_normal([1, output_size], stddev=0.01))
    y = tf.matmul(x, W) + w
    return y


def relu_layer(x, input_size, output_size, Layer_num):
    y = tf.nn.relu(affine_layer(x, input_size, output_size, Layer_num))
    return y

#K = 20
#N = 30
#B = 1000
train_iter = 10000  # 1000000
test_iter = 1000
low_snr_db_train = 7.0
high_snr_db_train = 14.0
low_snr_db_test = 8.0
high_snr_db_test = 13.0
num_snr = 6
fc_size = 20
num_of_hidden_layers = 2
startingLearningRate = 0.0003
decay_factor = .8#0.97
decay_step_size = 10000#1000
#H = np.genfromtxt('Top06_30_20.csv', dtype=None, delimiter=',')
# end of parameters

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

NNinput = tf.placeholder(tf.float32, shape=[None,  N], name='input')
org_siganl = tf.placeholder(tf.float32, shape=[None, K], name='org_siganl')
batchSize = tf.placeholder(tf.int32)

# The network
h_fc = []
W_fc_input = weight_variable([N, fc_size])
b_fc_input = bias_variable([fc_size])

h_fc.append(tf.nn.relu(tf.matmul(NNinput, W_fc_input) + b_fc_input))

for i in range(num_of_hidden_layers):
    h_fc.append(relu_layer(h_fc[i - 1], fc_size, fc_size, 'relu' + str(i)))

W_fc_final = weight_variable([fc_size, K])
b_final = bias_variable([K])
#h_final = tf.matmul(h_fc[i], W_fc_final) + b_final
h_final = tf.matmul(h_fc[-1], W_fc_final) + b_final

ssd = tf.reduce_sum(tf.square(org_siganl - h_final))

global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(startingLearningRate, global_step, decay_step_size, decay_factor,
                                           staircase=True)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(ssd)

val = tf.reshape(org_siganl, tf.stack([K * batchSize]))
final = tf.reshape(h_final, tf.stack([K * batchSize]))
rounded = tf.sign(final)
eq = tf.equal(rounded, val)
eq2 = tf.reduce_sum(tf.cast(eq, tf.int32))

accuracy = ssd

sess.run(tf.global_variables_initializer())
"""
training phase og the network
"""
bits_train = []
loss_train = []
for i in range(train_iter):
    batch_Y, batch_X = generate_data(B,K, low_snr_train, high_snr_train)
    if i % 100 == 0:
        correct_bits = eq2.eval(feed_dict={NNinput: batch_Y, org_siganl: batch_X, batchSize: B})
        bits_train.extend([correct_bits])
        training_loss = accuracy.eval(feed_dict={NNinput: batch_Y, org_siganl: batch_X, batchSize: B})
        loss_train.extend([training_loss])
        print("step %d, Training loss %g, number of correct bits %d" % (i, training_loss, correct_bits))
    train_step.run(feed_dict={NNinput: batch_Y, org_siganl: batch_X, batchSize: B})

plt.figure(1)
plt.title('Training Loss and Bit Accuracy')
plt.subplot(2,1,1)
plt.plot(loss_train)
plt.ylabel('Training loss')

plt.subplot(2,1,2)
plt.plot(bits_train)
plt.xlabel('Iteration x 100')
plt.ylabel('Correct bits')
plt.show()
