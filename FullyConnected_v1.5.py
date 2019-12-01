import tensorflow as tf
import numpy as np
import time as tm
import matplotlib.pyplot as plt


'''Variable defined from previous code
K transmitter, N receiver, B batch size
'''
#K = 20
#N = 30
#B = 1000
train_iter = 10000#1000000
test_iter = 1000
low_snr_db_train = 7.0
high_snr_db_train = 14.0
low_snr_db_test = 8.0
high_snr_db_test = 13.0
num_snr = 6
#fc_size = 200 # not used in this code anymore
#num_of_hidden_layers = 4
startingLearningRate = 0.0003
decay_factor = 0.97
decay_step_size = 1000
H=np.genfromtxt('Top06_30_20.csv', dtype=None, delimiter=',')


low_snr_train = 10.0 ** (low_snr_db_train/10.0)
high_snr_train = 10.0 ** (high_snr_db_train/10.0)
low_snr_test = 10.0 ** (low_snr_db_test/10.0)
high_snr_test = 10.0 ** (high_snr_db_test/10.0)
bers = np.zeros((1,num_snr))

'''Variable defined for new code'''
nodes_hidden_layer_1 = 200
nodes_hidden_layer_2 = 200
nodes_hidden_layer_3 = 200
nodes_hidden_layer_4 = 200

transmitter = 20
receiver = 30
batch_size = 1000

'''Data generated'''
def generate_data(B,K,N,snr_low,snr_high,H_org):
    W_ = np.zeros([B, K, K])
    x_ = np.sign(np.random.rand(B, K) - 0.5)
    y_ = np.zeros([B, N])
    w = np.random.randn(B, N)
    Hy_ = x_ * 0
    H_ = np.zeros([B, N, K])
    HH_ = np.zeros([B, K, K])
    SNR_ = np.zeros([B])
    for i in range(B):
        #print (i)

        SNR = np.random.uniform(low=snr_low,high=snr_high)
        H=H_org
        tmp_snr=(H.T.dot(H)).trace()/K
        H_[i,:,:]=H
        y_[i,:]=(H.dot(x_[i,:])+w[i,:]*np.sqrt(tmp_snr)/np.sqrt(SNR))
        Hy_[i,:]=H.T.dot(y_[i,:])
        HH_[i,:,:]=H.T.dot( H_[i,:,:])
        SNR_[i] = SNR
    return y_, H_, Hy_, HH_, x_, SNR


'''
Placeholder defiend for received signal, transmitted signal known as org_signal and batch size for the network feeding 
'''
received_signal = tf.placeholder(tf.float32, shape=[None, receiver], name='input')
org_siganl = tf.placeholder(tf.float32, shape=[None, transmitter], name='org_siganl')
batchSize = tf.placeholder(tf.int32)

'''
Neural network model is defined in 'def neural_network_model' 
4 Hidden layer and 1 output layer
signal is the input signal for the network
All the weights in the hidden layer are normally distributed with standard deviation 0.01
Final layer's weight are truncated normal and biases are fixed value 0.01
Neural network input passes through the network with similar operation such as: output = relu(input*weight+bias)
Neural network returns final layer output only
'''

hidden_layer_1 = {'weights':tf.Variable(tf.truncated_normal([receiver, nodes_hidden_layer_1], stddev=0.05)),
                  'biases':tf.Variable(0.1, [receiver])}

hidden_layer_2 = {'weights': tf.Variable(tf.random_normal([nodes_hidden_layer_1,nodes_hidden_layer_2],stddev=0.01)),
                  'biases': tf.Variable(tf.random_normal([nodes_hidden_layer_2],stddev=0.01))}

hidden_layer_3 = {'weights': tf.Variable(tf.random_normal([nodes_hidden_layer_2,nodes_hidden_layer_3],stddev=0.01)),
                  'biases': tf.Variable(tf.random_normal([nodes_hidden_layer_3],stddev=0.01))}

hidden_layer_4 = {'weights': tf.Variable(tf.random_normal([nodes_hidden_layer_3,nodes_hidden_layer_4],stddev=0.01)),
                  'biases': tf.Variable(tf.random_normal([nodes_hidden_layer_4],stddev=0.01))}

final_layer = {'weights': tf.Variable(tf.truncated_normal([nodes_hidden_layer_4, transmitter],
                                                          stddev=0.05)),'biases': tf.Variable(0.1, [transmitter])}


hidden_layer_1_output = tf.add(tf.matmul(received_signal,hidden_layer_1['weights']),hidden_layer_1['biases'])
hidden_layer_1_output = tf.nn.relu(hidden_layer_1_output)

hidden_layer_2_output = tf.add(tf.matmul(hidden_layer_1_output,hidden_layer_2['weights']),hidden_layer_2['biases'])
hidden_layer_2_output = tf.nn.relu(hidden_layer_2_output)

hidden_layer_3_output = tf.add(tf.matmul(hidden_layer_2_output,hidden_layer_3['weights']),hidden_layer_3['biases'])
hidden_layer_3_output = tf.nn.relu(hidden_layer_3_output)

hidden_layer_4_output = tf.add(tf.matmul(hidden_layer_3_output,hidden_layer_4['weights']),hidden_layer_4['biases'])
hidden_layer_4_output = tf.nn.relu(hidden_layer_4_output)

final_layer_output = tf.add(tf.matmul(hidden_layer_4_output,final_layer['weights']),final_layer['biases'])
estimated_signal = tf.nn.relu(final_layer_output)


'''Network is trained in this block of code'''

error = tf.reduce_sum(tf.square(org_siganl - estimated_signal))

global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(startingLearningRate, global_step, decay_step_size, decay_factor, staircase=True)
train_network = tf.train.AdamOptimizer(learning_rate).minimize(error)

'''Helping to calculate correct bits'''
val = tf.reshape(org_siganl, tf.stack([transmitter * batchSize]))
final = tf.reshape(estimated_signal, tf.stack([transmitter * batchSize]))
rounded = tf.sign(final)
eq = tf.equal(rounded, val)
eq2 = tf.reduce_sum(tf.cast(eq, tf.int32))

accuracy = error

'''Interactive session started'''
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

"""
training phase og the network
"""
bits = []
loss= []
for i in range(train_iter):
    batch_Y, batch_H, batch_HY, batch_HH, batch_X , SNR1= generate_data(batch_size, transmitter, receiver, low_snr_train, high_snr_train, H)
    if i % 100 == 0:
        correct_bits = eq2.eval(feed_dict={received_signal: batch_Y, org_siganl: batch_X, batchSize: batch_size})
        bits.extend([correct_bits])
        train_loss = accuracy.eval(feed_dict={received_signal: batch_Y, org_siganl: batch_X, batchSize: batch_size})
        loss.extend([train_loss])
        print("step %d, loss is %g, number of correct bits %d" % (i, train_loss,correct_bits))
    train_network.run(feed_dict={received_signal: batch_Y, org_siganl: batch_X, batchSize: batch_size})
print(bits)

plt.subplot(2,1,1)
plt.plot(loss)
plt.ylabel('Training loss')

plt.subplot(2,1,2)
plt.plot(bits)
plt.xlabel('Iteration x 100')
plt.ylabel('Correct bits')
plt.show()




