import tensorflow as tf
from scipy.spatial import distance
import numpy as np

for x in range(2, 30, 3):
  print(x)

'''
y_new_e = list()
for m in range(2):

    val = tf.ones([1, 2])
    val = val * m
    y_new_e.append(val)

tthshi = tf.stack(y_new_e)
hshi = tf.reshape(tthshi, [2, 2])

sess = tf.InteractiveSession()

print(sess.run(hshi))
print(sess.run(tthshi))
'''




'''
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


CONS_ALPHABET = constellation_alphabet('BPSK')


sess = tf.InteractiveSession()

y_t = [-5, -0.5, 1, 1.2]
y_t_z = np.zeros([1,4])
y = [1., 5., 5., -5.]
xhat = [1, 1, -1, -1]


y_reshape = tf.reshape(y, [2, 2])

softmax_y = tf.nn.softmax(y_reshape, 1)
relu_y = tf.nn.relu(y_reshape)
tanh_y = tf.nn.tanh(y_t)

test_threshold = tf.math.greater(y_t, y_t_z, name=None)
positive_result = tf.cast(test_threshold, dtype=float)
signed_value = tf.sign(y_t)

v1 = np.matmul(xhat, np.ones([1, CONS_ALPHABET.size]))
v2 = np.matmul(np.ones([4, 1]), CONS_ALPHABET)
idxhat = np.argmin(np.square(np.abs(v1 - v2)), axis=1)

idx = CONS_ALPHABET[:, idxhat]
accuracy_zf = np.equal(idx.flatten(), np.transpose(xhat))

error_zf = 1 - (np.sum(accuracy_zf) / 4)

session = tf.Session()

print("Y", session.run(y_reshape))
print("SOFTMAX Y", session.run(softmax_y))
print('tanh output', session.run(tanh_y))
print('test threshold out', session.run(test_threshold))
print('positive value counted', session.run(positive_result))
print('signed value', session.run(signed_value))
'''
'''
y = [1., 5., 5., 5., 5.5, 6.5, 4.5, 4.]
y_reshape = tf.reshape(y, [2, 2, 2])
softmax_y = tf.nn.softmax(y_reshape, 2)

session = tf.Session()

print("Y")
print(session.run(y_reshape))
print("SOFTMAX Y")
print(session.run(softmax_y))
'''
'''
t = [[[1, 1, 5], [2, 6, 2]],[[3, 3, 7], [4, 4, 8]]]

t_shape = tf.reshape(t,[4,3])

print(sess.run(t_shape))

print(sess.run(tf.argmax(t_shape,axis=1)))
'''
