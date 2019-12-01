import numpy as np
import tensorflow as tf

sess = tf.InteractiveSession()
'''
x = np.linspace(0, 1, 201)
y = np.random.random(201)

header = "SNR, BER\n"
#header += "This is a second line"
f = open('Results\AD_data.txt', 'wb')
np.savetxt(f, [], header=header)
for i in range(5):
    data = np.column_stack((x[i], y[i]))
    np.savetxt(f, data)
f.close()
'''

'''
file_append = open("Results\AD_data.txt","w")
file_append.write("Human resource")
file_append.close()
file = open("Results\AD_data.txt","r")
print(file.read())
file.close()
'''

#h = np.array([[-0.5232 - 0.5608j, 0.2169 + 0.1397j]])
'''
h = np.sqrt(0.5) * (np.random.rand(30, 20) + (1j * np.random.rand(30, 20)))

print('original h',h)
print(h.shape)

h_flat = h.reshape(h.shape[0]*h.shape[1],1)

print('flattened h',h_flat)

print(h.shape)

_, singular_value, _ = np.linalg.svd(h_flat)

norm = singular_value[np.argmax(singular_value)]

print(norm)

print(np.random.randint(13, size=(1, 40)) - 2)
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

print(CONS_ALPHABET.shape[1])

rand_symbol_ind = np.random.randint(0,CONS_ALPHABET.shape[1],size=(10,1))

transmitted_symbol = CONS_ALPHABET[:, rand_symbol_ind]

print('transmitted_symbol',transmitted_symbol)
print('rand_symbol_ind',rand_symbol_ind.shape)
print('CONS_ALPHABET',CONS_ALPHABET)

bit_per_symbol = np.log2(CONS_ALPHABET.shape[1])
'''
'''
some_list = ['1', 'B', '3', 'D', '5', 'F']
print(some_list[0:len(some_list)-1])
'''

import numpy as np
from scipy.stats import norm
from scipy import linalg
import matplotlib.pyplot as plt

ITERATION = 2
USER = 4
RECEIVER = 4


CONS_ALPHABET = np.array([[-1, 1]], np.complex)
signal_energy_avg = np.mean(np.square(np.abs(CONS_ALPHABET)))

snr_db_list = []
for snr in range(-2, 17, 2):
    snr_db_list.append(snr)

transmitted_symbol = np.transpose(np.sign(np.random.rand(1, USER) - 0.5))

SNR_lin = 10 ** (10 / 10)

noise_variance = signal_energy_avg * USER / SNR_lin

noise = np.sqrt(0.5) * (norm.ppf(np.random.rand(RECEIVER, 1)) + (1j * norm.ppf(np.random.rand(RECEIVER, 1))))

channel = np.sqrt(0.5) * (norm.ppf(np.random.rand(RECEIVER, USER)) + (1j * norm.ppf(np.random.rand(RECEIVER, USER))))

y = np.matmul(channel, transmitted_symbol) + np.sqrt(noise_variance) * noise

Radius = np.inf
PA = np.zeros([USER, 1])
ST = np.zeros([USER, CONS_ALPHABET.size])

#Preprocessing
Q, R = linalg.qr(channel, mode='economic')
y_hat = np.matmul(np.matrix.getH(Q),y)

#Add root node to stack
level = USER-1
sub1 = y_hat[level]
sub2 = R[level,level] * CONS_ALPHABET.T
ST[level, :] = (np.square(np.abs(sub1 - sub2))).T
#print(PA)
#print(PA.shape[0])

#some_list = ['1', 'B', '3', 'D', '5', 'F']
some_list = np.array([[1],[2],[3],[4]])
print(some_list.shape[0])
print(some_list[3:some_list.shape[0],0])

print(bin(1))
print(bin(2))
print(bin(1))
print(bin(15))


a = np.array([[1., 2., 3.], [4., 5., -2.],[6., 7., 9.]])

a[1,:] = np.zeros(3)
print(a.shape)
print(np.zeros(3))
print('sh',np.zeros(3).shape)
print(a)


test = np.array([[0, 2, 3, 4, 5],[1, 2, 3, 4, 5],[2, 2, 3, 4, 5],[3, 2, 3, 4, 5],[4, 2, 3, 4, 5]])

print(test[3:None,:])

snr_db_listrrr = []
for snr in range(10, 38, 3):
    snr_db_listrrr.append(snr)

print(snr_db_listrrr)



at = np.array([[-1-1j, -1+1j, 1-1j, 1+1j]], np.complex)

print(at[:,[1,2]])

com = at[0,1]
print(com.real)
print(com.imag)
