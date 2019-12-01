import numpy as np
from scipy.stats import norm
from scipy import linalg
import matplotlib.pyplot as plt
import math

'''..................................................................................................................'''
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


def mmse(s, n0, h, y):
    p1 = np.matmul(np.matrix.getH(h), y)
    p2 = np.matmul(np.matrix.getH(h), h) + (n0 / signal_energy_avg) * np.identity(USER)
    xhat = np.matmul(np.linalg.inv(p2), p1)

    v1 = np.matmul(xhat, np.ones([1, CONS_ALPHABET.size]))
    v2 = np.matmul(np.ones([USER, 1]), CONS_ALPHABET)
    idxhat = np.argmin(np.square(np.abs(v1 - v2)), axis=1)

    estimated_symbol = CONS_ALPHABET[:, idxhat]
    accuracy_mmse = np.equal(estimated_symbol.flatten(), np.transpose(s))

    #error_mmse = 1 - (np.sum(accuracy_mmse) / USER)
    error_mmse = 1 - (np.sum(accuracy_mmse) / (USER * math.log2(CONS_ALPHABET.shape[1])))

    return error_mmse

'''CHANGE YOUR PARAMETERS HERE'''
ITERATION = 100000#100000
USER = 16
RECEIVER = 32

snr_db_list = []
for snr in range(10, 38, 3):
    snr_db_list.append(snr)

CONS_ALPHABET = constellation_alphabet('BPSK') #choose anything from here BPSK,QPSK,16QAM,64QAM
'''CHANGEABLE PARAMETERS ARE ABOVE THE LINE'''

signal_energy_avg = np.mean(np.square(np.abs(CONS_ALPHABET)))

bers_mmse_in_iter = np.zeros([len(snr_db_list), ITERATION])

for iter_snr in range(len(snr_db_list)):
    snr_db = snr_db_list[iter_snr]

    for rerun in range(ITERATION):
        rand_symbol_ind = (np.random.randint(low=0, high=CONS_ALPHABET.shape[1], size=(USER,1))).flatten()
        transmitted_symbol = (CONS_ALPHABET[:, rand_symbol_ind]).T

        SNR_lin = 10 ** (snr_db / 10)

        noise_variance = signal_energy_avg * USER / SNR_lin
        noise = np.sqrt(0.5) * (norm.ppf(np.random.rand(RECEIVER, 1)) + (1j * norm.ppf(np.random.rand(RECEIVER, 1))))

        channel = np.sqrt(0.5) * (norm.ppf(np.random.rand(RECEIVER, USER)) + (1j * norm.ppf(np.random.rand(RECEIVER, USER))))

        received_signal = np.matmul(channel, transmitted_symbol) + np.sqrt(noise_variance) * noise

        bers_mmse_in_iter[iter_snr][rerun] = mmse(transmitted_symbol, noise_variance, channel, received_signal)


bers_mmse = np.mean(bers_mmse_in_iter, axis=1)

print('mmse error rate', bers_mmse)

plt.figure('Bit Error Rate')
plt.subplot(111)
plt.semilogy(snr_db_list, bers_mmse, color='black', marker='*', linestyle='-', linewidth=1, markersize=6, label='MMSE')
plt.title('BER vs SNR')
plt.xlabel('average SNR(dB) per receive antenna')
plt.xlim(-2, 16)
plt.xscale('linear')
plt.ylabel('BER')
plt.ylim(0.0001, 1)
plt.grid(b=True, which='major', color='#666666', linestyle='--')
plt.legend(title='Detectors:')
plt.show()
