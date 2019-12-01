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

    error_mmse = 1 - (np.sum(accuracy_mmse) / USER)
    #error_mmse = 1 - (np.sum(accuracy_mmse) / (USER * CONS_ALPHABET.shape[1]))

    return error_mmse


def zero_forcing(s, h, y):
    p1 = np.matmul(np.matrix.getH(h), h)
    p2 = np.matmul(np.matrix.getH(h), y)
    xhat = np.matmul(np.linalg.inv(p1), p2)

    v1 = np.matmul(xhat, np.ones([1, CONS_ALPHABET.size]))
    v2 = np.matmul(np.ones([USER, 1]), CONS_ALPHABET)
    idxhat = np.argmin(np.square(np.abs(v1 - v2)), axis=1)

    idx = CONS_ALPHABET[:, idxhat]
    accuracy_zf = np.equal(idx.flatten(), np.transpose(s))

    error_zf = 1 - (np.sum(accuracy_zf) / USER)
    #error_zf = 1 - (np.sum(accuracy_zf) / (USER * CONS_ALPHABET.shape[1]))

    return error_zf


def matched_filter(s, h, y):
    p1 = np.matmul(np.matrix.getH(h), y)

    h_flat = h.reshape(h.shape[0] * h.shape[1], 1)
    _, singular_value, _ = np.linalg.svd(h_flat)
    p2 = singular_value[np.argmax(singular_value)]  #norm

    xhat = p1 * (1/p2)

    v1 = np.matmul(xhat, np.ones([1, CONS_ALPHABET.size]))
    v2 = np.matmul(np.ones([USER, 1]), CONS_ALPHABET)
    idxhat = np.argmin(np.square(np.abs(v1 - v2)), axis=1)

    idx = CONS_ALPHABET[:, idxhat]
    accuracy_mf = np.equal(idx.flatten(), np.transpose(s))

    error_mf = 1 - (np.sum(accuracy_mf) / USER)
    #error_mf = 1 - (np.sum(accuracy_mf) / (USER * CONS_ALPHABET.shape[1]))

    return error_mf


def sphere_detector(s, H, y):
    # Initialization
    Radius = np.inf
    PA = np.zeros([USER, 1],dtype=int)
    ST = np.zeros([USER, CONS_ALPHABET.size])

    # Preprocessing
    Q, R = linalg.qr(H, mode='economic')
    y_hat = np.matmul(np.matrix.getH(Q), y)

    # Add root node to stack
    level = USER - 1
    sub1 = y_hat[level]
    sub2 = R[level, level]
    sub3 = sub2 * CONS_ALPHABET.T
    ST[level, :] = (np.square(np.abs(sub1 - sub3))).T
    path_flag = 1

    # Sphere detector begin
    while(level <= USER-1):
        minPED = np.amin(ST[level, :])
        idx = np.argmin(ST[level, :])

        # Proceed only if list is not empty
        if(minPED < np.inf):
            ST[level, idx] = np.inf

            if (path_flag <= 1 ):
                NewPath = idx

            else:
                new_path_t = PA[level + 1: None, 0]
                NewPath = np.hstack((idx,new_path_t))

            path_flag = path_flag + 1

            # Search child
            if(minPED < Radius):
                if(level > 0):

                    PA[level:None, 0] = NewPath.reshape(-1)

                    level = level - 1
                    PA_t = PA[level + 1: None, 0]
                    PA_t_inv = PA_t.reshape(PA_t.size,1)

                    R_t = R[level,level+1:None]
                    R_t_shape = R_t.reshape(1,R_t.size)

                    DF_t_2 = CONS_ALPHABET[0,PA_t_inv]
                    DF_t_2_inv = DF_t_2.reshape(DF_t_2.size,1)

                    DF = np.matmul(R_t_shape, DF_t_2_inv)


                    tub1 = y_hat[level]
                    tub2 = R[level, level]
                    tub3 = tub2 * CONS_ALPHABET.T

                    ST[level,:]  = minPED + (np.square(np.abs(tub1 - tub3 - DF))).T

                    print('debug')
                else:
                    idxhat = NewPath.reshape(NewPath.size,1)
                    idx = CONS_ALPHABET[:, idxhat]
                    Radius = minPED

        else:
            level = level + 1
    print('Done')

    accuracy_sd = np.equal(idx.flatten(), np.transpose(s))

    error_sd = 1 - (np.sum(accuracy_sd) / USER)
    return error_sd

'''..................................................................................................................'''

'''CHANGE YOUR PARAMETERS HERE'''
ITERATION = 1#100000
USER = 16
RECEIVER = 32

snr_db_list = []
for snr in range(10, 38, 3):
    snr_db_list.append(snr)

CONS_ALPHABET = constellation_alphabet('64QAM')   # choose anything from here BPSK,QPSK,16QAM,64QAM
'''CHANGEABLE PARAMETERS ARE ABOVE THE LINE'''

signal_energy_avg = np.mean(np.square(np.abs(CONS_ALPHABET)))

bers_mmse_in_iter = np.zeros([len(snr_db_list), ITERATION])
bers_zf_in_iter = np.zeros([len(snr_db_list), ITERATION])
bers_mf_in_iter = np.zeros([len(snr_db_list), ITERATION])
bers_sd_in_iter = np.zeros([len(snr_db_list), ITERATION])

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
        bers_zf_in_iter[iter_snr][rerun] = zero_forcing(transmitted_symbol, channel, received_signal)
        bers_mf_in_iter[iter_snr][rerun] = matched_filter(transmitted_symbol, channel, received_signal)
        bers_sd_in_iter[iter_snr][rerun] = sphere_detector(transmitted_symbol, channel, received_signal)

bers_mmse = np.mean(bers_mmse_in_iter, axis=1)
bers_zf = np.mean(bers_zf_in_iter, axis=1)
bers_mf = np.mean(bers_mf_in_iter, axis=1)
bers_sd = np.mean(bers_sd_in_iter, axis=1)

print('mmse error rate', bers_mmse)
print('zero forcing error rate', bers_zf)
print('matched filter error rate', bers_mf)
print('sphere detector error rate', bers_sd)

bits_per_symbol = math.log2(CONS_ALPHABET.shape[1])

bers_mmse = np.divide(bers_mmse, bits_per_symbol)
bers_zf = np.divide(bers_zf, bits_per_symbol)
bers_mf = np.divide(bers_mf, bits_per_symbol)
bers_sd = np.divide(bers_sd, bits_per_symbol)

print('mmse detector error rate after divided by bits_per_symbol', bers_mmse)
print('zero forcing detector error rate after divided by bits_per_symbol', bers_zf)
print('matched filter error rate after divided by bits_per_symbol', bers_mf)
print('sphere detector error rate after divided by bits_per_symbol', bers_sd)

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
