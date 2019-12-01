import numpy as np
from scipy.stats import norm
from scipy import linalg
import matplotlib.pyplot as plt

'''..................................................................................................................'''


def constellation_alphabet(mod):
    if mod == 'BPSK':
        return np.array([[-1, 1]], np.complex)
    elif mod == 'QPSK':
        return np.array([[-1 - 1j, -1 + 1j, 1 - 1j, 1 + 1j]], np.complex)
    elif mod == '16QAM':
        return np.array([[-3 - 3j, -3 - 1j, -3 + 3j, -3 + 1j, - 1 - 3j, -1 - 1j, -1 + 3j, -1 + 1j, + 3 - 3j, +3 - 1j,
                          +3 + 3j, +3 + 1j, + 1 - 3j, +1 - 1j, +1 + 3j, +1 + 1j]], np.complex)
    elif mod == '64QAM':
        return np.array([[-7 - 7j, -7 - 5j, -7 - 1j, -7 - 3j, -7 + 7j, -7 + 5j, -7 + 1j, -7 + 3j, - 5 - 7j, -5 - 5j,
                          -5 - 1j, -5 - 3j, -5 + 7j, -5 + 5j, -5 + 1j, -5 + 3j, - 1 - 7j, -1 - 5j, -1 - 1j, -1 - 3j,
                          -1 + 7j, -1 + 5j, -1 + 1j, -1 + 3j, - 3 - 7j, -3 - 5j, -3 - 1j, -3 - 3j, -3 + 7j, -3 + 5j,
                          -3 + 1j, -3 + 3j, + 7 - 7j, +7 - 5j, +7 - 1j, +7 - 3j, +7 + 7j, +7 + 5j, +7 + 1j, +7 + 3j,
                          + 5 - 7j, +5 - 5j, +5 - 1j, +5 - 3j, +5 + 7j, +5 + 5j, +5 + 1j, +5 + 3j, + 1 - 7j, +1 - 5j,
                          +1 - 1j, +1 - 3j, +1 + 7j, +1 + 5j, +1 + 1j, +1 + 3j, + 3 - 7j, +3 - 5j, +3 - 1j, +3 - 3j,
                          +3 + 7j, +3 + 5j, +3 + 1j, +3 + 3j]], np.complex)


#NewPath = []
def sphere_detector(s, H, y):
    #Initialization
    symbol = []
    Radius = np.inf
    PA = np.zeros([USER, 1], dtype=int)
    ST = np.zeros([USER, CONS_ALPHABET.size])

    #Preprocessing
    Q, R = linalg.qr(H, mode='economic')
    y_hat = np.matmul(np.matrix.getH(Q), y)

    #Add root node to stack
    level = USER - 1
    ST[level, :] = (np.square(np.abs(y_hat[level] - R[level, level] * CONS_ALPHABET.T))).T
    path_flag = 1

    #Sphere detector begin
    while level <= USER - 1:
        minPED = np.amin(ST[level, :])
        idx = np.argmin(ST[level, :])

        #Proceed only if list is not empty
        if minPED < np.inf:
            ST[level, idx] = np.inf

            if path_flag <= 1:
                new_path = idx

            else:
                new_path = np.hstack((idx, PA[level + 1: None, 0]))

            path_flag = path_flag + 1

            #Search child
            if minPED < Radius:
                if level > 0:
                    PA[level:None, 0] = new_path.reshape(-1)
                    level = level - 1

                    PA_t = PA[level + 1: None, 0]
                    R_t = R[level, level + 1:None]
                    DF_t = CONS_ALPHABET[0, PA_t.reshape(PA_t.size, 1)]

                    DF = np.matmul(R_t.reshape(1, R_t.size), DF_t.reshape(DF_t.size, 1))

                    ST[level, :] = minPED + (np.square(np.abs(y_hat[level] - R[level, level] * CONS_ALPHABET.T - DF))).T

                else:
                    idxhat = new_path.reshape(new_path.size, 1)
                    symbol = CONS_ALPHABET[:, idxhat.reshape(-1)]
                    Radius = minPED

        else:
            level = level + 1

    accuracy_sd = np.equal(symbol.flatten(), np.transpose(s))
    error_sd = 1 - (np.sum(accuracy_sd) / USER)
    return error_sd


'''..................................................................................................................'''

'''CHANGE YOUR PARAMETERS HERE'''
ITERATION = 10
USER = 16
RECEIVER = 32

snr_db_list = []
for snr in range(0, 13, 2):
    snr_db_list.append(snr)

CONS_ALPHABET = constellation_alphabet('QPSK')  # choose anything from here BPSK,QPSK,16QAM,64QAM
'''CHANGEABLE PARAMETERS ARE ABOVE THE LINE'''

signal_energy_avg = np.mean(np.square(np.abs(CONS_ALPHABET)))

bers_mmse_in_iter = np.zeros([len(snr_db_list), ITERATION])
bers_zf_in_iter = np.zeros([len(snr_db_list), ITERATION])
bers_mf_in_iter = np.zeros([len(snr_db_list), ITERATION])
bers_sd_in_iter = np.zeros([len(snr_db_list), ITERATION])

for iter_snr in range(len(snr_db_list)):
    snr_db = snr_db_list[iter_snr]

    for rerun in range(ITERATION):
        rand_symbol_ind = (np.random.randint(low=0, high=CONS_ALPHABET.shape[1], size=(USER, 1))).flatten()
        transmitted_symbol = (CONS_ALPHABET[:, rand_symbol_ind]).T

        SNR_lin = 10 ** (snr_db / 10)

        noise_variance = signal_energy_avg * USER / SNR_lin
        noise = np.sqrt(0.5) * (norm.ppf(np.random.rand(RECEIVER, 1)) + (1j * norm.ppf(np.random.rand(RECEIVER, 1))))

        channel = np.sqrt(0.5) * (
                    norm.ppf(np.random.rand(RECEIVER, USER)) + (1j * norm.ppf(np.random.rand(RECEIVER, USER))))

        received_signal = np.matmul(channel, transmitted_symbol) + np.sqrt(noise_variance) * noise


        bers_sd_in_iter[iter_snr][rerun] = sphere_detector(transmitted_symbol, channel, received_signal)

        if(rerun % 1000 == 0):
            print(rerun, 'iteration completed', 'for', snr_db, 'SNR')



bers_sd = np.mean(bers_sd_in_iter, axis=1)
print('sphere detector error rate', bers_sd)

bits_per_symbol = len(CONS_ALPHABET)
bers_sd = np.divide(bers_sd,bits_per_symbol)

print('sphere detector error rate after divided by bits_per_symbol', bers_sd)

plt.figure('Bit Error Rate')
plt.subplot(111)
plt.semilogy(snr_db_list, bers_sd, color='magenta', marker='s', linestyle='-', linewidth=1, markersize=6, label='SD')
plt.title('BER vs SNR')
plt.xlabel('average SNR(dB) per receive antenna')
plt.xlim(0, 12)
plt.xscale('linear')
plt.ylabel('BER')
plt.ylim(0.0001, 1)
plt.grid(b=True, which='major', color='#666666', linestyle='--')
plt.legend(title='Detectors:')
plt.show()