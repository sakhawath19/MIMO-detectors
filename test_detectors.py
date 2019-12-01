import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

ITERATION = 100000
USER = 20
RECEIVER = 30


CONS_ALPHABET = np.array([[-1, 1]], np.complex)
signal_energy_avg = np.mean(np.square(np.abs(CONS_ALPHABET)))

snr_db_list = []
for snr in range(-2, 17, 2):
    snr_db_list.append(snr)


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

    return error_mmse

bers_mmse_in_iter = np.zeros([len(snr_db_list), ITERATION])


for iter_snr in range(len(snr_db_list)):
    snr_db = snr_db_list[iter_snr]

    for rerun in range(ITERATION):
        transmitted_symbol = np.transpose(np.sign(np.random.rand(1, USER) - 0.5))

        SNR_lin = 10 ** (snr_db / 10)

        noise_variance = signal_energy_avg * USER / SNR_lin

        noise = np.sqrt(0.5) * (norm.ppf(np.random.rand(RECEIVER, 1)) + (1j * norm.ppf(np.random.rand(RECEIVER, 1))))
        #noise = np.sqrt(0.5) * (np.random.rand(RECEIVER, 1) + (1j * np.random.rand(RECEIVER, 1)))
        # n = np.sqrt(0.5) * (norm.ppf(np.random.rand(MR,1)) + 1j * norm.ppf(np.random.rand(MR,1)))
        channel = np.sqrt(0.5) * (norm.ppf(np.random.rand(RECEIVER, USER)) + (1j * norm.ppf(np.random.rand(RECEIVER, USER))))
        #channel = np.sqrt(0.5) * (np.random.rand(RECEIVER, USER) + (1j * np.random.rand(RECEIVER, USER)))

        received_signal = np.matmul(channel, transmitted_symbol) + np.sqrt(noise_variance) * noise

        bers_mmse_in_iter[iter_snr][rerun] = mmse(transmitted_symbol, noise_variance, channel, received_signal)

bers_mmse = np.mean(bers_mmse_in_iter, axis=1)

print('mmse error rate', bers_mmse)


plt.figure('Bit Error Rate')
plt.subplot(111)
plt.semilogy(snr_db_list, bers_mmse, color='black', marker='*', linestyle='-', linewidth=1, markersize=6, label='MMSE')
plt.title('BER vs SNR')
plt.xlabel('SNR(dB)')
plt.xlim(-2, 16)
plt.xscale('linear')
plt.ylabel('BER')
#plt.yscale('log')
plt.ylim(0.00001, 1)
plt.grid(b=True, which='major', color='#666666', linestyle='--')
plt.legend(title='Detectors:')
plt.show()