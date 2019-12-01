#from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

iteration = 100000

snr_db_list = []
for snr in range(-2, 21, 2):
    snr_db_list.append(snr)

user = 20
receiver = 30

cons_alphabet = np.array([[-1, 1]], np.complex)

signal_energy_avg = np.mean(np.square(np.abs(cons_alphabet)))


def mmse(s, n0, h, y):

    p1 = np.matmul(np.matrix.getH(h), y)
    p2 = np.matmul(np.matrix.getH(h), h) + (n0 / signal_energy_avg) * np.identity(user)
    xhat = np.matmul(np.linalg.inv(p2), p1)

    v1 = np.matmul(xhat, np.ones([1, cons_alphabet.size]))
    v2 = np.matmul(np.ones([user, 1]), cons_alphabet)
    idxhat = np.argmin(np.square(np.abs(v1 - v2)), axis=1)

    idx = cons_alphabet[:, idxhat]
    accuracy_mmse = np.equal(idx.flatten(), np.transpose(s))

    error_mmse = 1 - (np.sum(accuracy_mmse) / user)

    return error_mmse


def zero_forcing(s, h, y):

    p1 = np.matmul(np.matrix.getH(h), y)
    p2 = np.matmul(np.matrix.getH(h), h)
    xhat = np.matmul(np.linalg.inv(p2), p1)

    v1 = np.matmul(xhat, np.ones([1, cons_alphabet.size]))
    v2 = np.matmul(np.ones([user, 1]), cons_alphabet)
    idxhat = np.argmin(np.square(np.abs(v1 - v2)), axis=1)

    idx = cons_alphabet[:, idxhat]
    accuracy_mmse = np.equal(idx.flatten(), np.transpose(s))

    error_zf = 1 - (np.sum(accuracy_mmse) / user)

    return error_zf


bers_mmse_in_iter = np.zeros([len(snr_db_list), iteration])
bers_zf_in_iter = np.zeros([len(snr_db_list), iteration])

for iter_snr in range(len(snr_db_list)):
    snr_db = snr_db_list[iter_snr]

    for rerun in range(iteration):
        transmitted_symbol = np.transpose(np.sign(np.random.rand(1, user) - 0.5))

        SNR_lin = 10 ** (snr_db / 10)

        noise_variance = signal_energy_avg * user / SNR_lin

        noise = np.sqrt(0.5) * (np.random.rand(receiver, 1) + (1j * np.random.rand(receiver, 1)))

        channel = np.sqrt(0.5) * (np.random.rand(receiver, user) + (1j * np.random.rand(receiver, user)))

        received_signal = np.matmul(channel, transmitted_symbol) + np.sqrt(noise_variance) * noise

        bers_mmse_in_iter[iter_snr][rerun] = mmse(transmitted_symbol, noise_variance, channel, received_signal)

        bers_zf_in_iter[iter_snr][rerun] = zero_forcing(transmitted_symbol, channel, received_signal)

bers_mmse = np.mean(bers_mmse_in_iter, axis=1)
bers_zf = np.mean(bers_zf_in_iter, axis=1)

print('mmse error rate', bers_mmse)
print('zero forcing error rate', bers_zf)

fig2 = plt.figure(2)
ax2 = fig2.add_subplot(111)
ax2.plot(snr_db_list, bers_mmse, color='black', marker='*', linestyle='-', linewidth=1, markersize=6, label='MMSE')
ax2.plot(snr_db_list, bers_zf, color='blue', marker='d', linestyle='-', linewidth=1, markersize=5, label='ZF')
ax2.set_title('BER vs SNR')
ax2.set_xlabel('SNR(dB)')
ax2.set_ylabel('BER')
ax2.set_yscale('log')
ax2.set_ylim(0.00001, 0.1)
ax2.set_xlim(-2, 16)
plt.grid(b=True, which='major', color='#666666', linestyle='--')
plt.legend(title='Detectors:')
plt.show()
