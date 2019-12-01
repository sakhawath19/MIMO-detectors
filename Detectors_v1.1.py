import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
#from numpy import array


iteration = 100000

snr_db_list = [-2, 0, 2, 4, 6, 8,10,12,14,16,18,20]

MT = 20
MR = 30

cons_alphabet = np.array([[-1, 1]], np.complex)
#cons_alphabet = np.matrix([[-1, 1]]) #constellation alphabet

Es = np.mean(np.square(np.abs(cons_alphabet)))

def mmse(s, snr_db):

    SNR_lin = 10 ** (snr_db / 10)

    N0 = Es*MT/SNR_lin

    n = np.sqrt(0.5) * (np.random.rand(MR,1) + (1j * np.random.rand(MR,1)))
    #n = np.sqrt(0.5) * (norm.ppf(np.random.rand(MR,1)) + 1j * norm.ppf(np.random.rand(MR,1)))

    H = np.sqrt(0.5) * (np.random.rand(MR,MT) + (1j * np.random.rand(MR,MT)))
    #H = np.sqrt(0.5) * (norm.ppf(np.random.rand(MR,MT)) + 1j * norm.ppf(np.random.rand(MR,MT)))

    y = np.matmul(H,s) + np.sqrt(N0) * n

    p1 = np.matmul(np.matrix.getH(H), y)
    p2 = np.matmul(np.matrix.getH(H), H) + (N0 / Es) * np.identity(MT)
    xhat = np.matmul(np.linalg.inv(p2),p1)

    v1 = np.matmul(xhat, np.ones([1, cons_alphabet.size]))
    v2 = np.matmul(np.ones([MT, 1]), cons_alphabet)
    idxhat = np.argmin(np.square(np.abs(v1 - v2)), axis=1)

    idx = cons_alphabet[:, idxhat]
    accuracy_mmse = np.equal(idx.flatten(), np.transpose(s))

    error_mmse = 1 - (np.sum(accuracy_mmse) / MT)

    return error_mmse

def zero_forcing(snr_db, s, Es):

    SNR_lin = 10 ** (snr_db / 10)

    N0 = Es*MT/SNR_lin

    n = np.sqrt(0.5) * (np.random.rand(MR,1) + (1j * np.random.rand(MR,1)))
    #n = np.sqrt(0.5) * (norm.ppf(np.random.rand(MR,1)) + 1j * norm.ppf(np.random.rand(MR,1)))

    H = np.sqrt(0.5) * (np.random.rand(MR,MT) + (1j * np.random.rand(MR,MT)))
    #H = np.sqrt(0.5) * (norm.ppf(np.random.rand(MR,MT)) + 1j * norm.ppf(np.random.rand(MR,MT)))

    y = np.matmul(H,s) + np.sqrt(N0) * n

    p1 = np.matmul(np.matrix.getH(H), y)
    p2 = np.matmul(np.matrix.getH(H), H)
    xhat = np.matmul(np.linalg.inv(p2),p1)

    v1 = np.matmul(xhat, np.ones([1, cons_alphabet.size]))
    v2 = np.matmul(np.ones([MT, 1]), cons_alphabet)
    idxhat = np.argmin(np.square(np.abs(v1 - v2)), axis=1)

    idx = cons_alphabet[:, idxhat]
    accuracy_mmse = np.equal(idx.flatten(), np.transpose(s))

    error_zf = 1 - (np.sum(accuracy_mmse) / MT)

    return error_zf

bers_mmse_in_iter = np.zeros([len(snr_db_list), iteration])
bers_zf_in_iter = np.zeros([len(snr_db_list), iteration])

for iter_snr in range(len(snr_db_list)):
    snr = snr_db_list[iter_snr]

    for iter in range(iteration):

        transmitted_symbol = np.transpose(np.sign(np.random.rand(1, MT) - 0.5))

        SNR_lin = 10 ** (snr / 10)

        noise_variance = Es * MT / SNR_lin

        noise = np.sqrt(0.5) * (np.random.rand(MR, 1) + (1j * np.random.rand(MR, 1)))

        noise_std = np.sqrt(noise_variance)

        channel = np.sqrt(0.5) * (np.random.rand(MR, MT) + (1j * np.random.rand(MR, MT)))

        received_signal = np.matmul(channel, transmitted_symbol) + noise_std * noise

        bers_mmse_in_iter[iter_snr][iter] = mmse(transmitted_symbol,snr)

        bers_zf_in_iter[iter_snr][iter] = zero_forcing(snr, transmitted_symbol, Es)

bers_mmse = np.mean(bers_mmse_in_iter, axis=1)
bers_zf = np.mean(bers_zf_in_iter, axis=1)

print('mmse error rate',bers_mmse)
print('zero forcing error rate',bers_zf)

fig2 = plt.figure(2)
ax2 = fig2.add_subplot(111)
ax2.plot(snr_db_list,bers_mmse, color='black', marker='*', linestyle='-',linewidth=1, markersize=6,label='MMSE')
ax2.plot(snr_db_list,bers_zf, color='blue', marker='d', linestyle='-',linewidth=1, markersize=5,label='ZF')
ax2.set_title('BER vs SNR')
ax2.set_xlabel('SNR(dB)')
ax2.set_ylabel('BER')
ax2.set_yscale('log')
ax2.set_ylim(0.00001,0.1)
ax2.set_xlim(-2,16)
plt.grid(b=True, which='major', color='#666666', linestyle='--')
plt.legend(title = 'Detectors:')
plt.show()

