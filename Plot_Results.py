import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

snr_db_list = []
for snr in range(7, 13, 1):
    snr_db_list.append(snr)

bers_mmse = [1.6260e-03, 7.5375e-04, 3.5100e-04, 1.4850e-04, 6.7000e-05, 2.4750e-05]
bers_zf = [2.67575e-03, 1.29175e-03, 6.01750e-04, 2.72750e-04, 1.08750e-04, 4.30000e-05]
bers_nn = [0.0440172,  0.0278407,  0.0160341,  0.0082269,  0.00370295, 0.0014039 ]


# bits_per_symbol = 6

# bers_mmse = np.divide(bers_mmse,bits_per_symbol)
# bers_zf = np.divide(bers_zf,bits_per_symbol)
# bers_nn = np.divide(bers_zf,bits_per_symbol)
# bers_mf = np.divide(bers_mf,bits_per_symbol)
# bers_sd = np.divide(bers_mf,bits_per_symbol)

fig1 = plt.figure('Python, BPSK, TxR=20x30, Iteration=1000000,SNR=7-12')
#fig2 = plt.figure('Python, 64QAM, TxR=16x32, Iteration=100000,SNR=-2-2-16, WRONG RESULT')
ax1 = fig1.add_subplot(111)
#ax1.plot(snr_db_list, bers_mmse, color='black', marker='*', linestyle='-', linewidth=1, markersize=6, label='MMSE')
#ax1.plot(snr_db_list, bers_zf, color='blue', marker='d', linestyle='-', linewidth=1, markersize=5, label='ZF')
ax1.plot(snr_db_list, bers_nn, color='green', marker='>', linestyle='-', linewidth=1, markersize=5, label='NN(20x30)')
#ax1.plot(snr_db_list, bers_mf, color='red', marker='x', linestyle='-', linewidth=1, markersize=6, label='MF')
#ax1.plot(snr_db_list, bers_sd, color='magenta', marker='s', linestyle='-', linewidth=1, markersize=6, label='SD')
ax1.set_title('BER vs SNR')
ax1.set_xlabel('SNR(dB)')
ax1.set_ylabel('BER')
ax1.set_yscale('log')
ax1.set_ylim(0.00001, 1)
ax1.set_xlim(7, 12)
plt.grid(b=True, which='major', color='#666666', linestyle='--')
plt.legend(title='Detectors:')
plt.show()


