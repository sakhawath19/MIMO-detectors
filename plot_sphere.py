import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

snr_db_list = []
for snr in range(0, 13, 2):
    snr_db_list.append(snr)

bers_sd = [0.19627, 0.1611825, 0.117385, 0.07412625, 0.03795375, 0.0152875, 0.00458875]

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


