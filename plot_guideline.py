import numpy as np
import matplotlib.pyplot as plt


snr_db_list = [-2, 0, 2, 4, 6, 8,10,12,14,16,18,20]

bers_mmse = [3.910505e-01, 3.232850e-01, 2.423175e-01, 1.587920e-01, 8.847150e-02, 3.998150e-02, 1.367650e-02, 3.472000e-03, 5.570000e-04, 5.350000e-05, 6.500000e-06, 0.000000e+00]
bers_zf = [2.128320e-01, 1.592365e-01, 1.050585e-01, 5.808150e-02, 2.488750e-02, 7.535500e-03, 1.380000e-03, 1.370000e-04, 1.000000e-05, 5.000000e-07, 0.000000e+00, 0.000000e+00]
bers_mf = [0.4453535, 0.4267075, 0.4120415, 0.41491,   0.383046,  0.376442,  0.394149, 0.4039215, 0.403902, 0.3975615, 0.38854,   0.380159 ]

plt.figure('Bit Error Rate-1')
plt.subplot(211)
plt.plot(snr_db_list, bers_mmse, color='black', marker='*', linestyle='-', linewidth=1, markersize=6, label='MMSE')
plt.title('BER vs SNR')
plt.xlabel('SNR(dB)')
plt.xlim(-2, 16)
plt.ylabel('BER')
plt.yscale('log')
plt.ylim(0.00001, 1)
plt.grid(b=True, which='major', color='#666666', linestyle='--')
plt.legend(title='Detectors:')


plt.figure('Bit Error Rate-2')
plt.subplot(111)
plt.plot(snr_db_list, bers_mmse, color='black', marker='*', linestyle='-', linewidth=1, markersize=6, label='MMSE')
plt.plot(snr_db_list, bers_zf, color='blue', marker='d', linestyle='-', linewidth=1, markersize=5, label='ZF')
plt.plot(snr_db_list, bers_mf, color='red', marker='x', linestyle='-', linewidth=1, markersize=6, label='MF')
plt.title('BER vs SNR')
plt.xlabel('SNR(dB)')
plt.xlim(-2, 16)
plt.xscale('linear')
plt.ylabel('BER')
plt.yscale('log')
plt.ylim(0.00001, 1)
plt.grid(b=True, which='major', color='#666666', linestyle='--')
plt.legend(title='Detectors:')


plt.figure('Bit Error Rate-1')
plt.subplot(212)
plt.plot(snr_db_list, bers_zf, color='blue', marker='d', linestyle='-', linewidth=1, markersize=5, label='ZF')
plt.title('BER vs SNR')
plt.xlabel('SNR(dB)')
plt.xlim(-2, 16)
plt.xscale('linear')
plt.ylabel('BER')
plt.yscale('log')
plt.ylim(0.00001, 1)
plt.grid(b=True, which='major', color='#666666', linestyle='--')
plt.legend(title='Detectors:')
plt.show()

'''

x1 = np.linspace(0.0, 5.0)
x2 = np.linspace(0.0, 2.0)

y1 = np.cos(2 * np.pi * x1) * np.exp(-x1)
y2 = np.cos(2 * np.pi * x2)

plt.figure('1')
plt.subplot(3, 1, 1)
plt.plot(x1, y1, 'o-')
plt.title('A tale of 2 subplots')
plt.ylabel('Damped oscillation')
plt.ylim()

plt.subplot(3, 1, 2)
plt.plot(x2, y2, '.-')
plt.xlabel('time (s)')
plt.ylabel('Undamped')



plt.figure('2')
plt.subplot(2, 1, 1)
plt.plot(x1, y1, 'o-')
plt.title('A tale of 2 subplots')
plt.ylabel('Damped oscillation')
plt.yscale('log')

plt.subplot(2, 1, 2)
plt.plot(x2, y2, '.-')
plt.xlabel('time (s)')
plt.ylabel('Undamped')

plt.figure('1')
plt.subplot(3, 1, 3)
plt.plot(x1, y1, 'o-')
plt.title('A tale of 2 subplots')
plt.ylabel('Damped oscillation')


'''