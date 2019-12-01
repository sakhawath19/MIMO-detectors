import matplotlib.pyplot as plt
a = [pow(10,i) for i in range (10)]

fig2 = plt.figure(2)
ax2 = fig2.add_subplot(111)
ax2.plot(a)
ax2.set_title('gggggggg')
ax2.set_xlabel('gggggggg')
ax2.set_ylabel('gggggggg')
ax2.set_yscale('log')

plt.show()
'''
plt.ylabel('BER')
plt.xlabel('SNR(dB)')
plt.show()
'''