import numpy as np
from scipy.stats import norm
from scipy import linalg
import matplotlib.pyplot as plt

ITERATION = 4
USER = 4
RECEIVER = 4
NewPath = []

#CONS_ALPHABET = np.array([[-1, 1]], np.complex)
CONS_ALPHABET = np.array([[-1-1j, -1+1j, 1-1j, 1+1j]], np.complex)
signal_energy_avg = np.mean(np.square(np.abs(CONS_ALPHABET)))

snr_db_list = []
for snr in range(-2, 3, 2):
    snr_db_list.append(snr)


#def SD(par,H,y):
def SD(H,y):
    #Initialization
    Radius = np.inf
    PA = np.zeros([USER, 1],dtype=int)
    ST = np.zeros([USER, CONS_ALPHABET.size])

    #Preprocessing
    Q, R = linalg.qr(H, mode='economic')
    y_hat = np.matmul(np.matrix.getH(Q),y)

    #Add root node to stack
    level = USER - 1
    sub1 = y_hat[level]
    sub2 = R[level, level]
    sub3 = sub2 * CONS_ALPHABET.T
    #sub4 = sub1 - sub3
    #sub5 = np.abs(sub4)
    #sub6 = (np.square(sub5)).T
    ST[level, :] = (np.square(np.abs(sub1 - sub3))).T
    path_flag = 1

    #Sphere detector begin
    while(level <= USER-1):
        minPED = np.amin(ST[level, :])
        idx = np.argmin(ST[level, :])

        #Proceed only if list is not empty
        if(minPED < np.inf):
            ST[level, idx] = np.inf

            if (path_flag <= 1 ):
                NewPath = idx

            else:
                new_path_t = PA[level + 1: None, 0]
                NewPath = np.hstack((idx,new_path_t))

            path_flag = path_flag + 1


            #Search child
            if(minPED < Radius):
                if(level > 0):

                    PA[level:None,0] = NewPath.reshape(-1)

                    level = level - 1
                    PA_t = PA[level + 1: None, 0]
                    PA_t_inv = PA_t.reshape(PA_t.size,1)

                    R_t =  R[level,level+1:None]
                    R_t_shape =  R_t.reshape(1,R_t.size)

                    DF_t_2 = CONS_ALPHABET[0,PA_t_inv]
                    DF_t_2_inv = DF_t_2.reshape(DF_t_2.size,1)

                    DF = np.matmul(R_t_shape, DF_t_2_inv)


                    tub1 = y_hat[level]
                    tub2 = R[level, level]
                    tub3 = tub2 * CONS_ALPHABET.T

                    ST[level,:]  = minPED + (np.square(np.abs(tub1 - tub3 - DF))).T

                    print('debug')
                else:
                    idxML = NewPath.reshape(NewPath.size,1)
                    bitML = CONS_ALPHABET[:, idxML]
                    Radius = minPED

        else:
            level = level + 1
    print('Done')
    return bitML


bers_SD_in_iter = np.zeros([len(snr_db_list), ITERATION])


for iter_snr in range(len(snr_db_list)):
    snr_db = snr_db_list[iter_snr]

    for rerun in range(ITERATION):
        transmitted_symbol = np.transpose(np.sign(np.random.rand(1, USER) - 0.5))

        SNR_lin = 10 ** (snr_db / 10)

        noise_variance = signal_energy_avg * USER / SNR_lin

        noise = np.sqrt(0.5) * (norm.ppf(np.random.rand(RECEIVER, 1)) + (1j * norm.ppf(np.random.rand(RECEIVER, 1))))

        channel = np.array([[ 0.5015144+0.26078877j,  0.2987252-1.5511648j,  -1.15782837+0.03402555j,  0.07356947+0.1406075j ],
                   [ 0.5320748-1.13851981j,  1.40312018+1.53508133j, -0.82438685+0.20425308j,  0.63467716-1.27192349j],
                   [-0.02480245-0.95957718j,  0.73754743+0.53069404j, -0.1931817 +0.16103801j,  0.33843276+0.48109717j],
                   [-1.41735352+0.35408847j, -0.14613504+0.04884191j,  1.15626494-0.20988942j, -0.34643862-0.84644837j]])
        #channel = np.sqrt(0.5) * (norm.ppf(np.random.rand(RECEIVER, USER)) + (1j * norm.ppf(np.random.rand(RECEIVER, USER))))

        received_signal = np.array([[-2.26710803+3.39126938j], [-4.66079697+0.47054256j], [-3.12954827-0.62459929j],
                           [-3.14256288+1.13771521j]])
        #received_signal = np.matmul(channel, transmitted_symbol) + np.sqrt(noise_variance) * noise

        print(channel)
        print(received_signal)
        temp_sym = SD(channel, received_signal)
        #bers_SD_in_iter[iter_snr][rerun] = SD(channel, received_signal)
        print('detected symbol',temp_sym)


bers_mmse = np.mean(bers_SD_in_iter, axis=1)

