import uniMes
import matplotlib.pyplot as plt


vec_set_1, vec_set_2 = uniMes.functions.vector_array(uniMes.data_points_resampled, [26,24,"Bende:r_knee","Bende:r_hip"],[24,12,"Bende:r_hip","Bende:r_shoulder"])

angles = uniMes.functions.angle_plotting_pair(vec_set_1,vec_set_2,False)

pair = 'squat_2'
ot = 'ot_'+pair+'.csv'
mp_w = 'mp_pose_world_'+pair+'.csv'
mp = 'mp_pose_'+pair+'.csv'

angle_dev_mp = []
# angles[pair][ot].pop()
# angles[pair][mp_w].pop()
# try: 
#     for i in range(len(angles[pair][ot])):
#         angle_dev_mp.append(abs(angles[pair][ot][i]-angles[pair][mp][i]))
# except KeyError:
#     pass
angle_dev_mp_w = []
try: 
    for i in range(len(angles[pair][ot])):
        angle_dev_mp_w.append(abs(angles[pair][ot][i]-angles[pair][mp_w][i]))
except KeyError:
    pass
# for i in range(len(angles[pair][ot])):
#     angle_dev_mp_w.append(abs(angles[pair][ot][i]-angles[pair][mp_w][i]))


# label_for_plots = [mp, mp_w]
# angle_dev = [angle_dev_mp, angle_dev_mp_w]
# label_for_plots = [mp_w]
# angle_dev = [angle_dev_mp_w]
# fig, ax = plt.subplots()
# plt.title("Bezárt szögek: " + pair)
# bp_adatok = ax.boxplot(angle_dev, labels=label_for_plots, notch=True, showmeans=True)
# plt.ylabel("Bezárt szögek [°]")
# plt.xlabel("Adatsor")
# plt.show()
# f = "C:/dev/thesis/data/plots/BoxPlotd/angles" + pair + ".svg"
# plt.savefig(f, format='svg')
# uniMes.functions.bp_data(bp_adatok, pair)

xn = angles[pair][ot]

from scipy.signal import butter, lfilter

t = uniMes.time_resampled[pair][ot]

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

from scipy import signal

fs = 5000.0
lowcut = 500.0
highcut = 1250.0

# b, a = butter_bandpass_filter(angle_dev_mp_w,lowcut,highcut,fs)

b, a = signal.butter(3, [0.001, 0.5], btype='band')

zi = signal.lfilter_zi(b, a)

z, _ = signal.lfilter(b, a, xn, zi=zi*xn[0])
z2, _ = signal.lfilter(b, a, z, zi=zi*z[0])
y = signal.filtfilt(b, a, xn)
plt.figure

plt.plot(t, xn, 'b', alpha=0.75)

plt.plot(t, z, 'r--', t, z2, 'r', t, y, 'k')

plt.legend(('noisy signal', 'lfilter, once', 'lfilter, twice',

            'filtfilt'), loc='best')

plt.grid(True)

plt.show()
import numpy as np

import matplotlib.pyplot as plt

t = uniMes.time_resampled[pair][ot]

sp = np.fft.fft(y)
# for i in range(5):
#     sp.pop(0)
# for i in range(300):
#     sp.pop()
freq = np.fft.fftfreq(t.shape[-1])
print(np.argmax(sp), np.argmin(sp))
print(sp[np.argmin(sp)], freq[np.argmin(sp)])
plt.plot(freq,sp.real)

plt.show()