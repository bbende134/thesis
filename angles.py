import uniMes
import matplotlib.pyplot as plt


vec_set_1, vec_set_2 = uniMes.functions.vector_array(uniMes.data_points_resampled, [16,14,"Bende:r_wrist","Bende:r_elbow"],[12,14,"Bende:r_shoulder","Bende:r_elbow"])
angles = uniMes.functions.angle_plotting_pair(vec_set_1,vec_set_2,True)

pair = 'star_1'
ot = 'ot_'+pair+'.csv'
mp_w = 'mp_pose_world_'+pair+'.csv'
mp = 'mp_pose_'+pair+'.csv'

angle_dev_mp = []
# angles[pair][ot].pop()
# angles[pair][mp_w].pop()
try: 
    for i in range(len(angles[pair][ot])):
        angle_dev_mp.append(abs(angles[pair][ot][i]-angles[pair][mp][i]))
except KeyError or IndexError:
    print("key error")
    pass
angle_dev_mp_w = []
try: 
    for i in range(len(angles[pair][ot])):
        angle_dev_mp_w.append(abs(angles[pair][ot][i]-angles[pair][mp_w][i]))
except KeyError or IndexError:
    print("key error")
    pass


mp = mp.replace("_", " ")
mp = mp.replace(".csv", "")
mp = mp.replace("1", "")
mp_w = mp_w.replace("_", " ")
mp_w = mp_w.replace(".csv", "")
mp_w = mp_w.replace("1", "")
label_for_plots = [mp, mp_w]
angle_dev = [angle_dev_mp, angle_dev_mp_w]
# label_for_plots = [mp]
# angle_dev = [angle_dev_mp]
fig, ax = plt.subplots()
plt.title("Elbow angles in: " + pair)
bp_adatok = ax.boxplot(angle_dev, labels=label_for_plots, notch=True, showmeans=True)
plt.ylabel("Angles [°]")
plt.xlabel("Dataset")
f = "C:/dev/thesis/data/plots/BoxPlotd/angles_" + pair + ".svg"
plt.savefig(f, format='svg')
plt.show()
uniMes.functions.bp_data(bp_adatok, pair)

# xn = angles[pair][ot]

# from scipy.signal import butter, lfilter

# t = uniMes.time_resampled[pair][ot]

# def butter_bandpass(lowcut, highcut, fs, order=5):
#     nyq = 0.5 * fs
#     low = lowcut / nyq
#     high = highcut / nyq
#     b, a = butter(order, [low, high], btype='band')
#     return b, a


# def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
#     b, a = butter_bandpass(lowcut, highcut, fs, order=order)
#     y = lfilter(b, a, data)
#     return y

# from scipy import signal

# fs = 100
# lowcut = 4
# highcut = 20

# b, a = butter_bandpass(lowcut,highcut,fs)

# #b, a = signal.butter(3, [0.001, 0.5], btype='band')

# zi = signal.lfilter_zi(b, a)

# z, _ = signal.lfilter(b, a, xn, zi=zi*xn[0])
# z2, _ = signal.lfilter(b, a, z, zi=zi*z[0])
# y = signal.filtfilt(b, a, xn)
# plt.figure

# plt.plot(t, xn, 'b', alpha=0.75)

# plt.plot(t, z, 'r--', t, z2, 'r', t, y, 'k')

# plt.legend(('noisy signal', 'lfilter, once', 'lfilter, twice',

#             'filtfilt'), loc='best')

# plt.grid(True)

# plt.show()
# import numpy as np

# import matplotlib.pyplot as plt

# t = uniMes.time_resampled[pair][ot]

# sp = np.fft.fft(y)
# # for i in range(5):
# #     sp.pop(0)
# # for i in range(300):
# #     sp.pop()
# freq = np.fft.fftfreq(t.shape[-1])
# print(np.argmax(sp), np.argmin(sp))
# print(sp[np.argmin(sp)], freq[np.argmin(sp)])
# print(sp[np.argmax(sp)], freq[np.argmax(sp)])
# plt.plot(freq,sp.real)

# plt.show()