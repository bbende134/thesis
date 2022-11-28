import uniMes
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

def legending (leg):
    leg = leg.replace(":", "")
    leg = leg.replace(pair, "")
    leg = leg.replace("_", " ")
    leg = leg.replace(".csv", "")
    leg = leg.replace("mp pose world", "MPW")
    leg = leg.replace("mp pose", "MPP")
    leg = leg.replace("ot", "OT [m]")
    return leg

angles_between = {"right elbow":[[16,14,"Bende:r_wrist","Bende:r_elbow"],[12,14,"Bende:r_shoulder","Bende:r_elbow"]],
"upper arms":[[14,12,"Bende:r_elbow","Bende:r_shoulder"],[11,13,"Bende:l_shoulder","Bende:l_elbow"]],
"right knee":[[28,26,"Bende:r_ankle","Bende:r_knee"],[24,26,"Bende:r_hip","Bende:r_knee"]],
"right leg and torso":[[26,24,"Bende:r_knee","Bende:r_hip"],[12,24,"Bende:r_shoulder","Bende:r_hip"]]}

for pair in uniMes.data_points_resampled:
    for parts in angles_between:
        vec_set_1, vec_set_2 = uniMes.functions.vector_array(uniMes.data_points_resampled, angles_between[parts][0],angles_between[parts][1])
        angles = uniMes.functions.angle_plotting_pair(vec_set_1,vec_set_2,uniMes.time_resampled,parts,False)


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


        label_for_plots = [legending(mp), legending(mp_w)]
        angle_dev = [angle_dev_mp, angle_dev_mp_w]
        temp_pair = pair.replace("_1", "")
        fig, ax = plt.subplots()
        plt.title("Absolute angle deviations in: " + temp_pair + " for: " + parts)
        green_triangle = mlines.Line2D([], [], color='green', marker='^', linestyle='None',
                            markersize=10, label='Average')
        orange_line = mlines.Line2D([], [], color='orange',
                            label='Median')
        plt.legend(handles=[green_triangle, orange_line])
        bp_adatok = ax.boxplot(angle_dev, labels=label_for_plots, notch=True, showmeans=True)
        plt.ylabel("Angle deviations compared to OT [°]")
        plt.xlabel("Dataset")
        f = "C:/dev/thesis/data/plots/BoxPlotd/angles/" + temp_pair + "_" + parts + ".svg"
        plt.savefig(f, format='svg')
        plt.clf()

        uniMes.functions.bp_data(bp_adatok, temp_pair)

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