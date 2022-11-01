import uniMes
import matplotlib.pyplot as plt


vec_set_1, vec_set_2 = uniMes.functions.vector_array(uniMes.data_points_resampled, [13,15,"Bende:l_elbow","Bende:l_wrist"],[13,11,"Bende:l_elbow","Bende:l_shoulder"])

angles = uniMes.functions.angle_plotting_pair(vec_set_1,vec_set_2,True)

pair = 'karhajlitas_1'

# angle_dev = []
# for i in range(len(angles[pair]['ot_karhajlitas_1.csv'])):
#     angle_dev.append(abs(angles[pair]['ot_karhajlitas_1.csv'][i]-angles[pair]['mp_pose_world_karhajlitas_1.csv'][i]))
angle_dev = []
for i in range(len(angles[pair]['ot_karhajlitas_1.csv'])):
    angle_dev.append(abs(angles[pair]['ot_karhajlitas_1.csv'][i]-angles[pair]['mp_pose_world_karhajlitas_1.csv'][i]))

label_for_plots = [pair]
fig, ax = plt.subplots()
plt.title("Bezárt szögek: " + pair)
bp_adatok = ax.boxplot(angle_dev, labels=label_for_plots, notch=True, showmeans=True)
plt.ylabel("Bezárt szögek [°]")
plt.xlabel("Adatsor")
plt.show()
uniMes.functions.bp_data(bp_adatok, pair)