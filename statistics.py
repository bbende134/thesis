import uniMes
import scipy.stats as stats
import numpy as np

def box_plotting_for_all(dataset, title_name = None):
    from matplotlib import pyplot as plt
    plot_data = {}
    label_for_plots = ["OptiTrack","MediaPipe Pose","MediaPipe Pose World"]
    for pair in dataset:
        
        for record in dataset[pair]:
            if record.find('ot_') > -1:
                try:
                    for dists in dataset[pair][record]:
                        plot_data[label_for_plots[0]].append(dists.copy())
                except KeyError:
                    plot_data[label_for_plots[0]] = dataset[pair][record].copy()
            elif record.find('pose_world_') > -1:
                try:    
                    for dists in dataset[pair][record]:
                        plot_data[label_for_plots[1]].append(dists.copy())
                except KeyError:
                    plot_data[label_for_plots[1]] = dataset[pair][record].copy()
            else:
                try:
                    for dists in dataset[pair][record]:
                        plot_data[label_for_plots[2]].append(dists.copy())
                except KeyError:
                    plot_data[label_for_plots[2]] = dataset[pair][record].copy()
    temp_data = []
    for mts in plot_data:
        temp_data.append(plot_data[mts])
    fig, ax = plt.subplots()
    if title_name != None:
        title_name = "Boxplot: " + title_name
        ax.set_title(title_name)
        title_name = title_name.replace(" ", "_")
        title_name = title_name.replace(":", "_")
    else:
        ax.set_title("BoxPLot")
    bp = ax.boxplot(temp_data, labels=label_for_plots, notch=True, showmeans=True)
    plt.ylabel("Merevtest méretek OT [m], PW [m], P[-]")
    plt.xlabel("Adatsorok")
    f = "C:/dev/thesis/data/plots/BoxPlotd/" + title_name + ".svg"
    plt.savefig(f, format='svg')
    plt.show()
    return temp_data


rigid_bodies = {"bal alkar":[15,13, "Bende:l_wrist","Bende:l_elbow"],
"jobb alkar":[16,14, "Bende:r_wrist","Bende:r_elbow"],
"bal felkar":[11,13, "Bende:l_elbow","Bende:l_shoulder"],
"jobb felkar":[12,14, "Bende:r_elbow","Bende:r_shoulder"],
"váll":[12,11, "Bende:l_shoulder","Bende:r_shoulder"],
"csípő":[24,23, "Bende:l_hip","Bende:r_hip"],
"bal comb":[23,25, "Bende:l_hip","Bende:l_knee"],
"jobb comb":[24,26, "Bende:r_hip","Bende:r_knee"],
"bal lábszár":[25,27, "Bende:l_knee","Bende:l_ankle"],
"jobb lábszár":[26,28, "Bende:r_knee","Bende:r_ankle"],
"bal lábfej":[29,31, "Bende:l_heel","Bende:l_toe"],
"jobb lábfej":[30,32, "Bende:r_heel","Bende:r_toe"],
}

# l_dist_mp_hands = functions.distance_plotting_pair(data_points_resampled, [15,13, "Bende:l_wrist","Bende:l_elbow"], False, time_resampled)
#r_dist_mp_hands = uniMes.functions.distance_plotting_pair(uniMes.data_points_resampled, rigid_bodies['left hand forearm'], False, uniMes.time_resampled)


statistic_data = {}
temp = []
first = True
for name in rigid_bodies:
    lengths = uniMes.functions.distance_plotting_pair(uniMes.data_points_resampled, rigid_bodies[name], False, uniMes.time_resampled)
    stat_data = box_plotting_for_all(lengths, name)
    temp.append(stat_data[1].copy())
    print(np.var(stat_data[1]), np.var(stat_data[2]))
    print(stats.ttest_ind(a=stat_data[1], b=stat_data[0], equal_var=False))

for i in range(len(temp)):
    print(np.var(temp[0]), np.var(temp[1]))
    print(stats.ttest_ind(a=temp[2], b=temp[3], equal_var=True))
    