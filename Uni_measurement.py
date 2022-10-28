# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 09:58:39 2022

@author: BAR7BP
"""
#%% find

import functions
from matplotlib import pyplot as plt
import numpy as np

path = 'C:/dev/thesis/data/'

files = functions.files_from_path(path, '.csv')

data_points = {}
time_data = {}

for i in range(len(files)):
    data_points[files[i]], time_data[files[i]] = functions.read_dataset(files[i], path)


# %%
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation


# x_data = []
# y_data = []

# fig, ax = plt.subplots()
# ax.set_xlim(0, max(time_data['Wed_Oct_12_16_06_18_2022_pose_csillag.csv']))
# ax.set_ylim(min(data_points['Wed_Oct_12_16_06_18_2022_pose_csillag.csv'][24]['y']),
#  max(data_points['Wed_Oct_12_16_06_18_2022_pose_csillag.csv'][24]['y']))
# line, = ax.plot(0, 0)

# def animation_frame(i):
	
# 	x_data.append(time_data['Wed_Oct_12_16_06_18_2022_pose_csillag.csv'][i])
# 	print(data_points['Wed_Oct_12_16_06_18_2022_pose_csillag.csv'][24]['y'][i])
# 	y_data.append(data_points['Wed_Oct_12_16_06_18_2022_pose_csillag.csv'][24]['y'][i])

# 	line.set_xdata(x_data)
# 	line.set_ydata(y_data)
# 	return line, 

# animation = FuncAnimation(fig,
#  func=animation_frame,
#   frames=len(data_points['Wed_Oct_12_16_06_18_2022_pose_csillag.csv'][24]['x']),
#    interval=((1/25)*1000))
# plt.show()
#%% Create new data for syncing

def Merge(dict1, dict2):
    res = {**dict1, **dict2}
    return res

dist_mp_hands = functions.distance_plotting(data_points, [16,15], False, time_data)
dist_ot_hands = functions.distance_plotting(data_points, ["Bende:l_wrist","Bende:r_wrist"], False, time_data)

dist_hands = Merge(dist_mp_hands, dist_ot_hands)


#%% Data sync

start_sync_datasample = functions.find_start_sync(dist_hands)

data_points_synced, time_synced = functions.mod_data(data_points,time_data, start_sync_datasample)

#%% Pairing data
paired_data_points, paired_time = functions.create_pairs(data_points_synced,time_synced)

#%% Cutting for same length

cutted_data_points, time_cutted = functions.manual_cut(paired_data_points,paired_time)

# %% Resampling phase 

data_points_resampled, time_resampled = functions.data_resample(cutted_data_points, time_cutted)

# %%

# dist_mp_hands = functions.distance_plotting(data_points_resampled, [24,28], True, time_resampled)
# dist_ot_hands = functions.distance_plotting(data_points_resampled, ["Bende:l_hip","Bende:l_ankle"], True, time_resampled)

#%% Getting distances for hands (should be the same length)
l_dist_mp_hands = functions.distance_plotting_pair(data_points_resampled, [15,13, "Bende:l_wrist","Bende:l_elbow"], False, time_resampled)
r_dist_mp_hands = functions.distance_plotting_pair(data_points_resampled, [16,14, "Bende:r_wrist","Bende:r_elbow"], False, time_resampled)


def box_plotting(dataset):
    for pair in dataset:
        fig, ax = plt.subplots()
        label_for_plots = []
        plot_data = []
        ax.set_title(pair)
        for record in dataset[pair]:
            label_for_plots.append(record)
            plot_data.append(dataset[pair][record].copy())
        ax.boxplot(plot_data, labels=label_for_plots, notch=True)
        plt.ylabel("Kéz hosszának szórása, átlaga")
        plt.xlabel("Adatsorok")
        plt.show()

# box_plotting(l_dist_mp_hands)




#%%

from mpl_toolkits.mplot3d.axes3d import Axes3D

setattr(Axes3D, 'arrow3D', functions._arrow3D)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(0,2)
ax.arrow3D(0,0,0,
           1,1,1,
           mutation_scale=20,
           arrowstyle="-|>",
           linestyle='dashed')

ax.arrow3D(data_points_resampled['csillag_1']['ot_csillag_1.csv']["Bende:r_wrist"]['x'][100],data_points_resampled['ot_csillag_1.csv']["Bende:r_wrist"]['z'][100],(-1)*data_points_resampled['ot_csillag_1.csv']["Bende:r_wrist"]['y'][100],
            data_points_resampled['csillag_1']['ot_csillag_1.csv']["Bende:l_wrist"]['x'][100],data_points_resampled['ot_csillag_1.csv']["Bende:l_wrist"]['z'][100],(-1)*data_points_resampled['ot_csillag_1.csv']["Bende:l_wrist"]['y'][100],
           
           mutation_scale=20,
           fc='red')
ax.arrow3D(data_points_resampled['csillag_1']['ot_csillag_1.csv']["Bende:l_wrist"]['x'][100],data_points_resampled['ot_csillag_1.csv']["Bende:l_wrist"]['z'][100],(-1)*data_points_resampled['ot_csillag_1.csv']["Bende:l_wrist"]['y'][100],
           data_points_resampled['csillag_1']['ot_csillag_1.csv']["Bende:r_wrist"]['x'][100],data_points_resampled['ot_csillag_1.csv']["Bende:r_wrist"]['z'][100],(-1)*data_points_resampled['ot_csillag_1.csv']["Bende:r_wrist"]['y'][100],
           mutation_scale=20,
           fc='red')
ax.set_title('3D Arrows Demo')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

import bodyPlot

bodyPlot.plot_world_landmarks(ax,data_points_resampled['ot_csillag_1.csv'],100, False)

fig.tight_layout()
plt.show()

#%% 
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis

dist_mp_hands = functions.distance_plotting(data_points_resampled, [16,15], False, time_resampled)
dist_ot_hands = functions.distance_plotting(data_points_resampled, ["Bende:l_wrist","Bende:r_wrist"], False, time_resampled)


s2 = list(dist_mp_hands['csillag_1']['mp_pose_world_csillag_1.csv'])
s1 = list(dist_ot_hands['csillag_1']['ot_csillag_1.csv'])

# if len(s1) > len(s2):
#     while len(s1) > len(s2):
#         s1.pop()
# elif len(s2) > len(s1):
#     while len(s2) > len(s1):
#         s2.pop()
for i in range(60):
    s1.pop()

for i in range(20):
    s1.pop(0)
    s2.pop(0)

path = dtw.warping_path(s1, s2)
dtwvis.plot_warping(s1, s2, path, filename="warp.png")
d, paths = dtw.warping_paths(s1, s2, window=100, psi=2)
best_path = dtw.best_path(paths)
dtwvis.plot_warpingpaths(s1, s2, paths, best_path)
# %%

# %%
