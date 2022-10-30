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

#%% Box plotting of rigid bodies

rigid_bodies = {"left hand forearm":[15,13, "Bende:l_wrist","Bende:l_elbow"],
"right hand forearm":[16,14, "Bende:r_wrist","Bende:r_elbow"],
"left hand upper arm":[11,13, "Bende:l_elbow","Bende:l_shoulder"],
"right hand upper arm":[12,14, "Bende:r_elbow","Bende:r_shoulder"],
"shoulder width":[12,11, "Bende:l_shoulder","Bende:r_shoulder"],
"hip width":[24,23, "Bende:l_hip","Bende:r_hip"],
"left thigh length":[23,25, "Bende:l_hip","Bende:l_knee"],
"right thigh length":[24,26, "Bende:r_hip","Bende:r_knee"],
"left lower leg length":[25,27, "Bende:l_knee","Bende:l_ankle"],
"right lower leg length":[26,28, "Bende:r_knee","Bende:r_ankle"],
"left foot length":[29,31, "Bende:l_heel","Bende:l_toe"],
"right foot length":[30,32, "Bende:r_heel","Bende:r_toe"],
}

# l_dist_mp_hands = functions.distance_plotting_pair(data_points_resampled, [15,13, "Bende:l_wrist","Bende:l_elbow"], False, time_resampled)
# r_dist_mp_hands = functions.distance_plotting_pair(data_points_resampled, [16,14, "Bende:r_wrist","Bende:r_elbow"], False, time_resampled)


# statistic_data = {}
# for name in rigid_bodies:
#     lengths = functions.distance_plotting_pair(data_points_resampled, rigid_bodies[name], False, time_resampled)
#     statistic_data[name] = functions.box_plotting_for_all(lengths, name)






#%%

from mpl_toolkits.mplot3d.axes3d import Axes3D

setattr(Axes3D, 'arrow3D', functions._arrow3D)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(0,2)


pair = 'kartarogatas_1'
rec = "mp_pose_world_kartarogatas_1.csv"
p_1 = 29
p_2 = 31

d_x = data_points_resampled[pair][rec][p_1]['x'][100]-data_points_resampled[pair][rec][p_2]['x'][100]
d_y = data_points_resampled[pair][rec][p_1]['y'][100]-data_points_resampled[pair][rec][p_2]['y'][100]
d_z = data_points_resampled[pair][rec][p_1]['z'][100]-data_points_resampled[pair][rec][p_2]['z'][100]

# ax.arrow3D(data_points_resampled[pair][rec][p_1]['x'][100],data_points_resampled[rec][p_1]['z'][100],(-1)*data_points_resampled[rec][p_1]['y'][100],
#             data_points_resampled[pair][rec][p_2]['x'][100],data_points_resampled[rec][p_2]['z'][100],(-1)*data_points_resampled[rec][p_2]['y'][100],
           
#            mutation_scale=20,
#            fc='red')
ax.arrow3D(data_points_resampled[pair][rec][p_2]['x'][100],data_points_resampled[pair][rec][p_2]['z'][100],(-1)*data_points_resampled[pair][rec][p_2]['y'][100],
           d_x,d_z,(-1)*d_y,
           mutation_scale=20,
           fc='red')
ax.set_title('3D virtual skeleton')
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_zlabel('z [m]')

import bodyPlot

bodyPlot.plot_world_landmarks(ax,data_points_resampled[pair][rec],100, True)

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
