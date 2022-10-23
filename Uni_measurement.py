# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 09:58:39 2022

@author: BAR7BP
"""
#%% find
import re
import objects
import functions
from matplotlib import pyplot as plt
import numpy as np

path = 'C:/dev/thesis/data/'

files = functions.files_from_path(path, '.csv')

data_points = {}
time_data = {}

for i in range(len(files)):
    data_points[files[i]], time_data[files[i]] = functions.read_dataset(files[i], path)

#%% Plotting above data

# for datas in data_points:
#     print(datas)

    
#     print((data_points[datas][0]['x']))


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

dist_mp_hands = functions.distance_plotting(data_points, [16,15], False)
dist_ot_hands = functions.distance_plotting(data_points, ["Bende:l_wrist","Bende:r_wrist"], False)

dist_hands = Merge(dist_mp_hands, dist_ot_hands)


#%% Data sync

start_sync_datasample = functions.find_start_sync(dist_hands)

data_points_synced, time_synced = functions.mod_data(data_points,time_data, start_sync_datasample)

# %% Resampling phase 

x = time_data['ot_csillag_1.csv']
y = data_points_synced['ot_csillag_1.csv']['Bende:l_wrist']['y']

data_points_resampled, time_resampled = functions.data_resample(data_points_synced, time_synced)

f = functions.resample_by_interpolation(y,len(data_points_synced['ot_csillag_1.csv']['Bende:l_wrist']['y']),int(25/120*len(data_points_synced['ot_csillag_1.csv']['Bende:l_wrist']['y'])))
#f = signal.resample(y,int(25/120*len(data_points_synced['ot_csillag_1.csv']['Bende:l_wrist']['y'])))
xnew = np.linspace(0, x[-1], int(25/120*len(data_points_synced['ot_csillag_1.csv']['Bende:l_wrist']['y'])), endpoint=False)

# x = np.linspace(0, 10, len(data_points_synced['mp_pose_world_csillag_1.csv'][15]['y']), endpoint=False)
# y = data_points_synced['mp_pose_world_csillag_1.csv'][15]['y']

plt.plot(x, y, '.-',xnew, f, '.-')
plt.legend(['MediaPipe', 'OptiTrack'], loc='best')
# %%

# dist_mp_hands = functions.distance_plotting(data_points_resampled, [24,28], True, time_resampled)
# dist_ot_hands = functions.distance_plotting(data_points_resampled, ["Bende:l_hip","Bende:l_ankle"], True, time_resampled)

# %%
dist_mp_hands = functions.distance_plotting(data_points_resampled, [16,14], False, time_resampled)
dist_ot_hands = functions.distance_plotting(data_points_resampled, ["Bende:l_wrist","Bende:l_elbow"], False, time_resampled)

#%%

import matplotlib.colors as mcolors


by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(color))),
                    name)
                for name, color in mcolors.TABLEAU_COLORS.items())
names = [name for hsv, name in by_hsv]
print(by_hsv)

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
ax.arrow3D(1,0,0,
           1,1,1,
           mutation_scale=20,
           fc='red')
ax.set_title('3D Arrows Demo')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

import bodyPlot

bodyPlot.plot_world_landmarks(ax,data_points_resampled['ot_csillag_1.csv'],150, False)

fig.tight_layout()
plt.show()

