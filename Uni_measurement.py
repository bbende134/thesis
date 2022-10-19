# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 09:58:39 2022

@author: BAR7BP
"""
#%% find
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

data_points_synced, time_synced = functions.mod_data(data_points,start_sync_datasample)

# %% Resampling phase 

import numpy as np
import matplotlib.pyplot as plt

def data_resample(dataset, time):
    for record in dataset:
        if record.find('ot_'):
            for joint in dataset[record]:
                for coordinates in dataset[record][joint]:
                    f = functions.resample_by_interpolation(y,
                        len(dataset[record][joint][coordinates]),
                        int(25/120*len(dataset[record][joint][coordinates])))


x = time_data['ot_csillag_1.csv']
y = data_points_synced['ot_csillag_1.csv']['Bende:l_wrist']['y']

f = functions.resample_by_interpolation(y,len(data_points_synced['ot_csillag_1.csv']['Bende:l_wrist']['y']),int(25/120*len(data_points_synced['ot_csillag_1.csv']['Bende:l_wrist']['y'])))
#f = signal.resample(y,int(25/120*len(data_points_synced['ot_csillag_1.csv']['Bende:l_wrist']['y'])))
xnew = np.linspace(0, x[-1], int(25/120*len(data_points_synced['ot_csillag_1.csv']['Bende:l_wrist']['y'])), endpoint=False)

# x = np.linspace(0, 10, len(data_points_synced['mp_pose_world_csillag_1.csv'][15]['y']), endpoint=False)
# y = data_points_synced['mp_pose_world_csillag_1.csv'][15]['y']

plt.plot(x, y, '.-',xnew, f, '.-', 10, y[0], 'ro')
plt.legend(['MediaPipe', 'OptiTrack'], loc='best')
plt.show()
# %%
# %%
