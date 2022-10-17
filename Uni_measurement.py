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
#%% Create new data for plotting and analyze

dist_mp_hands = functions.distance_plotting(data_points, [16,15])
dist_ot_hands = functions.distance_plotting(data_points, ["Bende:l_wrist","Bende:r_wrist"])




# %%
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

x = np.linspace(0, 10, len(data_points['ot_csillag_2.csv']['Bende:l_wrist']['y']), endpoint=False)
y = data_points['ot_csillag_2.csv']['Bende:l_wrist']['y']

f = signal.resample(y,
len(data_points['mp_pose_world_csillag.csv'][15]['y']))
xnew = np.linspace(0, 10, len(data_points['mp_pose_world_csillag.csv'][15]['y']), endpoint=False)

x = np.linspace(0, 10, len(data_points['mp_pose_world_csillag.csv'][15]['y']), endpoint=False)
y = data_points['mp_pose_world_csillag.csv'][15]['y']

plt.plot(x, y, '.-', xnew, f, '.-', 10, y[0], 'ro')
plt.legend(['MediaPipe', 'OptiTrack'], loc='best')
plt.show()
# %%
# %%
