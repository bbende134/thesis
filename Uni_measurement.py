# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 09:58:39 2022

@author: BAR7BP
"""
#%% find
import objects
import functions
from matplotlib import pyplot as plt

path = 'C:/dev/thesis/data/'

files = functions.files_from_path(path, '.csv')

data_points = {}
time_data = {}

for i in range(len(files)):
    data_points[files[i]], time_data[files[i]] = functions.read_dataset(files[i], path)


# for dataset in points:
#     functions.norm_vecs(dataset, 1, False)
    
# for i,new_pics in enumerate(points):
#     functions.create_image(new_pics,
#                        path,
#                        files[i])

#%% Plotting above data

for datas in data_points:
    print(datas)

    
    print((data_points[datas][0]['x']))


# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


x_data = []
y_data = []

fig, ax = plt.subplots()
ax.set_xlim(0, 200)
ax.set_ylim(0, 30)
line, = ax.plot(0, 0)

def animation_frame(i):

	x_data.append(i ** 2)
	y_data.append(i)

	line.set_xdata(x_data)
	line.set_ydata(y_data)
	return line, 

animation = FuncAnimation(fig, func=animation_frame, frames=len(), interval=10)
plt.show()