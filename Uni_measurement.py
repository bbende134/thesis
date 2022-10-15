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

points = {}
time_data = {}

for i in range(len(files)):
    points[files[i]], time_data[files[i]] = functions.read_dataset(files[i], path)


# for dataset in points:
#     functions.norm_vecs(dataset, 1, False)
    
# for i,new_pics in enumerate(points):
#     functions.create_image(new_pics,
#                        path,
#                        files[i])

#%% Plotting above data

for datas in points:
    print(datas)

    
    print((points[datas][0]['x']))




# #%% Evaluate videos using mediapipe and collect landmarks


# %%
