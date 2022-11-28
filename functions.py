# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 08:54:08 2022

@author: BAR7BP
"""
#%% Imports

import numpy as np
import mediapipe as mp
import cv2
import objects

#sfrom scripts import objects
import csv 
from os import walk

#%% Calculations

#testátló kiszámítása két pont között (térbeli távolságuk)
def calc_distance(coords, point1, point2):
    return np.sqrt((coords[point1].x-coords[point2].x)*(coords[point1].x-coords[point2].x)
        +(coords[point1].y-coords[point2].y)*(coords[point1].y-coords[point2].y)
        +(coords[point1].z-coords[point2].z)*(coords[point1].z-coords[point2].z))

def distance_2_point(x1, y1, z1, x2, y2, z2):
    return np.sqrt(pow((x1-x2),2)
                   +pow((y1-y2),2)
                   +pow((z1-z2),2))

#Deviation calculation between each frame
def calc_deviation(coords_earlier,coords_later,point):
    return np.sqrt(pow((coords_later[point].x-coords_earlier[point].x),2)
                   +pow((coords_later[point].y-coords_earlier[point].y),2)
                   +pow((coords_later[point].z-coords_earlier[point].z),2))

#Deviation between hips (0 point in mediapipe) and point
def calc_zero_deviation(coords, point = 30):
    return np.sqrt(pow((coords[point].x),2)
                   +pow((coords[point].y),2)
                   +pow((coords[point].z),2))

def unit_vector(vector): # unit vector from vector
    return vector / np.linalg.norm(vector)

# def calc_angle_2_vecs(points, coordinates):
#     vec1 = objects.Vector(coordinates[points[1]].x-coordinates[points[0]].x,
#            coordinates[points[1]].y-coordinates[points[0]].y,
#            coordinates[points[1]].z-coordinates[points[0]].z)
#     vec2 = objects.Vector(coordinates[points[3]].x-coordinates[points[2]].x,
#            coordinates[points[3]].y-coordinates[points[2]].y,
#            coordinates[points[3]].z-coordinates[points[2]].z)
#     return (np.rad2deg(angle_between(vec2.get_vals(),vec1.get_vals())))

def angle_between(v1, v2): # Angle between 2 3D vectors
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def num_derivation(dt, y0, y1, y2, order): #Numerical derivation with the given order
    if order == 1:
        return (y2-y0)/(2*dt)
    elif order == 2:
        return (y2-2*y1-y0)/pow(dt,2)
    else:
        print("Error: Unkown order")
        return 0
    
def dist_2_landmarks(landmarks, pairs):
    return np.sqrt(pow((landmarks[pairs[0]].x-landmarks[pairs[1]].x),2)+
                   pow((landmarks[pairs[0]].y-landmarks[pairs[1]].y),2))

def normalize(arr, t_min, t_max):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)    
    for i in arr:
        temp = (((i - min(arr))*diff)/diff_arr) + t_min
        norm_arr.append(temp)
    return norm_arr

#%% Reading dataset from csv file

def read_dataset(file, path):
    file = path + file
    time_series = [None]
    points = {}
    collect = False
    names_collected = False
    start = False
    with open(file, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            if len(row) != 0:
                if row[1] == 'Name':
                    names_collected = True
                    names = [None]*int((len(row)-2)/3)
                    for i in range(len(names)): 
                        names[i] = row[3*i+2]
                
                if row[0] == '0' or collect:
                    if start: # Here we collect the data to the existing arrays or objects
                        for i in range(int((len(row)-2)/3)):
                            if row[3*i+2] != '' and row[3*i+3] != '' and row[3*i+4] != '':
                                if names_collected:
                                    points[names[i]]["x"].append(float(row[3*i+2]))
                                    points[names[i]]["y"].append(float(row[3*i+3]))
                                    points[names[i]]["z"].append(float(row[3*i+4]))
                                else:
                                    points[i]["x"].append(float(row[3*i+2]))
                                    points[i]["y"].append(float(row[3*i+3]))
                                    points[i]["z"].append(float(row[3*i+4]))
                                time = True
                            else:
                                time = False
                        if time: 
                            time_series.append(float(row[1]))
                    else: # Here we create the new inputs from csv file
                        
                        collect = True
                        for i in range(int((len(row)-2)/3)):
                            if row[3*i+2] != '' and row[3*i+3] != '' and row[3*i+4] != '':
                                if names_collected:
                                    points[names[i]]={
                                    "x":[float(row[3*i+2])],
                                 "y":[float(row[3*i+3])],
                                  "z":[float(row[3*i+4])]
                                  }
                                else:
                                    points[i]={
                                        "x":[float(row[3*i+2])],
                                    "y":[float(row[3*i+3])],
                                    "z":[float(row[3*i+4])]
                                    }
                                time = True
                                start = True
                            else:
                                time = False
                        if time: 
                            time_series[0] = float(row[1])
    return points, time_series                  

def files_from_path(path, file_type):
    files = []
    for (dirpath, dirnames, filenames) in walk(path):
        break
    for i, f in enumerate(filenames):
        if f.endswith(file_type):
            files.append(f)
    
    return files

#%%
def distance_plotting(dataset, points_between, plotting, time=None):
    from matplotlib import pyplot as plt
    mes_dist = {}
    for record in dataset:
        dists = []
        temp_l = None
        temp_r = None
        for joints in dataset[record]:
            
            if joints == points_between[0]:
                temp_r = dataset[record][joints].copy()
                name_r = joints
            if joints == points_between[1]:
                temp_l = dataset[record][joints].copy()
                name_l = joints
            if joints == points_between[0]:
                temp_r = dataset[record][joints].copy()
                name_r = joints
            if joints == points_between[1]:
                temp_l = dataset[record][joints].copy()
                name_l = joints
        if temp_l and temp_r:
            for i in range(len(temp_l['x'])):
                
                dists.append(np.sqrt(pow((temp_l['x'][i]-temp_r['x'][i]),2)+
                pow((temp_l['y'][i]-temp_r['y'][i]),2)+
                pow((temp_l['z'][i]-temp_r['z'][i]),2)))
        if len(dists) > 1:
            mes_dist[record] = dists
            if plotting and time:
                leg = record
                title = record
                title = title.replace("1", "")
                title = title.replace(".csv", "")
                title = title.replace("mp_pose_world", "")
                title = title.replace("mp_pose", "")
                title = title.replace("ot", "")
                title = title.replace("_", "")
                leg = leg.replace(title,"")
                leg = leg.replace(":", "")
                leg = leg.replace("1", "")
                leg = leg.replace("_", " ")
                leg = leg.replace(".csv", "")
                leg = leg.replace("mp pose world", "MPW [m]")
                leg = leg.replace("mp pose", "MPP [1]")
                leg = leg.replace("ot", "OT [m]")
                leg = [leg]
                plt.plot(dists, '.-')
                
                title = "Exercise: " + title +  ", markers: " + str(name_r) + " and " + str(name_l)
                plt.title(title)
                plt.legend(leg)
                plt.ylabel("Distance between points")
                plt.xlabel("Sample [1]")
                title = title.replace(" ", "_")
                title = title.replace(":", "_")
                plt.savefig("C:/dev/thesis/data/plots/"+title+".svg")
                # plt.savefig("C:/dev/thesis/data/plots/"+title+".svg")
                plt.show()
            elif plotting:
                print("no time data")
                
    return mes_dist

# %% Distance plotting pairs

def distance_plotting_pair(dataset, points_between, plotting, time=None):
    from matplotlib import pyplot as plt
    mes_dist = {}
    for pair in dataset:
        mes_dist[pair] = {}
        for record in dataset[pair]:
            dists = []
            temp_l = None
            temp_r = None
            for joints in dataset[pair][record]:
                
                if joints == points_between[0]:
                    temp_r = dataset[pair][record][joints].copy()
                    name_r = joints
                if joints == points_between[1]:
                    temp_l = dataset[pair][record][joints].copy()
                    name_l = joints
                if joints == points_between[2]:
                    temp_r = dataset[pair][record][joints].copy()
                    name_r = joints
                if joints == points_between[3]:
                    temp_l = dataset[pair][record][joints].copy()
                    name_l = joints
            if temp_l and temp_r:
                for i in range(len(temp_l['x'])):
                    
                    dists.append(np.sqrt(pow((temp_l['x'][i]-temp_r['x'][i]),2)+
                    pow((temp_l['y'][i]-temp_r['y'][i]),2)+
                    pow((temp_l['z'][i]-temp_r['z'][i]),2)))
            if len(dists) > 1:
                mes_dist[pair][record] = dists
    for pair in mes_dist:
        temp_legend = []
        for record in mes_dist[pair]:
            leg = record
            leg = leg.replace(":", "")
            leg = leg.replace(pair, "")
            leg = leg.replace("_", " ")
            leg = leg.replace(".csv", "")
            leg = leg.replace("mp pose world", "MPW [m]")
            leg = leg.replace("mp pose", "MPP [1]")
            leg = leg.replace("ot", "OT [m]")
            temp_legend.append(leg)
            if plotting and time[pair]:
                # plot_data_y.append(mes_dist[pair][record])
                # plot_data_x.append(time[pair][record])
                plt.plot(time[pair][record], mes_dist[pair][record])
                #plt.plot(mes_dist[pair][record], '.-')
                
                temp_pair = pair.replace("_", " ")
                temp_pair = temp_pair.replace("1", "")
                title = "Exercise: " + temp_pair +  ", markers: " + str(name_r) + " and " + str(name_l)
                plt.title(title)
                plt.ylabel("Distance between points")
                plt.xlabel("Time [s]")
                title = title.replace(" ", "_")
                title = title.replace(":", "_")
                plt.savefig("C:/dev/thesis/data/plots/"+title+".svg")
                
            elif plotting:
                print("no time data")
        if plotting and time[pair]:
            plt.legend(temp_legend)
            plt.show()
                
    return mes_dist

#%% Syncing data 

def find_start_sync(dataset):
    start = {}
    for record in dataset:
        length = int(len(dataset[record])/2)
        temp = [None]*length
        for i in range(length):
            temp[i] = dataset[record][i]
        start[record] = min(range(len(temp)), key=temp.__getitem__)
    return start

def find_end_sync(dataset):
    end = {}
    for record in dataset:
        length = int(len(dataset[record])/2)
        temp = [None]*length
        for i in range(length):
            temp[i] = dataset[record][i+length]
        end[record] = length-min(range(len(temp)), key=temp.__getitem__)

    return end

#%% Pairing up data

def create_pairs(dataset, time):
    ot = 'ot_'
    pairs = {}
    time_pairs = {}
    exercises = []
    dataset_keys = dataset.keys()
    for names in dataset_keys:
        names = names.replace('.csv', '')
        if names.find(ot) > -1:
            exercises.append(names[len(ot):])
    for exercise in exercises:
        pairs[exercise] = {}
        time_pairs[exercise] = {}
        for record in dataset:
            if record.find(exercise) > -1:
                pairs[exercise][record] = dataset[record]
                time_pairs[exercise][record] = time[record]
    return pairs, time_pairs

#%% Manual cutting the same length

def manual_cut(dataset, time):
    import pointsCut
    for pair in dataset:
        for record in dataset[pair]:
            if record.find('mp_') > -1:
                for cut in range(pointsCut.cut_points[pair][record]):
                    for joint in dataset[pair][record]:
                        for coordinates in dataset[pair][record][joint]:
                            dataset[pair][record][joint][coordinates].pop()
                    time[pair][record].pop()
    return dataset, time

#%% Sync data

def mod_data(dataset, time, start_index, end_index):
    import matplotlib.pyplot as plt
    new_dataset = {}

    for start in start_index:
        new_dataset[start] = {}
        for joint in dataset[start]:
            new_dataset[start][joint] = {}
            # print("Hossz elő:" + str(len(dataset[start][joint]['z'])))
            for coordinates in dataset[start][joint]:
                if start.find("ot_star") > -1:
                    for i in range(start_index[start]+45):
                        dataset[start][joint][coordinates].pop(0)
                    for i in range(end_index[start]):
                        dataset[start][joint][coordinates].pop()
                elif start.find("ot_squat") > -1:
                    for i in range(start_index[start]+45):
                        dataset[start][joint][coordinates].pop(0)
                    for i in range(end_index[start]-15):
                        dataset[start][joint][coordinates].pop()
                elif start.find("ot_flies") > -1:
                    for i in range(start_index[start]+45):
                        dataset[start][joint][coordinates].pop(0)
                    for i in range(end_index[start]):
                        dataset[start][joint][coordinates].pop()
                else:
                    for i in range(start_index[start]):
                        dataset[start][joint][coordinates].pop(0)
                    for i in range(end_index[start]):
                        dataset[start][joint][coordinates].pop()
        # print("Hossz utó:" + str(len(dataset[start][joint][coordinates])))
        # print("hossz levág:" +str(start_index[start]+end_index[start]))

    new_time = {}
    for record in dataset:
        for i in range(start_index[record]):
            time[record].pop(0)
        for i in range(end_index[record]):
            time[record].pop()

                        
    return dataset, time

#%% Interpolation resampling

def resample_by_interpolation(signal, input_fs, output_fs):

    scale = output_fs / input_fs
    # calculate new length of sample
    n = round(len(signal) * scale)
    resampled_signal = np.interp(
        np.linspace(0.0, 1.0, n, endpoint=False),  # where to interpret
        np.linspace(0.0, 1.0, len(signal), endpoint=False),  # known positions
        signal,  # known data points
    )
    return resampled_signal

#%% Resample full dataset and time

def data_resample(dataset, time):
    resampled_dataset = {}
    resampled_time = {}
    resamp_data = {}
    for pair in dataset:
        for record in dataset[pair]:
            if record.find("mp_") >= 0:
                resamp_data[pair] = [len(dataset[pair][record][0]['x']),min(time[pair][record])]
    for pair in dataset:
        
        resampled_dataset[pair] = {}
        resampled_time[pair] = {}
        for record in dataset[pair]:
            if record.find("ot_") >= 0:
                rec_name = record.replace('.csv', '')
                rec_name = rec_name[3:]
                resampled_dataset[pair][record] = {}
                for joint in dataset[pair][record]:
                    resampled_dataset[pair][record][joint] = {}
                    for coordinates in dataset[pair][record][joint]:
                        resampled_dataset[pair][record][joint][coordinates] = resample_by_interpolation(
                            dataset[pair][record][joint][coordinates],
                            len(dataset[pair][record][joint][coordinates]),
                            resamp_data[pair][0])
                        
                x = time[pair][record]
                resampled_time[pair][record] = np.linspace(resamp_data[pair][1],
                x[-1],
                int(len(resampled_dataset[pair][record][joint][coordinates])),
                endpoint=False
                )
            else:
                resampled_dataset[pair][record] = dataset[pair][record]
                resampled_time[pair][record] = time[pair][record]
    return resampled_dataset, resampled_time

def data_resample_2(dataset, time):
    resampled_dataset = {}
    resampled_time = {}
    resamp_data = {}
    for pair in dataset:
        for record in dataset[pair]:
            if record.find("ot_") >= 0:
                resamp_data[pair] = [int(120*max(time[pair][record])),max(time[pair][record])]
    for pair in dataset:
        resampled_dataset[pair] = {}
        resampled_time[pair] = {}
        for record in dataset[pair]:
            resampled_dataset[pair][record] = {}
            for joint in dataset[pair][record]:
                resampled_dataset[pair][record][joint] = {}
                for coordinates in dataset[pair][record][joint]:
                    resampled_dataset[pair][record][joint][coordinates] = resample_by_interpolation(
                        dataset[pair][record][joint][coordinates],
                        len(dataset[pair][record][joint][coordinates]),
                        resamp_data[pair][0])
                    
            x = time[pair][record]
            resampled_time[pair][record] = np.linspace(0,
            resamp_data[pair][1],
            int(len(resampled_dataset[pair][record][joint][coordinates])),
            endpoint=False
            )

    return resampled_dataset, resampled_time

#%% Creating vector arrays

def vector_array(dataset, body_part_1, body_part_2):
    v_1 = {}
    v_2 = {}
    for pair in dataset:
        v_1[pair] = {}
        v_2[pair] = {}
        for record in dataset[pair]:
            v_1[pair][record] = []
            v_2[pair][record] = []
            if record.find('mp_') > -1:
                for i,coord in enumerate(dataset[pair][record][body_part_1[0]]['x']):
                    v_1[pair][record].append([None]*3)
                    v_1[pair][record][i][0] = dataset[pair][record][body_part_1[0]]['x'][i]- dataset[pair][record][body_part_1[1]]['x'][i]
                    v_1[pair][record][i][1] = dataset[pair][record][body_part_1[0]]['y'][i]- dataset[pair][record][body_part_1[1]]['y'][i]
                    v_1[pair][record][i][2] = dataset[pair][record][body_part_1[0]]['z'][i]- dataset[pair][record][body_part_1[1]]['z'][i]

                for i,coord in enumerate(dataset[pair][record][body_part_2[0]]['x']):
                    v_2[pair][record].append([None]*3)
                    v_2[pair][record][i][0] = dataset[pair][record][body_part_2[0]]['x'][i]- dataset[pair][record][body_part_2[1]]['x'][i]
                    v_2[pair][record][i][1] = dataset[pair][record][body_part_2[0]]['y'][i]- dataset[pair][record][body_part_2[1]]['y'][i]
                    v_2[pair][record][i][2] = dataset[pair][record][body_part_2[0]]['z'][i]- dataset[pair][record][body_part_2[1]]['z'][i]      
            if record.find('ot_') > -1:
                for i,coord in enumerate(dataset[pair][record][body_part_1[2]]['x']):
                    v_1[pair][record].append([None]*3)
                    v_1[pair][record][i][0] = dataset[pair][record][body_part_1[2]]['x'][i]- dataset[pair][record][body_part_1[3]]['x'][i]
                    v_1[pair][record][i][1] = dataset[pair][record][body_part_1[2]]['y'][i]- dataset[pair][record][body_part_1[3]]['y'][i]
                    v_1[pair][record][i][2] = dataset[pair][record][body_part_1[2]]['z'][i]- dataset[pair][record][body_part_1[3]]['z'][i]

                for i,coord in enumerate(dataset[pair][record][body_part_2[2]]['x']):
                    v_2[pair][record].append([None]*3)
                    v_2[pair][record][i][0] = dataset[pair][record][body_part_2[2]]['x'][i]- dataset[pair][record][body_part_2[3]]['x'][i]
                    v_2[pair][record][i][1] = dataset[pair][record][body_part_2[2]]['y'][i]- dataset[pair][record][body_part_2[3]]['y'][i]
                    v_2[pair][record][i][2] = dataset[pair][record][body_part_2[2]]['z'][i]- dataset[pair][record][body_part_2[3]]['z'][i]      
                        # d_x = data_points_resampled[pair][rec][p_1]['x'][100]-data_points_resampled[pair][rec][p_2]['x'][100]
                        # d_y = data_points_resampled[pair][rec][p_1]['y'][100]-data_points_resampled[pair][rec][p_2]['y'][100]
                        # d_z = data_points_resampled[pair][rec][p_1]['z'][100]-data_points_resampled[pair][rec][p_2]['z'][100]
    return v_1, v_2

#%% PLotting angles

def angle_plotting_pair(v_1, v_2,time_data,name, plotting):
    from matplotlib import pyplot as plt
    mes_angle = {}
    for pair in v_1:
        mes_angle[pair] = {}
        for record in v_1[pair]:
            mes_angle[pair][record] = []
            for i in range(len(v_1[pair][record])):
                mes_angle[pair][record].append(np.rad2deg(angle_between(v_1[pair][record][i],v_2[pair][record][i])))
    for pair in mes_angle:
        temp_legend = []
        for record in mes_angle[pair]:
            temp_legend.append(record)

                # plot_data_y.append(mes_dist[pair][record])
                # plot_data_x.append(time[pair][record])
                #plt.plot(time[pair][record], mes_dist[pair][record], '.-')
            plt.plot(time_data[pair][record],mes_angle[pair][record])
            plt.xlabel("Time [s]")
            plt.ylabel("Angles [°]")
            temp_pair = pair.replace("_", " ")
            temp_pair = temp_pair.replace("1", "")
            title = "Angles in exercise: " + temp_pair + "for: " + name
            plt.title(title)

            title = title.replace(" ", "_")
            title = title.replace(":", "_")
            # title = title.replace(" ", "_")
            # title = title.replace(":", "_")
            # plt.savefig("C:/dev/thesis/data/plots/"+title+".svg")

        for i,leg in enumerate(temp_legend):
            leg = leg.replace(pair, "")
            leg = leg.replace("_", " ")
            leg = leg.replace(".csv", "")
            leg = leg.replace("mp pose world", "MPW")
            leg = leg.replace("mp pose", "MPP")
            leg = leg.replace("ot", "OT")
            temp_legend[i] = leg
        plt.legend(temp_legend)

        f = "C:/dev/thesis/data/plots/BoxPlotd/angles/" + title + ".svg"
        plt.savefig(f, format='svg')
        if plotting:
            plt.show()
        else: 
            plt.clf()

                
    return mes_angle

#%% Bp data
# https://towardsdatascience.com/how-to-fetch-the-exact-values-from-a-boxplot-python-8b8a648fc813
def bp_data(bp, name):
    medians = [(round(item.get_ydata()[0] * 2, 3) / 2) for item in bp['medians']]
    means = [(round(item.get_ydata()[0] * 2, 3) / 2) for item in bp['means']]
    minimums = [(round(item.get_ydata()[0] * 2, 3) / 2) for item in bp['caps']][::2]
    maximums = [(round(item.get_ydata()[0] * 2, 3) / 2) for item in bp['caps']][1::2]
    q1 = [(round((min(item.get_ydata())*2), 3)/2) for item in bp['boxes']]
    q3 = [(round((max(item.get_ydata())*2), 3)/2)  for item in bp['boxes']]
    fliers = [item.get_ydata() for item in bp['fliers']]
    lower_outliers = []
    upper_outliers = []
    for i in range(len(fliers)):
        lower_outliers_by_box = []
        upper_outliers_by_box = []
        for outlier in fliers[i]:
            if outlier < q1[i]:
                lower_outliers_by_box.append((round(outlier*2, 3)/2))
            else:
                upper_outliers_by_box.append((round(outlier*2, 3)/2))
        lower_outliers.append(lower_outliers_by_box)
        upper_outliers.append(upper_outliers_by_box)    
        
    # New code
    stats = [medians, means, minimums, maximums, q1, q3, lower_outliers, upper_outliers]
    stats_names = ['Median', 'Mean', 'Minimum', 'Maximum', 'Q1', 'Q3', 'Lower outliers', 'Upper outliers'] # to be updated
    categories = [keys for keys in bp]
    file_name = "C:/dev/thesis/data/plots/Boxplotd/" + name
    with  open(file_name,'w') as f:
        print(f'\033[1m{name}\033[0m')
        f.write(name + ": \n")
        for j in range(len(stats)):
            print(f'{stats_names[j]}: {stats[j][i]}')
            f.write(f'{stats_names[j]}: {stats[j][i]} \n')
        f.write('\n')
        print('\n')
        f.close()

#%% 3D plotting
import objects
def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    '''Add an 3d arrow to an `Axes3D` instance.'''

    arrow = objects.Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)

#%% Landmarks array to csv

def landmark_to_csv(time_data, landmark, file):
    time_arr = [None]*len(time_data)

    for i,t in enumerate(time_data):
        if i > 0:
            time_arr[i] = time_arr[i-1] + t
        else:
            time_arr[i] = 0

    for i,marks in enumerate(landmark):
        marks.insert(0,time_arr[i])
        marks.insert(0,i)
    file.writerows(landmark)

#%% Box plotting functions

def time_plotting_for_pair(dataset, name=None):
    from matplotlib import pyplot as plt
    for pair in dataset:
        #fig, ax = plt.subplots()
        label_for_plots = []
        plot_data = []
        # ax.set_title(pair)
        for record in dataset[pair]:
            plot_data.append(dataset[pair][record].copy())
            record = record.replace("_"," ")
            record = record.replace(".csv","")
            label_for_plots.append(record)

        plt.plot(plot_data[0], '.-',plot_data[1], '.-', plot_data[2], '.-' )
        plt.legend(label_for_plots)
        plt.title("Standard deviation of " + name + " in " + pair)

        plt.ylabel("Deviations in OT [m], Pose [-], Pose World [m]")
        plt.xlabel("Sample []")
        plt.show()

def box_plotting_for_pair(dataset, name=None):
    from matplotlib import pyplot as plt
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches
    for pair in dataset:
        #fig, ax = plt.subplots()
        label_for_plots = []
        plot_data = []
        # ax.set_title(pair)
        for record in dataset[pair]:
            plot_data.append(dataset[pair][record].copy())
            leg = record
            leg = leg.replace(":", "")
            leg = leg.replace(pair, "")
            leg = leg.replace("_", " ")
            leg = leg.replace(".csv", "")
            leg = leg.replace("mp pose world", "MPW [m]")
            leg = leg.replace("mp pose", "MPP [1]")
            leg = leg.replace("ot", "OT [m]")
            label_for_plots.append(leg)
        
        green_triangle = mlines.Line2D([], [], color='green', marker='^', linestyle='None',
                          markersize=10, label='Average')
        orange_line = mlines.Line2D([], [], color='orange',
                           label='Median')
        plt.legend(handles=[green_triangle, orange_line])
        plt.boxplot(plot_data, labels=label_for_plots, notch=True, showmeans=True)
        pair = pair.replace("_1","")
        title = "Boxplot representation of " + name + " in " + pair
        plt.title("Boxplot representation of " + name + " in " + pair)
        plt.ylabel("Standard deviations")
        plt.xlabel("Dataset")
        title = title.replace(" ","_")
        plt.savefig("C:/dev/thesis/data/plots/Boxplotd/rigidbodi/"+title+".svg")
        plt.clf()
        # plt.show()

def box_plotting_for_pair_2(dataset, name=None):
    from matplotlib import pyplot as plt
    import matplotlib.lines as mlines
    tmp_dataset = {}
    label_for_plots = [None]*6
    names_for_plot = ["MediaPipe Pose World","MediaPipe Pose","OptiTrack"]
    for i,pair in enumerate(dataset):
        #fig, ax = plt.subplots()
        # ax.set_title(pair)
        for record in dataset[pair]:

            if record.find('ot_') >= 0:
                try:
                    tmp_dataset["OptiTrack"].append(dataset[pair][record].copy())
                except KeyError:
                    tmp_dataset["OptiTrack"] = [dataset[pair][record].copy()]
            elif record.find('pose_world_') >= 0:
                try:
                    tmp_dataset["MediaPipe Pose World"].append(dataset[pair][record].copy())
                except KeyError:
                    tmp_dataset["MediaPipe Pose World"] = [dataset[pair][record].copy()]
            else:
                try:
                    tmp_dataset["MediaPipe Pose"].append(dataset[pair][record].copy())
                except KeyError:
                    tmp_dataset["MediaPipe Pose"] = [dataset[pair][record].copy()]
                    
        leg = pair
        leg = leg.replace("_1", "")
        
        label_for_plots[i] = leg

    for i,data in enumerate(tmp_dataset):
        green_triangle = mlines.Line2D([], [], color='green', marker='^', linestyle='None',
                            markersize=10, label='Average')
        orange_line = mlines.Line2D([], [], color='orange',
                            label='Median')
        plt.legend(handles=[green_triangle, orange_line])
        plt.boxplot(tmp_dataset[data], labels=label_for_plots, notch=True, showmeans=True)
        
        title = "Boxplot representation of " + name + " of " + data
        plt.title(title)
        plt.ylabel("Standard deviations")
        plt.xlabel("Dataset")
        title = title.replace(" ","_")
        plt.savefig("C:/dev/thesis/data/plots/Boxplotd/rigidbodi/"+title+".svg")
        plt.clf()
        # plt.show()

def box_plotting_for_all(dataset, title_name = None):
    from matplotlib import pyplot as plt
    plot_data = {}
    label_for_plots = ["MediaPipe Pose World","MediaPipe Pose","OptiTrack"]
    print(label_for_plots[2])
    for pair in dataset:
        
        for record in dataset[pair]:
            if record.find('ot_') >= 0:
                try:
                    for dists in dataset[pair][record]:
                        plot_data["OptiTrack"].append(dists.copy())
                except KeyError:
                    plot_data["OptiTrack"] = dataset[pair][record].copy()
            elif record.find('pose_world_') >= 0:
                try:    
                    for dists in dataset[pair][record]:
                        plot_data["MediaPipe Pose World"].append(dists.copy())
                except KeyError:
                    plot_data["MediaPipe Pose World"] = dataset[pair][record].copy()
            else:
                try:
                    for dists in dataset[pair][record]:
                        plot_data["MediaPipe Pose"].append(dists.copy())
                except KeyError:
                    plot_data["MediaPipe Pose"] = dataset[pair][record].copy()
                    print(label_for_plots[2])
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
        ax.set_title("Szórási boxplot")
    bp = ax.boxplot(temp_data, labels=list(plot_data.keys()), notch=True, showmeans=True)
    plt.ylabel("Length OT [m], Pose [-], Pose World [m]")
    plt.xlabel("Datasets")

    f = "C:/dev/thesis/data/plots/BoxPlotd/" + title_name + ".svg"
    plt.savefig(f, format='svg')
    plt.show()
    medians = [(round(item.get_ydata()[0] * 2, 3) / 2) for item in bp['medians']]
    means = [(round(item.get_ydata()[0] * 2, 3) / 2) for item in bp['means']]
    minimums = [(round(item.get_ydata()[0] * 2, 3) / 2) for item in bp['caps']][::2]
    maximums = [(round(item.get_ydata()[0] * 2, 3) / 2) for item in bp['caps']][1::2]
    q1 = [(round((min(item.get_ydata())*2), 3)/2) for item in bp['boxes']]
    q3 = [(round((max(item.get_ydata())*2), 3)/2)  for item in bp['boxes']]
    fliers = [item.get_ydata() for item in bp['fliers']]
    lower_outliers = []
    upper_outliers = []
    for i in range(len(fliers)):
        lower_outliers_by_box = []
        upper_outliers_by_box = []
        for outlier in fliers[i]:
            if outlier < q1[i]:
                lower_outliers_by_box.append((round(outlier*2, 3)/2))
            else:
                upper_outliers_by_box.append((round(outlier*2, 3)/2))
        lower_outliers.append(lower_outliers_by_box)
        upper_outliers.append(upper_outliers_by_box)  
        
    # New code
    stats = [medians, means, minimums, maximums, q1, q3, lower_outliers, upper_outliers]
    stats_names = ['Median', 'Mean', 'Minimum', 'Maximum', 'Q1', 'Q3', 'Lower outliers', 'Upper outliers']
# to be updated
    file_name = "C:/dev/thesis/data/plots/Boxplotd/" + title_name
    with  open(file_name,'w') as f:
        for i,name in enumerate(label_for_plots):
            print(f'\033[1m{name}\033[0m')
            f.write(name + ": \n")
            for j in range(len(stats)):
                print(f'{stats_names[j]}: {stats[j][i]}')
                f.write(f'{stats_names[j]}: {stats[j][i]} \n')
            f.write('\n')
            print('\n')
        f.close()

    return bp_data

def box_plotting_for_exercise(dataset, title_name = None):
    from matplotlib import pyplot as plt
    plot_data = {}
    label_for_plots = ["OptiTrack","MediaPipe Pose","MediaPipe Pose World"]
    for pair in dataset:
        for record in dataset[pair]:
            plot_data[pair] = {}
            if record.find('ot_') > -1:
                try:
                    plot_data[label_for_plots[0]][record] = dataset[pair][record].copy()
                except KeyError:
                    plot_data[label_for_plots[0]] = {}
                    plot_data[label_for_plots[0]][record] = dataset[pair][record].copy()
            elif record.find('pose_world_') > -1:
                try:    
                    plot_data[label_for_plots[2]][record] = dataset[pair][record].copy()
                except KeyError:
                    plot_data[label_for_plots[2]] = {}
                    plot_data[label_for_plots[2]][record] = dataset[pair][record].copy()
            else:
                try:
                    plot_data[label_for_plots[1]][record] = dataset[pair][record].copy()
                except KeyError:
                    plot_data[label_for_plots[1]] = {}
                    plot_data[label_for_plots[1]][record] = dataset[pair][record].copy()
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
        ax.set_title("Szórási boxplot")
    bp = ax.boxplot(temp_data, labels=label_for_plots, notch=True, showmeans=True)
    plt.ylabel("Length")
    plt.xlabel("Datasets")

    f = "C:/dev/thesis/data/plots/BoxPlotd/" + title_name + ".svg"
    plt.savefig(f, format='svg')
    plt.show()
    medians = [(round(item.get_ydata()[0] * 2, 3) / 2) for item in bp['medians']]
    means = [(round(item.get_ydata()[0] * 2, 3) / 2) for item in bp['means']]
    minimums = [(round(item.get_ydata()[0] * 2, 3) / 2) for item in bp['caps']][::2]
    maximums = [(round(item.get_ydata()[0] * 2, 3) / 2) for item in bp['caps']][1::2]
    q1 = [(round((min(item.get_ydata())*2), 3)/2) for item in bp['boxes']]
    q3 = [(round((max(item.get_ydata())*2), 3)/2)  for item in bp['boxes']]
    fliers = [item.get_ydata() for item in bp['fliers']]
    lower_outliers = []
    upper_outliers = []
    for i in range(len(fliers)):
        lower_outliers_by_box = []
        upper_outliers_by_box = []
        for outlier in fliers[i]:
            if outlier < q1[i]:
                lower_outliers_by_box.append((round(outlier*2, 3)/2))
            else:
                upper_outliers_by_box.append((round(outlier*2, 3)/2))
        lower_outliers.append(lower_outliers_by_box)
        upper_outliers.append(upper_outliers_by_box)  
        
    # New code
    stats = [medians, means, minimums, maximums, q1, q3, lower_outliers, upper_outliers]
    stats_names = ['Median', 'Mean', 'Minimum', 'Maximum', 'Q1', 'Q3', 'Lower outliers', 'Upper outliers']
# to be updated
    file_name = "C:/dev/thesis/data/plots/Boxplotd/" + title_name
    with  open(file_name,'w') as f:
        for i,name in enumerate(label_for_plots):
            print(f'\033[1m{name}\033[0m')
            f.write(name + ": \n")
            for j in range(len(stats)):
                print(f'{stats_names[j]}: {stats[j][i]}')
                f.write(f'{stats_names[j]}: {stats[j][i]} \n')
            f.write('\n')
            print('\n')
        f.close()

    return bp_data   
#%% landmark to array

def land_to_arr(landmark):
    mark_arr = []
    for i,m in enumerate(landmark):
        mark_arr.append(m.x)
        mark_arr.append(m.y)
        mark_arr.append(m.z)
    return mark_arr

#%% find max values in a array of vector arrays for normalize it

def find_max(vect_arr):
    temp = [None]*3
    for i in range(3):
        temp[i] = max(vect_arr.get_vals()[i])
    return max(temp)
        
def norm_vecs(arrOfVectors, max_num, is_int):
    maxes = [None]*len(arrOfVectors)
    for i in range(len(arrOfVectors)):
        maxes[i] = find_max(arrOfVectors[i])   
    for i in range(len(arrOfVectors)):
        arrOfVectors[i].mod_vals(max_num/float(maxes[i]), is_int)
        
#%% Creating movement image
def create_image(data, save_path, name):
    from PIL import Image
    image = np.zeros((len(data),len(data[0].get_vals()[0]),3), dtype = np.uint8)
    for i in range(len(data)):
        for j in range(len(data[i].get_vals()[0])):
            for k in range(len(data[i].get_vals())):
                image[i][j][k] = data[i].get_vals()[k][j]
    im = Image.fromarray(image)
    save_path = save_path + name + ".bmp"
    im.save(save_path)

#%% Data filter

from scipy.signal import butter, lfilter

def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

#%% Image manipulation

# writing
def write_on_image(image, text, place, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    thickness = 2
    lineType = 2

    cv2.putText(image, text,
                place,
                font,
                fontScale,
                color,
                thickness,
                lineType)
    
# arrow
def draw_arrow(img, landmarks, start_point, end_point, color):
    start_points = int(landmarks[start_point].x), int(landmarks[start_point].y)
    end_points =int(landmarks[end_point].x), int(landmarks[end_point].y)
    cv2.arrowedLine(img, start_points, end_points, color, thickness = 1)

#%% Dictionary for record person's metrics

#%% Elapsed time

def create_elapsed_time(time):
    first_iter = True
    for i,t in enumerate(time):
        if first_iter:
            t_earlier = t
            time[i] = 0
            first_iter = False
        else:
            t_later = t
            time[i] = time[i-1] + t_later-t_earlier
            t_earlier = t
    return time
