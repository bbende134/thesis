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

#%% Reading and Evaluating videos, and creating data from it

# def video_evaluation(vid, files, path):

#     mp_pose = mp.solutions.pose
#     pose=mp_pose.Pose()
    
#     for i in range(len(files)):
#         names = path + files[i]
#         print(names)
#         cap = cv2.VideoCapture(names)
#         while True:
#             success, img = cap.read()

#             if success == True:

#                 i_h, i_w, _  = img.shape

#                 imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#                 results=pose.process(imgRGB)

#                 if results.pose_world_landmarks:

#                     if vid[i] == None:
#                         vid[i] = objects.Video(results.pose_world_landmarks.landmark, files[i])
#                     else:
#                         vid[i].add_element(results.pose_world_landmarks.landmark)
                        
#             else:
#                 print('Cannot read video')
#                 break
            
#         print('Video evaluated: \n', files[i])
    
#     return vid

#%% Reading dataset from csv file

def read_dataset(file, path):
    file = path + file
    time_series = [None]
    collect = False
    start = False
    with open(file, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            if len(row) != 0:
                if row[1] == 'Name':
                    names = [None]*(len(row)-2)
                    for i in range(len(names)): 
                        names[i] = row[i+2]
                
                if row[0] == '0' or collect:
                    if start: # Here we collect the data to the existing arrays or objects
                        for i in range(int((len(row)-2)/3)):
                            if row[3*i+2] != '' and row[3*i+3] != '' and row[3*i+4] != '':
                                points[i]["x"].append(float(row[3*i+2]))
                                points[i]["y"].append(float(row[3*i+3]))
                                points[i]["z"].append(float(row[3*i+4]))
                                time = True
                            else:
                                time = False
                        if time: 
                            time_series.append(float(row[1]))
                    else: # Here we create the new inputs from csv file
                        points = {}
                        collect = True
                        for i in range(int((len(row)-2)/3)):
                            if row[3*i+2] != '' and row[3*i+3] != '' and row[3*i+4] != '':
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
