# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 07:59:05 2022

@author: BAR7BP
"""
#%% main
import matplotlib.pyplot as plt
import numpy as np
import functions
import time
import mediapipe as mp
import cv2
import objects

COLOR_B = (255,0,0)
COLOR_G = (0,255,0)
COLOR_R = (0,0,255)
SET_FPS = 25

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic
mp_drawing_styles = mp.solutions.drawing_styles
pose=mp_pose.Pose()

x_values = []
y_values = []
dists = []

fig = plt.figure()

#angles = []

i = 0
cap = cv2.VideoCapture(0)
start = False 

up = False

max_length = False
min_length = False
left_hand_length = []
right_hand_length = []

hand_shoulder = []
r_hand_shoulder = []
l_hand_shoulder = []
elbows = []
r_hip_elbow = []
l_hip_elbow = []
hands = []


rep_time = []
j = 0
k = 0
rep_count = 0
inp = input()
with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
    #pers = objects.Person()
    rec_on = False
    rec = False
    moves_pose = []
    moves_world = []
    t_data = []
    fps_arr = []
    while cap.isOpened():
        t = time.time_ns()
        success, img = cap.read()
        i = i + 1
        if success == True:
            
            # img = cv2.resize(img, (960, 960))
            # img = cv2.resize(img, (960, 540))
            i_h, i_w, _  = img.shape
            
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
            results=pose.process(imgRGB)
    
            if results.pose_world_landmarks:
                
                mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                landmarks = results.pose_landmarks.landmark
                if landmarks[16].visibility >= 0.7 and landmarks[0].visibility >= 0.7:
                    functions.write_on_image(img, 'Visible', (10,70), (0, 255, 0))
                    start_dist = functions.dist_2_landmarks(landmarks, [20,19])

                    if (start_dist < 0.1) and rec_on:
                        rec = False
                    elif (start_dist < 0.1):
                        rec = True
                    if rec:        
                        moves_pose.append(functions.land_to_arr(results.pose_landmarks.landmark).copy())
                        moves_world.append(functions.land_to_arr(results.pose_world_landmarks.landmark).copy())
                        t_new = time.time_ns()
                        d_t = (t_new - t)/ (10 ** 9)
                        while (1/d_t) >= SET_FPS*1.02:
                            t_new = time.time_ns()
                            d_t = (t_new - t)/ (10 ** 9)
                        
                        fps = f"{(1/d_t):.1f}fps"
                        fps_arr.append(1/d_t)
                        functions.write_on_image(img, fps, (10,35),(0, 255, 255))
                        t_data.append(d_t)
                        
                
                else:
                    functions.write_on_image(img, 'Not Visible'+str(results.pose_landmarks.landmark[16].visibility), (10,70), (0, 0, 255))
                    
                
                
            
            
            cv2.imshow("Fitty", img)
            if cv2.waitKey(5) & 0xFF ==27: # ESC kilép
                break      
        else:
            break
    
    
    
    
if cap.isOpened():
    cap.release()
    cv2.destroyAllWindows()

# %%csv export

import csv
import datetime
date_act = datetime.datetime.now()
date_act = date_act.strftime("%c") 
date_act = date_act.replace(" ", "_")
date_act = date_act.replace(":", "_")

with open("C:/dev/thesis/data/mp_pose_"+ inp + date_act + ".csv", "w", newline="") as f:
    writer = csv.writer(f)
    functions.landmark_to_csv(t_data,moves_pose,writer)
with open("C:/dev/thesis/data/mp_pose_world_"+ inp  + date_act + ".csv", "w", newline="") as f:
    writer = csv.writer(f)
    functions.landmark_to_csv(t_data,moves_world,writer)

# %%



# %%
