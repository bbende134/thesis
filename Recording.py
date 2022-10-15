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


pairs = [[12,16],[11,15],[14,13],[14,24],[13,23], [15,16]]
arrays = [r_hand_shoulder, l_hand_shoulder, elbows, r_hip_elbow, l_hip_elbow, hands]
name_array = ["r_hand_shoulder","l_hand_shoulder","elbows","r_hip_elbow","l_hip_elbow", "hands"]


rep_time = []
j = 0
k = 0
rep_count = 0
with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
    #pers = objects.Person()
    rec_on = False
    rec = False
    moves = []
    t_data = []
    fps_arr = []
    while cap.isOpened():
        t = time.time_ns()/ (10 ** 9)
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
                
                # landmarks = results.pose_world_landmarks.landmark
                landmarks = results.pose_landmarks.landmark
                
                
                    
                    
                if landmarks[16].visibility >= 0.7 and landmarks[0].visibility >= 0.7:
                    functions.write_on_image(img, 'Visible', (10,70), (0, 255, 0))
                    start_dist = functions.dist_2_landmarks(landmarks, [20,19])

                    if (start_dist < 0.1) and rec_on:
                        rec = False
                    elif (start_dist < 0.1):
                        rec = True
                    if rec:
                        
                        """ dist_r = np.sqrt(pow((landmarks[16].x-landmarks[12].x),2)+
                                           pow((landmarks[16].y-landmarks[12].y),2))
                        dist_l = np.sqrt(pow((landmarks[15].x-landmarks[11].x),2)+
                                        pow((landmarks[15].y-landmarks[11].y),2))
                        
                        if start_rep_rec: k += 1 # only starts to record reps, if max length was measured
                        else: k = 0
                        # creating rep lists
                        
                        for i,array in enumerate(arrays):
                            array.append(functions.dist_2_landmarks(landmarks, pairs[i]))
                            arrays[i] = arrays[i][-k:]
                        

                        rep_time.append(time.time())
                        # hand_shoulder = hand_shoulder[-k:]
                        rep_time = rep_time[-k:]
                        
                        dists = pers.get_dists()
                        start = True
                        functions.write_on_image(img, "Reps: " + str(rep_count), (10,105), (0, 255, 255))
                        x_r = landmarks[16].x
                        y_r = landmarks[16].y
                        x_l = landmarks[15].x
                        y_l = landmarks[15].y
                        
                        # Drawings
                        if ((0.8*dists[0]) < dist_r < 1.5*dists[0]):
                            functions.write_on_image(img, "up", (int(i_w*x_r),int(i_h*y_r)),COLOR_G)
                        if ((0.8*dists[1]) < dist_r < 1.5*dists[1]):
                            functions.write_on_image(img, "up", (int(i_w*x_l),int(i_h*y_l)),COLOR_G)
                        if (0.8*dists[2]) < dist_r < (1.1*dists[2]):
                            functions.write_on_image(img, "down", (int(i_w*x_r),int(i_h*y_r)),COLOR_G)
                        if (0.8*dists[3]) < dist_r < (1.1*dists[3]):
                            functions.write_on_image(img, "down", (int(i_w*x_l),int(i_h*y_l)),COLOR_G)
                        
                        # dist_corr.correction(img, pairs, landmarks)
                        
                        # Logic
                        if ((0.8*dists[0]) < dist_r < 1.5*dists[1]) and ((0.8*dists[1]) < dist_l < 1.5*dists[1]):
                            # checking if hands are reaching the minimum area
                            up = True
                            
                        if ((0.8*dists[2]) < dist_r < (1.1*dists[2])) and ((0.8*dists[3]) < dist_l < (1.1*dists[3])):
                            # checking if hands are reaching the maximum area
                            down = True
                            start_rep_rec = True
                            
                            if up and down:
                                # if both area are reached, a new rep is being recorded
                                up = False
                                down = False
                                rep_count += 1
                                pers.reps(name_array,
                                            arrays,
                                            rep_time)
                                k = 0       """              
                        moves.append(functions.land_to_arr(landmarks).copy())
                        t_new = time.time_ns()/ (10 ** 9)
                        d_t = t_new - t
                        while (1/d_t) >= SET_FPS*1.02:
                            t_new = time.time_ns()/ (10 ** 9)
                            d_t = t_new - t
                        
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

with open("C:/dev/thesis/data/" + date_act + ".csv", "w", newline="") as f:
    writer = csv.writer(f)
    functions.landmark_to_csv(t_data,moves,writer)

# %%



# %%
