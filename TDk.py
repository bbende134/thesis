# For webcam input:
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 21:12:41 2022

@author: Admin
"""

import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import time
import functions
import numpy as np
import dict 

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic
mp_drawing_styles = mp.solutions.drawing_styles
pose=mp_pose.Pose()

z = []
cap = cv2.VideoCapture(0)
with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
        play = True
        while play: 
            play, landmarks = dict.frameEval(cap, pose, mp_drawing, mp_pose, cv2, np)

    # pers = objects.Person()
    # while cap.isOpened():
    #     t = time.time()
    #     success, img = cap.read()
    #     # i = i + 1
    #     if success == True:
            
    #         # img = cv2.resize(img, (960, 960))
    #         # img = cv2.resize(img, (960, 540))
    #         i_h, i_w, _  = img.shape
            
    #         imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    #         results=pose.process(imgRGB)
    
    #         if results.pose_world_landmarks:
                
    #             mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                
    #             # landmarks = results.pose_world_landmarks.landmark
    #             landmarks = results.pose_landmarks.landmark
                
    #             start_dist = np.sqrt(pow((results.pose_landmarks.landmark[19].x-results.pose_landmarks.landmark[20].x),2)+
    #                            pow((results.pose_landmarks.landmark[19].y-results.pose_landmarks.landmark[20].y),2))
                    
                    
    #             if landmarks[16].visibility >= 0.7 and landmarks[0].visibility >= 0.7:
    #                 functions.write_on_image(img, 'Visible', (10,70), (0, 255, 0))
                
    #             else:
    #                 functions.write_on_image(img, 'Not Visible'+str(results.pose_landmarks.landmark[16].visibility), (10,70), (0, 0, 255))
                    
                
                
    #         t_new = time.time()
    #         d_t = t_new - t
    #         while (1/d_t) < 25:
    #             t_new = time.time()
    #             d_t = t_new - t
            
    #         fps = f"{(1/d_t):.1f}fps"
    #         functions.write_on_image(img, fps, (10,35),(0, 255, 255))
            
    #         cv2.imshow("Fitty", img)
    #         if cv2.waitKey(5) & 0xFF ==27: # ESC kilép
    #             break      
    #     else:
    #         break
    
    
    
    
if cap.isOpened():
    cap.release()
    cv2.destroyAllWindows()
plt.plot(z)
plt.show()