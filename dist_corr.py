# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 18:31:43 2022

@author: BAR7BP
"""
import cv2
import functions
#%% distance correction 

COLOR_B = (255,0,0)
COLOR_G = (0,255,0)
COLOR_R = (0,0,255)


hand_dist_min = 0.3
hand_dist_max = 0.55

elbows_dist_min = 0.25
elbows_dist_max = 0.36


def correction(img, pairs, landmarks):
    i_h, i_w, _  = img.shape
    hands = 5
    if functions.dist_2_landmarks(landmarks, pairs[hands]) > hand_dist_max:
        cv2.arrowedLine(img,
                        (int(i_w*landmarks[pairs[hands][0]].x),int(i_h*landmarks[pairs[hands][0]].y)) ,
                        (int(i_w*landmarks[pairs[hands][0]].x)-25,int(i_h*landmarks[pairs[hands][0]].y)),
                        COLOR_R, 3, 8, 0, 0.1)
        cv2.arrowedLine(img,
                        (int(i_w*landmarks[pairs[hands][1]].x),int(i_h*landmarks[pairs[hands][1]].y)) ,
                        (int(i_w*landmarks[pairs[hands][1]].x)+25,int(i_h*landmarks[pairs[hands][1]].y)),
                        COLOR_R, 3, 8, 0, 0.1)
    if functions.dist_2_landmarks(landmarks, pairs[hands]) < hand_dist_min:
        cv2.arrowedLine(img,
                        (int(i_w*landmarks[pairs[hands][0]].x),int(i_h*landmarks[pairs[hands][0]].y)) ,
                        (int(i_w*landmarks[pairs[hands][0]].x)+25,int(i_h*landmarks[pairs[hands][0]].y)),
                        COLOR_R, 3, 8, 0, 0.1)
        cv2.arrowedLine(img,
                        (int(i_w*landmarks[pairs[hands][1]].x),int(i_h*landmarks[pairs[hands][1]].y)) ,
                        (int(i_w*landmarks[pairs[hands][1]].x)-25,int(i_h*landmarks[pairs[hands][1]].y)),
                        COLOR_R, 3, 8, 0, 0.1)
    elbows = 2
    if functions.dist_2_landmarks(landmarks, pairs[elbows]) > elbows_dist_max:
        cv2.arrowedLine(img,
                        (int(i_w*landmarks[pairs[elbows][0]].x),int(i_h*landmarks[pairs[elbows][0]].y)) ,
                        (int(i_w*landmarks[pairs[elbows][0]].x)-25,int(i_h*landmarks[pairs[elbows][0]].y)),
                        COLOR_R, 3, 8, 0, 0.1)
        cv2.arrowedLine(img,
                        (int(i_w*landmarks[pairs[elbows][1]].x),int(i_h*landmarks[pairs[elbows][1]].y)) ,
                        (int(i_w*landmarks[pairs[elbows][1]].x)+25,int(i_h*landmarks[pairs[elbows][1]].y)),
                        COLOR_R, 3, 8, 0, 0.1)
    if functions.dist_2_landmarks(landmarks, pairs[elbows]) < elbows_dist_min:
        cv2.arrowedLine(img,
                        (int(i_w*landmarks[pairs[elbows][0]].x),int(i_h*landmarks[pairs[elbows][0]].y)) ,
                        (int(i_w*landmarks[pairs[elbows][0]].x)+25,int(i_h*landmarks[pairs[elbows][0]].y)),
                        COLOR_R, 3, 8, 0, 0.1)
        cv2.arrowedLine(img,
                        (int(i_w*landmarks[pairs[elbows][1]].x),int(i_h*landmarks[pairs[elbows][1]].y)) ,
                        (int(i_w*landmarks[pairs[elbows][1]].x)-25,int(i_h*landmarks[pairs[elbows][1]].y)),
                        COLOR_R, 3, 8, 0, 0.1)