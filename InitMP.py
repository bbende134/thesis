
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import time
import functions
import numpy as np
import dict 


def initMP():
    cap = cv2.VideoCapture(-1)
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    mp_holistic = mp.solutions.holistic
    mp_drawing_styles = mp.solutions.drawing_styles
    pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) 
    return cap, pose, mp_drawing, mp_pose, cv2, np



    
