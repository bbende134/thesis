#Runs miltiple videos at once
import mediapipe as mp
#from matplotlib import pyplot as plt
#import os
#import numpy as np
import cv2

class Video:

    def __init__(self, x, y, z):
        self.x = [x]
        self.y = [y]
        self.z = [z]
   
    def add_element(self,  x, y, z):
        self.x.append(x)
        self.y.append(y)
        self.z.append(z)
        
    def display(self):
        print("First number = " + str(self.x))
        print("Second number = " + str(self.y))
        print("Addition of two numbers = " + str(self.z))   
        
    def get_x(self):
        return self.x
    def get_y(self):
        return self.y
    def get_z(self):
        return self.z

   
names = ['C:/Users/BAR7BP/Documents/Projects/Innohub/Video/szem.mp4', 'C:/Users/BAR7BP/Documents/Projects/Innohub/Video/old.mp4'];
window_titles = ['first', 'second']
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic

pose=mp_pose.Pose()

cap = [cv2.VideoCapture(i) for i in names]
#Creating new objects for each video

frames = [None] * len(names);
imgRGB = [None] * len(names);
ret = [None] * len(names);
members = []
for i in range(len(names)):
    asd.append(Vector(0,0,0))




while True:

    for i,c in enumerate(cap):
        if c is not None:
            ret[i], frames[i] = c.read();


    for i,f in enumerate(frames):
        if ret[i] == True:
            imgRGB[i] = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            results=pose.process(imgRGB[i])
            
            transformedMatrix = [None] * len(results.pose_world_landmarks.landmark)
            for j in range(len(results.pose_world_landmarks.landmark)):
                transformedMatrix[j] = Vector(
                    results.pose_world_landmarks.landmark[j].x, # x koordináták
                    results.pose_world_landmarks.landmark[j].y, # y koordináták
                    results.pose_world_landmarks.landmark[j].z # z koordináták
                )
            vid_vectors[i].append(transformedMatrix)
            
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(f, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.imshow(window_titles[i], imgRGB[i]);
            
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
for c in cap:
    if c is not None:
        c.release();

cv2.destroyAllWindows()