# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 08:24:57 2022

@author: BAR7BP
"""
#%%
import mediapipe as mp
import numpy as np
import cv2
import time
import functions
import matplotlib.pyplot as plt
import scipy.fftpack.basic as fourier
from scipy.signal import freqz
# %%

# Write some Text
COLOR_R = (0,0,255)
font = cv2.FONT_HERSHEY_SIMPLEX
upperLeftCornerOfText = (10, 35)
fontScale = 1
fontColor = (0, 255, 255)
thickness = 2
lineType = 2


# Video file-ból határozza meg és jeleníti meg a landmark adatokat. Pl. Botond videók!!

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_drawing_styles = mp.solutions.drawing_styles

pose = mp_pose.Pose()
# 'C:/Users/BAR7BP/Documents/Projects/Innohub/Video/Videok/Meresi_videok/telo_45.mp4'
cap = cv2.VideoCapture(0)
start = True
d_t = 1
x = []
dx = []
ddx = []
i = 0

fs = 500
lowcut = 5
highcut = 50
b, a = functions.butter_bandpass(lowcut, highcut, fs)
w, h = freqz(b, a, fs=fs, worN=2000)

while True:

    t = time.time()

    success, img = cap.read()
    i_h, i_w, _ = img.shape
    if success:
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result = pose.process(imgRGB)

        
        if result.pose_landmarks:
            mp_drawing.draw_landmarks(
                img, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            x.append(result.pose_landmarks.landmark[16].x)

            if i > 2:
                if i > 100:
                    i = 100

                dx.append(functions.num_derivation(d_t, x[i-2], 0, x[i], 1))
                ddx.append(functions.num_derivation(d_t, x[i-2], x[i-1], x[i], 2))

                
                dx = dx[-100:]
                ddx = ddx[-100:]

                # Filter a noisy signal.


                x_filtered = functions.butter_bandpass_filter(x, lowcut, highcut, fs, order=4)


        
                functions.draw_arrow(imgRGB, result.pose_landmarks.landmark, 15, 11, COLOR_R)
                xf = fourier.fft(x)

                plt.ylim(min(x), max(x))

                plt.plot(x)
                plt.show()
                # plt.scatter(3, y=max(x), color='black')
                # plt.scatter(3, x[i-3], color='red')
                # plt.scatter(3, min(x), color='black')

            # mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            i += 1

        fps = f"{d_t:.3f}s {(1/d_t):.3f}fps"

        img = cv2.resize(img, (960, 540))

        functions.write_on_image(img, fps, (10,35), (0, 255, 255))
        #cv2.imshow("Image", img)
        t_new = time.time()
        d_t = t_new - t

        if cv2.waitKey(5) & 0xFF == 27:  # ESC kilép
            break
    else:
        print("couldn't load video")

cap.release()
cv2.destroyAllWindows()

# %%
