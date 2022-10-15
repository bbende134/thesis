#%% Rep count
import matplotlib.pyplot as plt
import numpy as np
import functions
import time
import mediapipe as mp
import cv2
from datetime import datetime
import objects

COLOR_B = (255,0,0)
COLOR_G = (0,255,0)
COLOR_R = (0,0,255)
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic
mp_drawing_styles = mp.solutions.drawing_styles
pose=mp_pose.Pose()

x_values = []
y_values = []
dists = []

x = 0
y = 0

fig = plt.figure()

#angles = []

i = 0
cap = cv2.VideoCapture(0)
start = False

get_name = False
name = ""
letters = 0  

start_reps = False
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

rep_count = 0
start = False
line, = plt.plot(x,y,'o')

with mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.8) as pose:
    while cap.isOpened():
        t = time.time()
        success, img = cap.read()
        i = i + 1
        if success == True:
            
           # img = cv2.resize(img, (540, 960))
            # img = cv2.resize(img, (960, 540))
            i_h, i_w, _  = img.shape
            
            fig.canvas.draw()
            
            
            
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
            results=pose.process(imgRGB)
    
            if results.pose_world_landmarks:
                
                mp_drawing.draw_landmarks(imgRGB, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                x_values.append(results.pose_landmarks.landmark[16].x)
                # y_values.append(datetime.now())
                # x = results.pose_landmarks.landmark[16].x
                # y = results.pose_landmarks.landmark[16].y
                
                # start_dist = np.sqrt(pow((results.pose_landmarks.landmark[19].x-results.pose_landmarks.landmark[20].x),2)+
                #                pow((results.pose_landmarks.landmark[19].y-results.pose_landmarks.landmark[20].y),2))
                
                if results.pose_landmarks.landmark[16].visibility > 0.8:
                    functions.write_on_image(imgRGB, 'Visible', (10,70), (0, 255, 0))
                #     if start_dist < 0.02 or start:
                #         dist = np.sqrt(pow((x-results.pose_landmarks.landmark[12].x),2)+pow((y-results.pose_landmarks.landmark[12].y),2))
                #         dists.append(dist)
                #         start = True
                #         functions.write_on_image(imgRGB, str(rep_count), (10,105), (0, 255, 255))
                #         if (0.8*min(dists)) < dist < (2*min(dists)):
                #             functions.write_on_image(imgRGB, str(dist), (int(i_w*x),int(i_h*y)),(0, 255, 255))
                #             up = True
                #         if (0.7*max(dists)) < dist < (1.1*max(dists)):
                #             functions.write_on_image(imgRGB, str(dist), (int(i_w*x),int(i_h*y)),(0, 0, 255))
                #             down = True
                #             if up and down:
                #                 up = False
                #                 rep_count += 1
                else:
                    functions.write_on_image(imgRGB, 'Not Visible', (10,70), (0, 0, 255))
                    
                
                
                
                #functions.write_on_image(imgRGB, str(results.pose_landmarks.landmark[16].visibility), (10,70))
                
                x_values = x_values[-100:]
                # y_values = y_values[-100:]
                dists = dists[-100:]
                if i > 100:
                    i = 100
                plt.xlim(1, -1)
                # plt.ylim(1, -1)
    
                line.set_xdata(x_values)
                # line.set_ydata(y)
                plt.plot(x_values)
                
                img_plot = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,
                sep='')
                img_plot = img_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                
                img_plot = cv2.cvtColor(img_plot,cv2.COLOR_RGB2BGR)
                
                cv2.imshow("plot",img_plot)
                
                
            t_new = time.time()
            d_t = t_new - t
            
            fps = f"{d_t:.3f}s {(1/d_t):.3f}fps"
            functions.write_on_image(imgRGB, fps, (10,35),(0, 255, 255))
            
            #cv2.imshow("Image", imgRGB)
            if cv2.waitKey(5) & 0xFF ==27: # ESC kilép
                break      
        else:
            break
    
    
    

cap.release()
cv2.destroyAllWindows()
# %%
