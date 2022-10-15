
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


name = ""
letters = 0  

get_name = False
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
k = 0
rep_count = 0
with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
    pers = objects.Person()
    while cap.isOpened():
        t = time.time()
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
                
                start_dist = np.sqrt(pow((results.pose_landmarks.landmark[19].x-results.pose_landmarks.landmark[20].x),2)+
                               pow((results.pose_landmarks.landmark[19].y-results.pose_landmarks.landmark[20].y),2))
                
                
                # if not get_name:
                #     new_letter = cv2.waitKey(0)
                #     cv2.rectangle(img, (0,0), (i_w,i_h), (0,0,0), -1)
                #     if new_letter > 0 and chr(new_letter).isalpha():
                #         letters += 10
                #         name += chr(new_letter)
                        
                #     functions.write_on_image(img, name, (int(i_w*0.2),int(i_h/2)+35),(255, 255, 255))
                #     functions.write_on_image(img, "Write your name: ", (int(i_w*0.2),int(i_h/2)),(255, 255, 255))
                #     if new_letter == 13:
                #         get_name = True
                #         pers.get_name(name)
                #     elif new_letter == 27:
                #         break
                    
                    
                if landmarks[16].visibility >= 0.7 and landmarks[0].visibility >= 0.7:
                    functions.write_on_image(img, 'Visible', (10,70), (0, 255, 0))
                    
                    if start_reps == False:
                        if max_length: # measouring the lengths of a person
                            functions.write_on_image(img, "Arms' max length is measured", (10,105), COLOR_G)
                            if min_length:
                                functions.write_on_image(img, "Arms' min length is measured", (10,140), COLOR_G)
                                start_reps = True
                                k=0
                            else:
                                functions.write_on_image(img, "Bend your arms", (10,140), COLOR_R)
                        else:
                            functions.write_on_image(img, "Stretch your arms", (10,105), COLOR_R)
                        if not start and j == 0:
                            # exercise starts with a clapping
                            functions.write_on_image(img, "Clap your hands", (int(i_w/2),int(i_h/2)),(0, 255, 255))
                        if start_dist < 0.08 or start:
                            j += 1
                            cv2.arrowedLine(img,
                                            (int(i_w*landmarks[12].x),int(i_h*landmarks[12].y)) ,
                                            (int(i_w*landmarks[16].x),int(i_h*landmarks[16].y)),
                                            COLOR_R, 3, 8, 0, 0.1)
                            cv2.arrowedLine(img,
                                            (int(i_w*landmarks[11].x),int(i_h*landmarks[11].y)) ,
                                            (int(i_w*landmarks[15].x),int(i_h*landmarks[15].y)),
                                            COLOR_R, 3, 8, 0, 0.1)
                            start = True
                            right_hand_length.append(np.sqrt(pow((results.pose_landmarks.landmark[16].x-results.pose_landmarks.landmark[12].x),2)+
                                           pow((results.pose_landmarks.landmark[16].y-results.pose_landmarks.landmark[12].y),2)))
                            left_hand_length.append(np.sqrt(pow((results.pose_landmarks.landmark[15].x-results.pose_landmarks.landmark[11].x),2)+
                                           pow((results.pose_landmarks.landmark[15].y-results.pose_landmarks.landmark[11].y),2)))
                            if j > 60 and max(right_hand_length) > 0.1 and max(left_hand_length) > 0.1:
                                pers.max_length(max(left_hand_length),max(right_hand_length))
                                start = False
                                max_length = True
                        elif j > 60 and max_length == True and min_length == False: 
                            k += 1
                            
                            cv2.arrowedLine(img, 
                                            (int(i_w*landmarks[16].x),int(i_h*landmarks[16].y)) ,
                                            (int(i_w*landmarks[12].x),int(i_h*landmarks[12].y)),
                                            COLOR_R, 3, 8, 0, 0.1)
                            cv2.arrowedLine(img,
                                            (int(i_w*landmarks[15].x),int(i_h*landmarks[15].y)) ,
                                            (int(i_w*landmarks[11].x),int(i_h*landmarks[11].y)),
                                            COLOR_R, 3, 8, 0, 0.1)
                            
                            right_hand_length.append(np.sqrt(pow((results.pose_landmarks.landmark[16].x-results.pose_landmarks.landmark[12].x),2)+
                                           pow((results.pose_landmarks.landmark[16].y-results.pose_landmarks.landmark[12].y),2)))
                            left_hand_length.append(np.sqrt(pow((results.pose_landmarks.landmark[15].x-results.pose_landmarks.landmark[11].x),2)+
                                           pow((results.pose_landmarks.landmark[15].y-results.pose_landmarks.landmark[11].y),2)))
                            if k > 60 and min(right_hand_length) < 0.1 and min(left_hand_length) < 0.1 and min(left_hand_length) != 0:
                                pers.min_length(min(left_hand_length),min(right_hand_length))
                                start = False
                                min_length = True
                    elif start_reps:
                        if not start:
                            functions.write_on_image(img, "Clap your hands", (int(i_w/2),int(i_h/2)),(0, 255, 255))
                            start_rep_rec = False  
                        if start_dist < 0.08 or start:
                            
                            dist = np.sqrt(pow((landmarks[16].x-landmarks[12].x),2)+
                                           pow((landmarks[16].y-landmarks[12].y),2))
                            
                            if start_rep_rec: k += 1 # only starts to record reps, if max length was measured
                            else: k = 0
                            # creating rep lists
                            
                            for i,array in enumerate(arrays):
                                array.append(functions.dist_2_landmarks(landmarks, pairs[i]))
                                arrays[i] = arrays[i][-k:]
                            
                            # for array in arrays:
                            #     array = array[-k:]
                            # r_hand_shoulder = r_hand_shoulder[-k:]
                            # l_hand_shoulder = l_hand_shoulder[-k:]
                            # elbows = elbows[-k:]
                            # r_hip_elbow = r_hip_elbow[-k:]
                            # l_hip_elbow = l_hip_elbow[-k:]
                            # # hands = hands[-k:]
                            # hand_shoulder.append(dist)
                            rep_time.append(time.time())
                            # hand_shoulder = hand_shoulder[-k:]
                            rep_time = rep_time[-k:]
                            
                            dists = pers.get_dists()
                            start = True
                            functions.write_on_image(img, "Reps: " + str(rep_count), (10,105), (0, 255, 255))
                            
                            if ((0.8*dists[0]) < dist < 0.1):
                                # checking if hands are reaching the minimum area
                                #functions.write_on_image(img, str(dist), (int(i_w*x),int(i_h*y)),(0, 255, 255))
                                up = True
                                
                            if ((0.8*dists[2]) < dist < (1.1*dists[2])):
                                # checking if hands are reaching the maximum area
                                functions.write_on_image(img, str(dist), (int(i_w*x),int(i_h*y)),(0, 0, 255))
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
                                    k = 0           
                else:
                    functions.write_on_image(img, 'Not Visible'+str(results.pose_landmarks.landmark[16].visibility), (10,70), (0, 0, 255))
                    
                
                
            t_new = time.time()
            d_t = t_new - t
            
            fps = f"{(1/d_t):.1f}fps"
            functions.write_on_image(img, fps, (10,35),(0, 255, 255))
            
            cv2.imshow("Fitty", img)
            if cv2.waitKey(5) & 0xFF ==27: # ESC kilép
                break      
        else:
            break
    
    
    
    
if cap.isOpened():
    cap.release()
    cv2.destroyAllWindows()
try:
    pers.plot_data()
except:
    pass        
