import functions

import time






def frameEval(cap, pose, mp_drawing, mp_pose, cv2, np):
    t = time.time()
    success, img = cap.read()
    # i = i + 1
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
                
                
            if landmarks[16].visibility >= 0.7 and landmarks[0].visibility >= 0.7:
                functions.write_on_image(img, 'Visible', (10,70), (0, 255, 0))
            
            else:
                functions.write_on_image(img, 'Not Visible'+str(results.pose_landmarks.landmark[16].visibility), (10,70), (0, 0, 255))
        else:
            landmarks = []
            
            
        t_new = time.time()
        d_t = t_new - t
        # while (1/d_t) < 25:
        #     t_new = time.time()
        #     d_t = t_new - t
        
        fps = f"{(1/d_t):.1f}fps"
        functions.write_on_image(img, fps, (10,35),(0, 255, 255))
        
        cv2.imshow("Fitty", img)
        if cv2.waitKey(5) & 0xFF ==27: # ESC kilép
            return False, landmarks
        else:
            return True, landmarks