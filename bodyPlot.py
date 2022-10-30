# import matplotlib.colors as mcolors


# by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(color))),
#                     name)
#                 for name, color in mcolors.TABLEAU_COLORS.items())
# names = [name for hsv, name in by_hsv]
# print(by_hsv)
#%% 

def plot_world_landmarks(
    ax,
    joints,
    frame,
    mp
):
    

    if mp:
        face_index_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        right_arm_index_list = [11, 13, 15, 17, 19, 21]
        left_arm_index_list = [12, 14, 16, 18, 20, 22]
        right_body_side_index_list = [11, 23, 25, 27, 29, 31]
        left_body_side_index_list = [12, 24, 26, 28, 30, 32]
        shoulder_index_list = [11, 12]
        waist_index_list = [23, 24]
        landmark_point = []
        ax.set_xlim3d(-1, 1)
        ax.set_ylim3d(-1, 1)
        ax.set_zlim3d(-1, 1)

        for joint in joints:
            landmark_point.append(
                [0.6, (joints[joint]['x'][frame], joints[joint]['y'][frame], joints[joint]['z'][frame])])
    else:
        face_index_list = ['Bende:m_head','Bende:l_head','Bende:r_head']
        right_arm_index_list = ['Bende:l_shoulder', 'Bende:l_elbow', 'Bende:l_wrist', 'Bende:l_pinky', 'Bende:l_pointer']
        left_arm_index_list = ['Bende:r_shoulder', 'Bende:r_elbow', 'Bende:r_wrist', 'Bende:r_pinky', 'Bende:r_pointer']
        right_body_side_index_list = ['Bende:l_shoulder', 'Bende:l_hip', 'Bende:l_knee', 'Bende:l_ankle', 'Bende:l_heel', 'Bende:l_toe']
        left_body_side_index_list = ['Bende:r_shoulder', 'Bende:r_hip', 'Bende:r_knee', 'Bende:r_ankle', 'Bende:r_heel', 'Bende:r_toe']
        shoulder_index_list = ['Bende:l_shoulder', 'Bende:r_shoulder']
        waist_index_list = ['Bende:l_hip', 'Bende:r_hip']
        landmark_point = {}
        
        ax.set_xlim3d(1, 3)
        ax.set_ylim3d(-0.5, 1.5)
        ax.set_zlim3d(-2, 0)

        for joint in joints:
            landmark_point[joint] = [0.6, (joints[joint]['x'][frame], joints[joint]['y'][frame], joints[joint]['z'][frame])]

    # face
    face_x, face_y, face_z = [], [], []
    for index in face_index_list:
        point = landmark_point[index][1]
        face_x.append(point[0])
        face_y.append(point[2])
        face_z.append(point[1] * (-1))

    # right arm
    right_arm_x, right_arm_y, right_arm_z = [], [], []
    for index in right_arm_index_list:
        point = landmark_point[index][1]
        right_arm_x.append(point[0])
        right_arm_y.append(point[2])
        right_arm_z.append(point[1] * (-1))

    # left arm
    left_arm_x, left_arm_y, left_arm_z = [], [], []
    for index in left_arm_index_list:
        point = landmark_point[index][1]
        left_arm_x.append(point[0])
        left_arm_y.append(point[2])
        left_arm_z.append(point[1] * (-1))

    # right body side
    right_body_side_x, right_body_side_y, right_body_side_z = [], [], []
    for index in right_body_side_index_list:
        point = landmark_point[index][1]
        right_body_side_x.append(point[0])
        right_body_side_y.append(point[2])
        right_body_side_z.append(point[1] * (-1))

    # left body side
    left_body_side_x, left_body_side_y, left_body_side_z = [], [], []
    for index in left_body_side_index_list:
        point = landmark_point[index][1]
        left_body_side_x.append(point[0])
        left_body_side_y.append(point[2])
        left_body_side_z.append(point[1] * (-1))

    # shoulder
    shoulder_x, shoulder_y, shoulder_z = [], [], []
    for index in shoulder_index_list:
        point = landmark_point[index][1]
        shoulder_x.append(point[0])
        shoulder_y.append(point[2])
        shoulder_z.append(point[1] * (-1))

    # waist
    waist_x, waist_y, waist_z = [], [], []
    for index in waist_index_list:
        point = landmark_point[index][1]
        waist_x.append(point[0])
        waist_y.append(point[2])
        waist_z.append(point[1] * (-1))
            
    #ax.cla()
    

    ax.scatter(face_x, face_y, face_z)
    ax.plot(right_arm_x, right_arm_y, right_arm_z)
    ax.plot(left_arm_x, left_arm_y, left_arm_z)
    ax.plot(right_body_side_x, right_body_side_y, right_body_side_z)
    ax.plot(left_body_side_x, left_body_side_y, left_body_side_z)
    ax.plot(shoulder_x, shoulder_y, shoulder_z)
    ax.plot(waist_x, waist_y, waist_z)
    

    return