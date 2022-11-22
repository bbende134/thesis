import uniMes

rigid_bodies = {"left hand forearm":[15,13, "Bende:l_wrist","Bende:l_elbow"],
"right hand forearm":[16,14, "Bende:r_wrist","Bende:r_elbow"],
"left hand upper arm":[11,13, "Bende:l_elbow","Bende:l_shoulder"],
"right hand upper arm":[12,14, "Bende:r_elbow","Bende:r_shoulder"],
"shoulder width":[12,11, "Bende:l_shoulder","Bende:r_shoulder"],
"hip width":[24,23, "Bende:l_hip","Bende:r_hip"],
"left thigh length":[23,25, "Bende:l_hip","Bende:l_knee"],
"right thigh length":[24,26, "Bende:r_hip","Bende:r_knee"],
"left lower leg length":[25,27, "Bende:l_knee","Bende:l_ankle"],
"right lower leg length":[26,28, "Bende:r_knee","Bende:r_ankle"],
"left foot length":[29,31, "Bende:l_heel","Bende:l_toe"],
"right foot length":[30,32, "Bende:r_heel","Bende:r_toe"],
}

# l_dist_mp_hands = functions.distance_plotting_pair(data_points_resampled, [15,13, "Bende:l_wrist","Bende:l_elbow"], False, time_resampled)
r_dist_mp_hands = uniMes.functions.distance_plotting_pair(uniMes.data_points_resampled, rigid_bodies['left hand forearm'], False, uniMes.time_resampled)


statistic_data = {}
# for name in rigid_bodies:
#     lengths = uniMes.functions.distance_plotting_pair(uniMes.data_points_resampled, rigid_bodies[name], False, uniMes.time_resampled)
    
#     statistic_data[name] = uniMes.functions.box_plotting_for_all(lengths, name)

for name in rigid_bodies:
    lengths = uniMes.functions.distance_plotting_pair(uniMes.data_points_resampled, rigid_bodies[name], False, uniMes.time_resampled)
    
    statistic_data[name] = uniMes.functions.box_plotting_for_pair(lengths, name)