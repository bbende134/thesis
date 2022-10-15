import objects
import random

r_hand_shoulder = []
l_hand_shoulder = []
elbows = []
r_hip_elbow = []
l_hip_elbow = []
time = [0.1,0.2,0.3,0.4,0.5]


pairs = [[12,16],[11,15],[14,13],[14,24],[13,23]]
arrays = [r_hand_shoulder, l_hand_shoulder, elbows, r_hip_elbow, l_hip_elbow]
name_array = ["r_hand_shoulder","l_hand_shoulder","elbows","r_hip_elbow","l_hip_elbow"]

for bodypart in arrays:
    for i in range(12):
        bodypart.append(random.random())

ex = objects.Exercise(name_array, arrays, time)
print(ex.get_data())

for i in range(3):
    for bodypart in arrays:
        time = [0.2,0.3,0.4,0.5,0.6]
        for i in range(5):
            bodypart[i]=(random.random())
    ex.new_set(arrays, time)

name, move, time = ex.get_data()
print(time)