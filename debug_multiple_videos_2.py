import functions
import matplotlib as plt
import os
import numpy as np
    


   

#Importing videos and save Mediapipe landmark data

names = ['C:/Users/BAR7BP/Documents/Projects/Innohub/Video/Videok/Albert_squat_1.mp4',
         'C:/Users/BAR7BP/Documents/Projects/Innohub/Video/Videok/Albert_squat_2.mp4',
         'C:/Users/BAR7BP/Documents/Projects/Innohub/Video/Videok/Albert_squat_3.mp4'];



vid = [None]*len(names)
functions.video_evaluation(vid, names)


## Calculating Zero deviations between hips and heels

plots = [None]*len(names)

for j in range(len(plots)):
    for i in range(len(vid[j].getData())):
        if plots[j] == None:
            plots[j] = objects.PlotData(names[j], functions.calc_zero_deviation(vid[j].getData()[i]))
        else:
            plots[j].add_element(functions.calc_zero_deviation(vid[j].getData()[i]))
    
    plt.pyplot.plot(plots[j].getData(),label=names[j])
    

plt.tight_layout()
plt.legend() 
plt.show()
        

