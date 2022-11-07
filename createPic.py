import uniMes
import numpy as np



def create_pic(dataset):
    pics = {}
    maxes = {}
    mins = {}
    deviations = {}
    for pair in dataset:
        
        for record in dataset[pair]:
            if record.find("pose_world") > -1:
                pics[pair] = np.empty((0,len(dataset[pair][record][1]['x'])))
                for i,joint in enumerate(dataset[pair][record]):
                    for j,coordinates in enumerate(dataset[pair][record][joint]):
                        
                        pics[pair] = np.vstack([pics[pair],dataset[pair][record][joint][coordinates]])
        maxes[pair] = np.amax(pics[pair])
        mins[pair] = np.amin(pics[pair])
        deviations[pair] = maxes[pair] - mins[pair]
        pics[pair] -= mins[pair]
        pics[pair] *= (255/deviations[pair])
        pics[pair] = pics[pair].astype(np.uint8)

    return pics

picturing = create_pic(uniMes.data_points_resampled)

from matplotlib import pyplot as plt
for pair in picturing:
    plt.imshow(picturing[pair])
    plt.title(pair)
    plt.xlabel("Mintavételek")
    plt.ylabel("Virtuális izületek, és azok x,y,z koordinátái sorrendben")
    plt.show()

