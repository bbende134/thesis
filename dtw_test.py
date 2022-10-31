#%% 
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis
import random
import numpy as np
x = np.arange(0, 20, .5)
s1 = np.sin(x)
s2 = np.sin(x - 1)
random.seed(1)
for idx in range(len(s2)):
    if random.random() < 0.05:
        s2[idx] += (random.random() - 0.5) / 2
d, paths = dtw.warping_paths(s1, s2, window=25, psi=2)
best_path = dtw.best_path(paths)
dtwvis.plot_warpingpaths(s1, s2, paths, best_path)

# %%
import numpy as np

## A noisy sine wave as query
idx = np.linspace(0,2*6.28,num=100)
query = np.sin(idx-2) + np.random.uniform(size=100)/10.0

## A cosine is for template; sin and cos are offset by 25 samples
template = np.cos(idx)

## Find the best match with the canonical recursion formula
from dtw import *
alignment = dtw(query, template, keep_internals=True)
print(alignment)
## Display the warping curve, i.e. the alignment curve
alignment.plot(type="threeway")
# %%
import functions

print(functions.asd)
# %%
import scipy.signal 
import matplotlib.pyplot as plt
import numpy as np

# DISCLAIMER: This function is copied from https://github.com/nwhitehead/swmixer/blob/master/swmixer.py, 
#             which was released under LGPL. 
def resample_by_interpolation(signal, input_fs, output_fs):

    scale = output_fs / input_fs
    # calculate new length of sample
    n = round(len(signal) * scale)

    # use linear interpolation
    # endpoint keyword means than linspace doesn't go all the way to 1.0
    # If it did, there are some off-by-one errors
    # e.g. scale=2.0, [1,2,3] should go to [1,1.5,2,2.5,3,3]
    # but with endpoint=True, we get [1,1.4,1.8,2.2,2.6,3]
    # Both are OK, but since resampling will often involve
    # exact ratios (i.e. for 44100 to 22050 or vice versa)
    # using endpoint=False gets less noise in the resampled sound
    resampled_signal = np.interp(
        np.linspace(0.0, 1.0, n, endpoint=False),  # where to interpret
        np.linspace(0.0, 1.0, len(signal), endpoint=False),  # known positions
        signal,  # known data points
    )
    return resampled_signal

x = np.linspace(0, 10, 256, endpoint=False)
y = np.cos(-x**2/6.0)
yre = scipy.signal.resample(y,20)
xre = np.linspace(0, 10, len(yre), endpoint=False)

yre_polyphase = scipy.signal.resample_poly(y, 20, 256)
yre_interpolation = resample_by_interpolation(y, 256, 20)
yre_polyphase = scipy.signal.resample_poly(y, 20, 256)
yre_interpolation = resample_by_interpolation(y, 256, 20)

plt.figure(figsize=(10, 6))
plt.plot(x,y,'b', xre,yre,'or-')
plt.plot(xre, yre_polyphase, 'og-')
plt.plot(xre, yre_interpolation, 'ok-')
plt.legend(['original signal', 'scipy.signal.resample', 'scipy.signal.resample_poly', 'interpolation method'], loc='lower left')
plt.show()
# %%
# Import libraries
import matplotlib.pyplot as plt
import numpy as np
 
 
# Creating dataset
np.random.seed(10)
data = np.random.normal(100, 20, 200)
 
fig = plt.figure(figsize =(10, 7))
 
# Creating plot
plt.boxplot(data)
 
# show plot
plt.show()
# %%

import Uni_measurement
#%%
print(len(Uni_measurement.data_points_resampled))

# %%
