import uniMes
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D

setattr(Axes3D, 'arrow3D', uniMes.functions._arrow3D)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(0,2)


pair = 'kartarogatas_1'
rec = "ot_kartarogatas_1.csv"
p_1 = 'Bende:l_wrist'
p_2 = 31
d_x = uniMes.data_points_resampled[pair][rec][p_1]['y']
# d_x = uniMes.data_points_resampled[pair][rec][p_1]['x'][100]-uniMes.data_points_resampled[pair][rec][p_2]['x'][100]
# d_y = uniMes.data_points_resampled[pair][rec][p_1]['y'][100]-uniMes.data_points_resampled[pair][rec][p_2]['y'][100]
# d_z = uniMes.data_points_resampled[pair][rec][p_1]['z'][100]-uniMes.data_points_resampled[pair][rec][p_2]['z'][100]

# ax.arrow3D(data_points_resampled[pair][rec][p_1]['x'][100],data_points_resampled[rec][p_1]['z'][100],(-1)*data_points_resampled[rec][p_1]['y'][100],
#             data_points_resampled[pair][rec][p_2]['x'][100],data_points_resampled[rec][p_2]['z'][100],(-1)*data_points_resampled[rec][p_2]['y'][100],
           
#            mutation_scale=20,
#            fc='red')
# ax.arrow3D(uniMes.data_points_resampled[pair][rec][p_2]['x'][100],uniMes.data_points_resampled[pair][rec][p_2]['z'][100],(-1)*uniMes.data_points_resampled[pair][rec][p_2]['y'][100],
#            d_x,d_z,(-1)*d_y,
#            mutation_scale=20,
#            fc='red')
ax.set_title('3D virtual skeleton')
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_zlabel('z [m]')

import bodyPlot

bodyPlot.plot_world_landmarks(ax,uniMes.data_points_resampled[pair][rec],100, False)

fig.tight_layout()
plt.show()

#%%
import numpy as np

import matplotlib.pyplot as plt

t = np.arange(256)

sp = list(np.fft.fft(d_x))
for i in range(2):
    sp.pop(i)
for i in range(400):
    sp.pop()
freq = np.fft.fftfreq(t.shape[-1])

plt.plot(sp)

plt.show()
# %%
