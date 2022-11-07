import uniMes
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.axes3d import Axes3D
import bodyPlot

setattr(Axes3D, 'arrow3D', uniMes.functions._arrow3D)

# def ot_to_mp(dataset):
#     for pair in v_1:
#         for record in v_1[pair]:
#             for i in :
 


pair = 'kartarogatas_1'
rec = "mp_pose_world_kartarogatas_1.csv"
p_1 = 15
p_2 = 'Bende:r_hip'
t = 50
new_t = np.linspace(0,2,len(uniMes.time_resampled[pair][rec]))
# for k in range(49):
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     temp_t = []
#     temp_x = []
#     temp_y = []
#     temp_z = []
#     for i in range((k+1)*10):
        
#         temp_x.append(uniMes.data_points_resampled[pair][rec][p_1]['x'][t*(k+1)-i])
#         temp_y.append(uniMes.data_points_resampled[pair][rec][p_1]['z'][t*(k+1)-i])
#         temp_z.append(-1*(uniMes.data_points_resampled[pair][rec][p_1]['y'][t*(k+1)-i]))
#         temp_t.append(new_t[i])
        
#     ax.set_title('3D virtual skeleton')
#     ax.set_xlabel('x [m]')
#     ax.set_ylabel('y [m]')
#     ax.set_zlabel('z [m]')


#     bodyPlot.plot_world_landmarks(ax,uniMes.data_points_resampled[pair][rec],t*(k+1), True)
#     temp_t_x = []
#     for j in range((k+1)*10):
#         temp_t_x.append(temp_t[j] -uniMes.data_points_resampled[pair][rec][p_1]['y'][t*(k+1)])

#     ax.plot(temp_x,temp_t_x , zs=uniMes.data_points_resampled[pair][rec][p_1]['z'][t*(k+1)], zdir='y', label='curve in (x, y)')
#     temp_t_y = []
#     for j in range((k+1)*10):
#         temp_t_y.append(temp_t[j]+uniMes.data_points_resampled[pair][rec][p_1]['x'][t*(k+1)])
#     ax.plot(temp_t_y,temp_y , zs=-1*uniMes.data_points_resampled[pair][rec][p_1]['y'][t*(k+1)], zdir='z', label='curve in (x, y)')
#     for j in range((k+1)*10):
#         temp_t[j] *= (-1)
#         temp_t[j] += uniMes.data_points_resampled[pair][rec][p_1]['z'][t*(k+1)]
#     ax.plot(temp_t,temp_z , zs=uniMes.data_points_resampled[pair][rec][p_1]['x'][t*(k+1)], zdir='x', label='curve in (x, y)')



#     ax.arrow3D(0,0,0,
#             d_x,d_z,(-1)*d_y,
#             mutation_scale=20,
#             fc='red')

#     fig.tight_layout()
#     plt.show()
# d_x = uniMes.data_points_resampled[pair][rec][p_1]['y']
# d_x = uniMes.data_points_resampled[pair][rec][p_1]['x'][100]-uniMes.data_points_resampled[pair][rec][p_2]['x'][100]
# d_y = uniMes.data_points_resampled[pair][rec][p_1]['y'][100]-uniMes.data_points_resampled[pair][rec][p_2]['y'][100]
# d_z = uniMes.data_points_resampled[pair][rec][p_1]['z'][100]-uniMes.data_points_resampled[pair][rec][p_2]['z'][100]

new_t = np.linspace(0,2,len(uniMes.time_resampled[pair][rec]))
for k in range(49):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    bodyPlot.plot_world_landmarks(ax,uniMes.data_points_resampled[pair][rec],t*(k+1), True)
    for joint in uniMes.data_points_resampled[pair][rec]:
        d_x = uniMes.data_points_resampled[pair][rec][joint]['x'][t*(k+1)]
        d_y = uniMes.data_points_resampled[pair][rec][joint]['y'][t*(k+1)]
        d_z = uniMes.data_points_resampled[pair][rec][joint]['z'][t*(k+1)]
    #     d_x = uniMes.data_points_resampled[pair][rec][p_1]['x'][t*(k+1)]
    #     d_y = uniMes.data_points_resampled[pair][rec][p_1]['y'][t*(k+1)]
    #     d_z = uniMes.data_points_resampled[pair][rec][p_1]['z'][t*(k+1)]
        ax.arrow3D(0,0,0,
                d_x,d_z,(-1)*d_y,
                mutation_scale=10,
                fc='red')

    fig.tight_layout()
    plt.show()
ax.arrow3D(uniMes.data_points_resampled[pair][rec][p_2]['x'][100],uniMes.data_points_resampled[pair][rec][p_2]['z'][100],(-1)*uniMes.data_points_resampled[pair][rec][p_2]['y'][100],
           d_x,d_z,(-1)*d_y,
           mutation_scale=20,
           fc='red')
# p_1 = 'Bende:r_knee'
# p_2 = 'Bende:r_hip'



#%%
# import numpy as np

# import matplotlib.pyplot as plt

# t = np.arange(256)

# sp = list(np.fft.fft(d_x))
# for i in range(2):
#     sp.pop(i)
# for i in range(400):
#     sp.pop()
# freq = np.fft.fftfreq(t.shape[-1])

# plt.plot(sp)

# plt.show()
# %%
