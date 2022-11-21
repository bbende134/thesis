
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation
import pandas as pd

import uniMes
from mpl_toolkits.mplot3d.axes3d import Axes3D
import bodyPlot

setattr(Axes3D, 'arrow3D', uniMes.functions._arrow3D)

def update_graph(num):
    pair = 'kartarogatas_1'
    rec = "mp_pose_world_kartarogatas_1.csv"
    new_t = np.linspace(0,2,len(uniMes.time_resampled[pair][rec]))
    num += 50
    p_1 = 15
    ax.clear()
    bodyPlot.plot_world_landmarks(ax,uniMes.data_points_resampled[pair][rec],num, True)
    # for points in uniMes.data_points_resampled[pair][rec]:
    #     graph.set_data (uniMes.data_points_resampled[pair][rec][points]['x'][num],
    #     uniMes.data_points_resampled[pair][rec][points]['x'][num])
    #     graph.set_3d_properties(data.z)
    ax.view_init(elev=90., azim=0)
    temp_t = []
    temp_x = []
    temp_y = []
    temp_z = []
    for i in range(num):
        
        temp_x.append(uniMes.data_points_resampled[pair][rec][p_1]['x'][num-i])
        temp_y.append(uniMes.data_points_resampled[pair][rec][p_1]['z'][num-i])
        temp_z.append(-1*(uniMes.data_points_resampled[pair][rec][p_1]['y'][num-i]))
        temp_t.append(new_t[i])
    temp_t_x = []
    for j in range(num):
        temp_t_x.append(temp_t[j] -uniMes.data_points_resampled[pair][rec][p_1]['y'][num])

    ax.plot(temp_x,temp_t_x , zs=uniMes.data_points_resampled[pair][rec][p_1]['z'][num], zdir='y', label='curve in (x, y)')
    temp_t_y = []
    for j in range(num):
        temp_t_y.append(temp_t[j]+uniMes.data_points_resampled[pair][rec][p_1]['x'][num])
    ax.plot(temp_t_y,temp_y , zs=-1*uniMes.data_points_resampled[pair][rec][p_1]['y'][num], zdir='z', label='curve in (x, y)')
    for j in range(num):
        temp_t[j] *= (-1)
        temp_t[j] += uniMes.data_points_resampled[pair][rec][p_1]['z'][num]
    ax.plot(temp_t,temp_z , zs=uniMes.data_points_resampled[pair][rec][p_1]['x'][num], zdir='x', label='curve in (x, y)')

    d_x = uniMes.data_points_resampled[pair][rec][p_1]['x'][num]
    d_y = uniMes.data_points_resampled[pair][rec][p_1]['y'][num]
    d_z = uniMes.data_points_resampled[pair][rec][p_1]['z'][num]
    ax.arrow3D(0,0,0,
        d_x,d_z,(-1)*d_y,
        mutation_scale=20,
        fc='red')
    title.set_text('3D Test, time={}'.format(num))
    return title, graph, 


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
title = ax.set_title('3D Test')


ax.set_xlim3d(-1, 1)
ax.set_ylim3d(-1, 1)
ax.set_zlim3d(-1, 1)

pair = 'kartarogatas_1'
rec = "mp_pose_world_kartarogatas_1.csv"

graph, = bodyPlot.plot_world_landmarks(ax,uniMes.data_points_resampled[pair][rec],50, True)



ani = matplotlib.animation.FuncAnimation(fig, update_graph 
                               , interval=((1/120)),save_count=300)


f = r"c://dev/thesis/data/plots/animation_vect_traj_fel.gif" 
writergif = matplotlib.animation.PillowWriter(fps=240) 
ani.save(f, writer=writergif)



plt.show()

#%% Csak a csontváz

def update_graph(num):
    pair = 'karhajlitas_1'
    rec = "mp_pose_world_karhajlitas_1.csv"
    new_t = np.linspace(0,2,len(uniMes.time_resampled[pair][rec]))
    num += 50
    p_1 = "Bende:l_wrist"
    ax.clear()
    bodyPlot.plot_world_landmarks(ax,uniMes.data_points_resampled[pair][rec],num, True)
    # for points in uniMes.data_points_resampled[pair][rec]:
    #     graph.set_data (uniMes.data_points_resampled[pair][rec][points]['x'][num],
    #     uniMes.data_points_resampled[pair][rec][points]['x'][num])
    #     graph.set_3d_properties(data.z)
    ax.view_init(elev=0., azim=-90)
    title.set_text('3D Test, time={}'.format(num))
    return title, graph, 


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
title = ax.set_title('3D Test')


ax.set_xlim3d(-1, 1)
ax.set_ylim3d(-1, 1)
ax.set_zlim3d(-1, 1)

pair = 'karhajlitas_1'
rec = "mp_pose_world_karhajlitas_1.csv"

graph, = bodyPlot.plot_world_landmarks(ax,uniMes.data_points_resampled[pair][rec],50, True)



ani = matplotlib.animation.FuncAnimation(fig, update_graph 
                               , interval=((1/120)),save_count=300)


f = r"c://dev/thesis/data/plots/animation_karhajlitas.gif" 
writergif = matplotlib.animation.PillowWriter(fps=240) 
ani.save(f, writer=writergif)



plt.show()

#%%
# import matplotlib.pyplot as plt
# from numpy import *
# from numpy import linalg
# from mpl_toolkits.mplot3d import Axes3D

# x       = array([0, 0, 0, 0, 50, 50, 50, 50])
# y       = array([50,50,50,50,50,50,50,50])
# z       = array([12.5,37.5,62.5,87.5,25,50,75,0])

# data = concatenate((x[:,newaxis],y[:,newaxis],z[:,newaxis]), axis=1)

# center = data.mean(axis=0)

# distances = empty((0))
# for row in data:
#     distances = append(distances, linalg.norm(row - center))

# vertices = distances.argsort()[-4:]
# Vertices_reorder = [vertices[0], vertices[2], vertices[1], vertices[3], vertices[0]]

# # plot:
# fig = plt.figure()
# ax  = fig.add_subplot(111, projection = '3d')

# ax.set_xlim(0,100)
# ax.set_ylim(0,100)
# ax.set_zlim(0,100)
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')

# #ax.view_init(elev=90, azim=90)
# ax.scatter(x, y, z, zdir='z', s=20, c='g')
# ax.plot(x[Vertices_reorder], y[Vertices_reorder], z[Vertices_reorder])
# plt.show()

# #%%
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d.axes3d import Axes3D
# import functions


# setattr(Axes3D, 'arrow3D', functions._arrow3D)


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.set_xlim(0,2)
# ax.arrow3D(0,0,0,
#            1,1,1,
#            mutation_scale=20,
#            arrowstyle="-|>",
#            linestyle='dashed')
# ax.arrow3D(1,0,0,
#            1,1,1,
#            mutation_scale=20,
#            fc='red')
# ax.set_title('3D Arrows Demo')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# fig.tight_layout()




# #%%

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import numpy as np

# soa = np.array([[0, 0, 1, 1, -2, 0], [0, 0, 2, 1, 1, 0],
#                 [0, 0, 3, 2, 1, 0], [0, 0, 4, 0.5, 0.7, 0]])

# X, Y, Z, U, V, W = zip(*soa)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.quiver(X, Y, Z, U, V, W)
# ax.set_xlim([-1, 0.5])
# ax.set_ylim([-1, 1.5])
# ax.set_zlim([-1, 8])
# plt.show()

#%%

from matplotlib import pyplot as plt
import numpy as np
#from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

def gen(n):
    phi = 0
    while phi < 2*np.pi:
        yield np.array([np.cos(phi), np.sin(phi), phi])
        phi += 2*np.pi/n

def update(num, data, line):
    line.set_data(data[:2, :num])
    line.set_3d_properties(data[2, :num])

N = 100
data = np.array(list(gen(N)))
data = data.T
print(data[:2, :10])
print(data[2, :10])
line, = ax.plot(data[0, 0:1], data[1, 0:1], data[2, 0:1])

# Setting the axes properties
ax.set_xlim3d([-1.0, 1.0])
ax.set_xlabel('X')

ax.set_ylim3d([-1.0, 1.0])
ax.set_ylabel('Y')

ax.set_zlim3d([0.0, 10.0])
ax.set_zlabel('Z')

ani = animation.FuncAnimation(fig, update, N, fargs=(data, line), interval=10000/N, blit=False, )
#ani.save('matplot003.gif', writer='imagemagick')
plt.show()

#%% 
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# from math import sin, radians


# class AnimationHandler:
#     def __init__(self, ax):

#         self.ax = ax

#         self.lines   = [self.ax.plot([], []), self.ax.plot([], [])]
#         self.colors  = ['cyan', 'red']
#         self.n_steps = [360, 360]
#         self.step = 0

#     def init_animation(self):
#         for anim_idx in [0, 1]:
#             self.lines[anim_idx], = self.ax.plot([0, 10], [0, 0], c=self.colors[anim_idx], linewidth=2)
#         self.ax.set_ylim([-2, 2])
#         self.ax.axis('off')

#         return tuple(self.lines)

#     def update_slope(self, step, anim_idx):
#         self.lines[anim_idx].set_data([0, 10], [0, sin(radians(step))])

#     def animate(self, step):
#         # animation 1
#         if 0 < step < self.n_steps[0]:
#             self.update_slope(step, anim_idx=0)

#         # animation 2
#         if self.n_steps[0] < step < sum(self.n_steps):
#             self.update_slope(step - self.n_steps[0], anim_idx=1)

#         return tuple(self.lines)


# if __name__ == '__main__':
#     fig, axes = plt.subplots()
#     animator = AnimationHandler(ax=axes)
#     my_animation = animation.FuncAnimation(fig,
#                                            animator.animate,
#                                            frames=sum(animator.n_steps),
#                                            interval=1,
#                                            blit=True,
#                                            init_func=animator.init_animation,
#                                            repeat=False)

#     # Writer = animation.writers['ffmpeg']
#     # writer = Writer(fps=24, metadata=dict(artist='Me'), bitrate=1800)
#     # my_animation.save('./anim_test.mp4', writer=writer)

#     plt.show()

# #%%
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# import random

# x_data = []
# y_data = []
# rand_data = []

# fig, ax = plt.subplots()
# ax.set_xlim(0, 200)
# ax.set_ylim(0, 30)
# line, = ax.plot(0, 0)

# def animation_frame(i):

# 	x_data.append(i ** 2)
# 	y_data.append(i)

# 	line.set_xdata(x_data)
# 	line.set_ydata(y_data)
# 	return line, 

# animation = FuncAnimation(fig, func=animation_frame, frames=np.arange(0, 20, 0.1), interval=10)
# plt.show()


# # %%
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation


# fig = plt.figure(figsize=(16, 12))
# ax = fig.add_subplot(111)
# # You can initialize this with whatever
# im = ax.imshow(np.random.rand(6, 10), cmap='bone_r', interpolation='nearest')


# def animate(i):
#     aux = np.zeros(60)
#     aux[i] = 1
#     image_clock = np.reshape(aux, (6, 10))
#     im.set_array(image_clock)

# ani = animation.FuncAnimation(fig, animate, frames=60, interval=100)
# plt.show()
# %%



# %%
