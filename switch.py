# -*- coding: utf-8 -*-

#%% 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from math import sin, radians


class AnimationHandler:
    def __init__(self, ax):

        self.ax = ax

        self.lines   = [self.ax.plot([], []), self.ax.plot([], [])]
        self.colors  = ['cyan', 'red']
        self.n_steps = [360, 360]
        self.step = 0

    def init_animation(self):
        for anim_idx in [0, 1]:
            self.lines[anim_idx], = self.ax.plot([0, 10], [0, 0], c=self.colors[anim_idx], linewidth=2)
        self.ax.set_ylim([-2, 2])
        self.ax.axis('off')

        return tuple(self.lines)

    def update_slope(self, step, anim_idx):
        self.lines[anim_idx].set_data([0, 10], [0, sin(radians(step))])

    def animate(self, step):
        # animation 1
        if 0 < step < self.n_steps[0]:
            self.update_slope(step, anim_idx=0)

        # animation 2
        if self.n_steps[0] < step < sum(self.n_steps):
            self.update_slope(step - self.n_steps[0], anim_idx=1)

        return tuple(self.lines)


if __name__ == '__main__':
    fig, axes = plt.subplots()
    animator = AnimationHandler(ax=axes)
    my_animation = animation.FuncAnimation(fig,
                                           animator.animate,
                                           frames=sum(animator.n_steps),
                                           interval=1,
                                           blit=True,
                                           init_func=animator.init_animation,
                                           repeat=False)

    # Writer = animation.writers['ffmpeg']
    # writer = Writer(fps=24, metadata=dict(artist='Me'), bitrate=1800)
    # my_animation.save('./anim_test.mp4', writer=writer)

    plt.show()

#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

x_data = []
y_data = []

fig, ax = plt.subplots()
ax.set_xlim(0, 200)
ax.set_ylim(0, 30)
line, = ax.plot(0, 0)

def animation_frame(i):
	x_data.append(i ** 2)
	y_data.append(i)

	line.set_xdata(x_data)
	line.set_ydata(y_data)
	return line, 

animation = FuncAnimation(fig, func=animation_frame, frames=np.arange(0, 20, 0.1), interval=10)
plt.show()


# %%
