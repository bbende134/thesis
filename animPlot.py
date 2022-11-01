import uniMes
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


x_data = []
y_data = []

fig, ax = plt.subplots()
ax.set_xlim(0, max(uniMes.time_data['ot_kitores_4.csv']))
ax.set_ylim(min(uniMes.data_points['ot_kitores_4.csv']['Bende:l_wrist']['y']),
 max(uniMes.data_points['ot_kitores_4.csv']['Bende:l_wrist']['y']))
line, = ax.plot(0, 0)

def animation_frame(i):
	
	x_data.append(uniMes.time_data['ot_kitores_4.csv'][i])
	y_data.append(uniMes.data_points['ot_kitores_4.csv']['Bende:l_wrist']['y'][i])

	line.set_xdata(x_data)
	line.set_ydata(y_data)
	return line, 

animation = FuncAnimation(fig,
 func=animation_frame,
  frames=len(uniMes.data_points['ot_kitores_4.csv']['Bende:l_wrist']['x']),
   interval=((1/120)*1000))
plt.show()