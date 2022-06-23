from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
plt.rc('font', family='Times New Roman')
fig = plt.figure()
ax = fig.gca(projection='3d')

# window data.
# X = np.array([10,15,20,25,30])
# Y = np.array([5,10,15,20,25])
# X,Y = np.meshgrid(X, Y)
# Z = np.array([[96.71,95.57,95.61,95.84,96.58],[96.9,97.14,97.49,97.86,96.97],[97.04,97.21,97.57,97.74,96.59],[96.79,97.53,96.77,96.49,97.47],[96.3,97.22,97.51,97.2,96.88]])

# window data.
X = np.array([2,3,4,5,6])
Y = np.array([6,12,24,48,64])
X,Y = np.meshgrid(X, Y)
Z = np.array([[93.98,95.86,96.49,96.62,96.74],[96.87,97.12,96.99,96.99,96.99],[96.37,97.37,97.49,97.37,97.24],[96.37,96.99,96.37,95.99,96.87],[96.62,96.62,96.87,95.99,95.24]])


# Plot the surface.
surf = ax.plot_surface(X, Y, Z ,cmap=cm.OrRd,
                       linewidth=0, antialiased=False)
# zdir = 'z', offset = -2 表示投影到z = -2上
# ax.contour(X, Y, Z, zdir = 'z', offset = 90, cmap = cm.OrRd)
# Customize the z axis.
ax.tick_params(labelsize=15)
ax.set_zlim(90, 100)
ax.yaxis.set_major_locator(LinearLocator(5))
ax.set_yticklabels([6,12,24,48,64])
# ax.set_yticklabels([5,10,15,20,25])
ax.zaxis.set_major_locator(LinearLocator(5))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.01f'))

# Add a color bar which maps values to colors.
position=fig.add_axes([0.9, 0.1, 0.04, 0.7])#位置[左,下,右,上]
a = plt.colorbar(surf,cax = position ,orientation='vertical', shrink=0.5, aspect=5)
a.ax.tick_params(labelsize=15)
# ax.set_xlabel('Window size', fontsize=15)
# ax.set_ylabel('Step size', fontsize=15)
# ax.set_zlabel('Accuracy (%)', fontsize=15)
# plt.savefig("./window.svg", dpi=500, bbox_inches='tight',format="svg")

ax.set_xlabel('Hidden layer depth', fontsize=15)
ax.set_ylabel('Channel count', fontsize=15)
ax.set_zlabel('Accuracy (%)', fontsize=15)
plt.savefig("./hidden.svg", dpi=500, bbox_inches='tight',format="svg")

plt.show()