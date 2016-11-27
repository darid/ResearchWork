import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation


def update_lines(num, dataLines, lines):
    for line, data in zip(lines, dataLines):
        # NOTE: there is no .set_data() for 3 dim data...
        line.set_data(data[0:2, :num])
        line.set_3d_properties(data[2, :num])
        print num
        print data[0:2, :num]
        print data[2, :num]
    return lines


fig = plt.figure()
ax = p3.Axes3D(fig)
######################
# input must be array
X = np.array([[0.1, 0.2, 0.6, 0.4, 0.5], [0.2, 0.3, 0.4, 0.9, 0.6], [0.2, 0.5, 0.9, 0.8, 0.3]])
Y = np.array([[0.2, 0.3, 0.4, 0.9, 0.6], [0.2, 0.5, 0.9, 0.8, 0.3], [0.1, 0.2, 0.6, 0.4, 0.5]])
Z = np.array([[0.1, 0.2, 0.6, 0.4, 0.5], [0.2, 0.3, 0.4, 0.9, 0.6], [0.2, 0.5, 0.9, 0.8, 0.3]])
#
data = [np.vstack((X, Y, Z))]
######################################################################
lines = [ax.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1])[0] for dat in data]

ax.set_xlim3d([0.0, 1.0])
ax.set_xlabel('X')

ax.set_ylim3d([0.0, 1.0])
ax.set_ylabel('Y')

ax.set_zlim3d([0.0, 1.0])
ax.set_zlabel('Z')

line_ani = animation.FuncAnimation(fig, update_lines, 25, fargs=(data, lines),
                                   interval=50, blit=True)

# line_ani.save('lines.mp4')
plt.show()