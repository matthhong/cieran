
import cieran
import numpy as np
import matplotlib.pyplot as plt

def draw_chart(cmap):

    t = np.linspace(0, 2 * np.pi, 1024)
    data2d = np.sin(t)[:, np.newaxis] * np.cos(t)[np.newaxis, :]
    # Draw a chart of data2d with the given colormap
    fig, ax = plt.subplots()
    ax.imshow(data2d, cmap=cmap)

    plt.show()

env, traj = cieran.query([48.325, -19.2993, -16.5717], render=draw_chart)