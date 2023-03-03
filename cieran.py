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

env = cieran.initialize('#fff9db')
print(len(env.centroids))


if __name__ == '__main__':

    # Export both env.centroids and num_ramps as a space-delimited file, where each line is a color and the number of ramps for that color
    with open('ramps_per_color.txt', 'w') as f:
        for i, centroid in enumerate(env.centroids):
            env = cieran.initialize(list(centroid))
            print("Step " + str(i))
            print("Color: " + str(list(centroid)) + ", # of ramps: " + str(len(env.fitted_ramps)))
            for v in list(centroid):
                f.write(str(v) + ' ')
            f.write(str(len(env.fitted_ramps)))
            f.write('\n')

    # Export both env.centroids and num_ramps as a space-delimited file, where each line is a color and the number of ramps for that color
    with open('ramps_per_color.txt', 'w') as f:
        for i in range(len(env.centroids)):
            f.write(str(list(env.centroids[i])) + ' ' + str(num_ramps[i]))