
from scipy.stats.qmc import Halton
import numpy as np
from coloraide import Color
# Plot the results in 3D
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from cieran.basics.trajectory import Trajectory
from cieran.basics.environment import Environment


if __name__ == "__main__":

    def in_gamut(l, a, b):
        return Color("lab({}% {} {} / 1)".format(l, a, b)).in_gamut('srgb')

    dimensions = [(0,100), (-128,128), (-128,128)]
    
    # Initialize the Halton sampler
    sampler = Halton(len(dimensions))

    # Generate samples
    samples = sampler.random(1000)

    # Map the samples of size (num_samples, 3) with values between 0 and 1 to the desired dimensions across the 3 axes
    samples = np.array([dimensions[i][0] + (dimensions[i][1] - dimensions[i][0]) * samples[:, i] for i in range(len(dimensions))]).T

    points = []
    # Remove samples that are outside gamut or in collision with obstacles (circles of radius obstacle_rad)
    for i in range(len(samples)):
        if in_gamut(*samples[i]):
            points.append(samples[i])
    print("Number of points:", len(points))

    # Color distance function
    def color_distance(p1, p2):
        # breakpoint()
        return Color("lab({}% {} {} / 1)".format(*p1)).delta_e(Color("lab({}% {} {} / 1)".format(*p2)), method='2000')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the points array
    # ax.scatter([point[0] for point in points], [point[1] for point in points], [point[2] for point in points], c='b', marker='o', s=10)

    # Plot the centroids while coloring each point according to its CIELAB values, without point opacity
    ax.scatter([centroid[0] for centroid in points], [centroid[1] for centroid in points], [centroid[2] for centroid in points], c=[np.array(Color("lab({}% {} {} / 1)".format(*centroid)).convert('srgb')[:3]) for centroid in points], marker='o', s=40, alpha=1)

    # path = [np.array([0, 0, 0]), np.array([ 19.2989826 ,  41.74382203, -38.76011605]), np.array([ 32.28726385,  27.86865056, -41.53451605]), np.array([ 37.48745916,  -2.38207341, -36.01835605]), np.array([ 50.01187323, -21.99646146, -15.25931605]), np.array([ 51.50113104, -34.32994721,   6.33204395]), np.array([ 62.27993963, -45.39380943,  27.61332395]), np.array([ 70.80044744, -44.6164889 ,  49.12308395]), np.array([ 82.76333807, -28.52595392,  72.16692395]), np.array([85.80288885, -7.81036178, 78.41748395]), np.array([100,   0,   0])]

    # # traj = Trajectory(None, path)
    # # curve = traj.get_curve(1000)
    # # Plot the continuous curve
    # # breakpoint()
    # # ax.plot(*zip(*curve), c='r', linewidth=2)

    # # Plot the discrete path
    # ax.scatter([point[0] for point in path], [point[1] for point in path], [point[2] for point in path], c='r', marker='o')

    # label axes
    ax.set_xlabel('L*')
    ax.set_ylabel('a*')
    ax.set_zlabel('b*')

    # set axis ranges
    ax.set_xlim(0, 100)
    ax.set_ylim(-128, 128)
    ax.set_zlim(-128, 128)
    plt.show()

    # Final output:
    # Number of points: 2541
    # Average distance between colors: 5.13318935822155
    # Number of centroids: 512