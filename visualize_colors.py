
from scipy.stats.qmc import Halton
import numpy as np
from coloraide import Color
# Plot the results in 3D
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

if __name__ == "__main__":

    def in_gamut(l, a, b):
        return Color("lab({}% {} {} / 1)".format(l, a, b)).in_gamut('srgb')

    dimensions = [(0,100), (-128,128), (-128,128)]
    
    # Initialize the Halton sampler
    sampler = Halton(len(dimensions))

    # Generate samples
    samples = sampler.random(2000)

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
    ax.scatter([centroid[0] for centroid in points], [centroid[1] for centroid in points], [centroid[2] for centroid in points], c=[Color("lab({}% {} {} / 1)".format(*centroid)).convert('srgb')[:3] for centroid in points], marker='o', s=40, alpha=1)

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