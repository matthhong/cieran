import numpy as np
from coloraide import Color

# MacQueenAlgorithm: points -> centroids
class MacQueenAlgorithm:

    def __init__(self, points, k=1000):
        self.points = points
        self.k = k
        # Initialize the centroids to the first k points
        self.centroids = self.points[:k]

    def run(self, distance_function=None):
        # Initialize k clusters that already have the first k points
        clusters = [[] for _ in range(self.k)]
        for i in range(self.k):
            clusters[i].append(self.points[i])

        # Assign rest of the points to clusters, and update the centroids
        for point in self.points[self.k:]:
            # Find the closest centroid given a custom distance function
            if distance_function is not None:
                closest = np.argmin([distance_function(point, centroid) for centroid in self.centroids])
            # Find the closest centroid using the Euclidean distance
            else:
                closest = np.argmin([np.linalg.norm(point - centroid) for centroid in self.centroids])
            
            # Add the point to the cluster
            clusters[closest].append(point)

            # Update the centroids
            self.centroids[closest] = np.mean(clusters[closest], axis=0)
        
        return self.centroids


    
# Test the algorithm
if __name__ == "__main__":
    from scipy.stats.qmc import Halton

    def in_gamut(l, a, b):
        return Color("lab({}% {} {} / 1)".format(l, a, b)).in_gamut('srgb')

    dimensions = [(0,100), (-128,128), (-128,128)]
    
    # Initialize the Halton sampler
    sampler = Halton(len(dimensions))

    # Generate samples
    samples = sampler.random(20000)

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

    # # Run the algorithm
    # centroids = MacQueenAlgorithm(points, k=512).run(color_distance)

    # # Remove nan centroids
    # centroids = [centroid for centroid in centroids if not np.isnan(centroid).any()]

    # # Find average distance between each centroid and its 5 nearest neighbors
    # distances = []
    # for centroid in centroids:
    #     distances.append(np.mean(sorted([color_distance(centroid, other) for other in centroids])[:5]))
    
    # print("Average distance between centroids:", np.mean(distances))
    # print("Number of centroids:", len(centroids))

    # # Make sure the centroids are in gamut
    # for centroid in centroids:
    #     assert in_gamut(*centroid)

    # # Save the centroids to a file
    # with open("centroids.txt", "w") as f:
    #     for centroid in centroids:
    #         f.write("{} {} {}\n".format(*centroid))

    # Load the centroids from a file
    with open("centroids.txt", "r") as f:
        centroids = [[float(x) for x in line.split()] for line in f.readlines()]

    # Plot the results in 3D
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the points array
    # ax.scatter([point[0] for point in points], [point[1] for point in points], [point[2] for point in points], c='b', marker='o', s=10)

    # Plot the centroids while coloring each point according to its CIELAB values, without point opacity
    ax.scatter([centroid[0] for centroid in centroids], [centroid[1] for centroid in centroids], [centroid[2] for centroid in centroids], c=[Color("lab({}% {} {} / 1)".format(*centroid)).convert('srgb')[:3] for centroid in centroids], marker='o', s=40, alpha=1)

    plt.show()

    # Final output:
    # Number of points: 2541
    # Average distance between colors: 5.13318935822155
    # Number of centroids: 512