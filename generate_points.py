
import numpy as np
from coloraide import Color

# LloydAlgorithm: points -> centroids
class LloydAlgorithm:

    def __init__(self, points):
        self.points = points
        # Initialize the centroids to random samples from the points
        # Create indices for the points
        indices = np.arange(len(self.points))
        # Randomly sample 1000 indices
        indices = np.random.choice(indices, 1000)
        # Initialize the centroids to the points at the sampled indices
        self.centroids = [self.points[index] for index in indices]

    
    def run(self, distance_function=None):
        # Repeat the algorithm until convergence
        counter = 0
        while True:
            counter += 1
            print(counter)
            # Store cluster assignments
            clusters = [[] for _ in range(len(self.centroids))]

            # Assign points to clusters
            for point in self.points:
                # Find the closest centroid given a custom distance function
            
                if distance_function is not None:
                    closest = np.argmin([distance_function(point, centroid) for centroid in self.centroids])
                # Find the closest centroid using the Euclidean distance
                else:
                    closest = np.argmin([np.linalg.norm(point - centroid) for centroid in self.centroids])
                
                # Add the point to the cluster
                clusters[closest].append(point)

            # Calculate the new centroids
            new_centroids = [np.mean(cluster, axis=0) for cluster in clusters]

            # Check for convergence
            if np.array_equal(self.centroids, new_centroids):
                break

            # Update the centroids
            self.centroids = new_centroids

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
    samples = sampler.random(10000)

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

    # Run the algorithm
    centroids = LloydAlgorithm(points).run(color_distance)

    # Plot the results in 3D
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the points array
    ax.scatter([point[0] for point in points], [point[1] for point in points], [point[2] for point in points], c='b', marker='o', s=10)

    # Plot the centroids
    ax.scatter([centroid[0] for centroid in centroids], [centroid[1] for centroid in centroids], [centroid[2] for centroid in centroids], c='r', marker='o', s=10)

    plt.show()

