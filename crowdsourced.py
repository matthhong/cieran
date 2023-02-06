from coloraide import Color
import numpy as np
import networkx as nx
from KDTree import KDTree
from scipy.stats.qmc import Halton
# from scipy.cluster import KMeans

# def custom_round(x, base=10):
#     nearest = base * np.round(x / base)
#     # nearest = x
#     if x >= 0:
#         return np.floor(nearest)
#     else:
#         return np.ceil(nearest)

class CrowdsourcedStates:

    RAMPS_FILE = 'ramps.csv'
    CENTROID_FILE = 'centroids.txt'
    START = np.array([100, 0, 0])
    END = np.array([0, 0, 0])

    def __init__(self, color):
        self.color = color
        self.ramps = self.load_ramps()
        self.centroids = self.load_centroids()

        self.graph = nx.DiGraph()

        # rounded_color = [custom_round(x) for x in self.color]
        # diff = np.array(rounded_color) - np.array(self.color)

        # fit the ramps to the given color
        self.fitted_ramps = []
        tree = KDTree(self.centroids)
        num_out_of_gamut = 0
        for ramp in self.ramps:
            translated_ramp = self.fit_ramp_to_color(ramp, self.color)   

            new_ramp = []

            last_centroid = None
            # Check if every point in the ramp is in gamut
            for i in range(len(translated_ramp)):
                point = translated_ramp[i]
                # round the point to the nearest integer
                # point = [custom_round(x) - diff[i] for i, x in enumerate(point)]

                # Find the nearest centroid to the point using KDTree
                tree = KDTree(self.centroids)
                nearest_centroid = tree.query(point, k=1)[1][0]
                # if not Color("lab({}% {} {} / 1)".format(*point)).in_gamut('srgb'):
                #     if not out_of_gamut:
                #         num_out_of_gamut += 1
                #     out_of_gamut = True
                # self.graph.add_node(tuple(point))
                # if not out_of_gamut:
                # breakpoint()
                if last_centroid is not None:
                    # Compare the first element between the last centroid and the current centroid
                    l_diff = last_centroid[0] - nearest_centroid[0]
                    if l_diff > 0:
                        continue
                new_ramp.append(nearest_centroid)
                last_centroid = nearest_centroid
        
            self.fitted_ramps.append(new_ramp)
        
        print("num ramps: " + str(len(self.fitted_ramps)))
        print("num out of gamut: " + str(num_out_of_gamut))
        
        # Add the ramp points to the graph, and edges between adjacent ramp points
        for i, ramp in enumerate(self.fitted_ramps):
            # Put start and end points into the ramp
            self.fitted_ramps[i] = [self.END] + ramp + [self.START]
            ramp = self.fitted_ramps[i]
            for i in range(len(ramp)-1, 0, -1):
                distance = np.linalg.norm(ramp[i] - ramp[i-1])
                if ramp[i] == self.END or ramp[i-1] == self.START:
                    distance = 0
                self.graph.add_edge(tuple(ramp[i]), tuple(ramp[i-1]), weight=distance)

        self.longest_path_length = nx.dag_longest_path_length(self.graph)
    

    def load_centroids(self):
        # centroids = np.empty((0,3))
        # with open(self.CENTROID_FILE, 'r') as f:
        #     for line in f:
        #         values = line.replace('\n','').split(' ')
        #         centroids = np.append(centroids, [np.array([float(values[0]), float(values[1]), float(values[2])])], axis=0)
        # return centroids
        dimensions = [(0,100), (-128,128), (-128,128)]
    
        # Initialize the Halton sampler
        sampler = Halton(len(dimensions))

        # Generate samples
        samples = sampler.random(10000)

        # Map the samples of size (num_samples, 3) with values between 0 and 1 to the desired dimensions across the 3 axes
        samples = np.array([dimensions[i][0] + (dimensions[i][1] - dimensions[i][0]) * samples[:, i] for i in range(len(dimensions))]).T

        points = np.empty((0,3))
        # Remove samples that are outside gamut or in collision with obstacles (circles of radius obstacle_rad)
        for i in range(len(samples)):
            if self.in_gamut(*samples[i]):
                points = np.append(points, [samples[i]], axis=0)
        print("Number of points:", len(points))
        return points

    def in_gamut(self, l, a, b):
        return Color("lab({}% {} {} / 1)".format(l, a, b)).in_gamut('srgb')


    def load_ramps(self):
        # Each line is a ramp, where each triplet is a 3D point of LAB color, and there are 9 points per ramp
        # load into np array
        ramps = np.empty((0,9,3))
        with open(self.RAMPS_FILE, 'r') as f:
            for line in f:
                values =  line.replace('\n','').split(',')
                ramp = np.empty((0,3))
                for i in range(0, len(values), 3):
                    point = np.array([float(values[i]), float(values[i+1]), float(values[i+2])])
                    ramp = np.append(ramp, [point], axis=0)
                # Sort the ramp by luminance
                ramp = ramp[ramp[:,0].argsort()]
                ramps = np.concatenate((ramps, ramp[np.newaxis, ...]), axis=0)
        return ramps


    def fit_ramp_to_color(self, ramp, lab_color):
        min_distance = 99999
        min_distance_index = -1
        distance = 0
        given_luminance = lab_color[0]

        for i in range(len(ramp)):
            distance = abs(given_luminance - ramp[i][0])
            if distance < min_distance:
                min_distance = distance
                min_distance_index = i

        closest_point = ramp[min_distance_index]

        difference_vector = [0, 0, 0]
        new_start_color = [0, 0, 0]
        for i in range(3):
            difference_vector[i] = closest_point[i] - lab_color[i]
            new_start_color[i] = ramp[0][i] - difference_vector[i]

        new_ramp = self.translate_curve(ramp, new_start_color)

        return new_ramp


    def translate_curve(self, curve, starting_point):
        translation_vector_x = curve[0][0] - starting_point[0]
        translation_vector_y = curve[0][1] - starting_point[1]
        translation_vector_z = curve[0][2] - starting_point[2]

        translated_curve = []

        for point in curve:
            translated_point = []
            translated_point.append(point[0] - translation_vector_x)
            translated_point.append(point[1] - translation_vector_y)
            translated_point.append(point[2] - translation_vector_z)

            translated_curve.append(translated_point)

        return translated_curve


if __name__ == "__main__":
    states = CrowdsourcedStates([26.6128, 37.85, -44.51])

    # Visualize the states in 3D LAB space
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for ramp in states.fitted_ramps:
        ramp = np.array(ramp)
        ax.plot(ramp[:,0], ramp[:,1], ramp[:,2])

    # Display all nodes as scatter plot, in gray
    print("Number of nodes:", len(states.graph.nodes))

    print("Number of original colors:" , len(states.fitted_ramps * 9))
    for node in states.graph.nodes:
        # if not in any ramp
        if not any(np.all(node == ramp) for ramp in states.fitted_ramps):
            ax.scatter(node[0], node[1], node[2], c='gray', marker='o')
    
    # Label the axes
    ax.set_xlabel('L')
    ax.set_ylabel('A')
    ax.set_zlabel('B')

    plt.show()

    breakpoint()