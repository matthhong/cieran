import numpy as np
from networkx import DiGraph, astar_path, dijkstra_path, shortest_simple_paths
from coloraide import Color
from scipy.stats.qmc import Halton
from KDTree import KDTree
from colorspace import color_distance, lab_to_rgb

from utils import lipschitz, total_variation, lipschitz_3d, total_variation_3d
from itertools import product, islice

import igraph as ig

class Planning:
    # Ramp planner
    def __init__(self, waypoints=None, obstacles=None, obstacle_rad=0, num_samples=1000, min_c=0):

        self.min_c = min_c

        # convert waypoints to a list of lists of floats
        if waypoints is not None:
            self.waypoints = np.array([[float(waypoint[0]), float(waypoint[1]), float(waypoint[2])] for waypoint in waypoints])

        # Insert start (0.0, 0.0, 0.0)
        self.waypoints = np.append(self.waypoints, [[0.0, 0.0, 0.0]], axis=0)
        self.waypoints = np.append(self.waypoints, [[100.0, 0.0, 0.0]], axis=0)

        # sort waypoints by first element
        self.waypoints = self.waypoints[self.waypoints[:,0].argsort()]
        
        if obstacles is not None:
            # convert obstacles to numpy array of floats
            self.obstacles = np.array([[float(obstacle[0]), float(obstacle[1]), float(obstacle[2])] for obstacle in obstacles])

        self.obstacle_rad = obstacle_rad
        self.num_samples = num_samples

        # samples is a np.array of size (num_samples, 3)
        self.samples = np.empty((0,3))
        self.dimensions = [(0,100), (-128,128), (-128,128)]

        # Load samples from centroid file
        with open('centroids.txt', 'r') as f:
            for line in f:
                # Append if it doesn't hit an obstacle
                centroid = np.array([float(x) for x in line.split()])
                if not self.hits_obstacle(*centroid):
                    self.samples = np.append(self.samples, [centroid], axis=0)
        

        # Add start, waypoints, and end to the samples
        for i in range(len(self.waypoints)):
            self.samples = np.append(self.samples, [self.waypoints[i]], axis=0)

        # Sort samples by first element
        self.samples = self.samples[self.samples[:,0].argsort()]
    
        # Keep track of each node's in-degree as a dict
        self.not_connected = self.samples.copy()[:-1]

        self.graph = DiGraph()

        self.paths = None
        

    def add_edges(self):
        self.add_edges_inner(self.not_connected, direction=1)
        self.add_edges_inner(self.not_connected, direction=-1)
    

    def add_edges_inner(self, nodes, direction):
        # iterate over pairs of nodes and add edges within a distance if their midpoint is collision free
        tree = KDTree(self.samples, constrained_axis=0, direction=direction, obstacles=self.obstacles, obstacle_cost_multiplier=0.05)

        for n1 in nodes:
            # for each node connect try to connect to k nearest self.samples
            distances, points = tree.query(n1, 16)
            
            for k, n2 in enumerate(points):
                # check if n2 is a waypoint, otherwise check if the midpoint is collision free
                if np.isin(n2, self.waypoints).any() or self.can_connect(n1, n2):
                    # Depending on the direction, add the edge in the correct direction
                    if direction == 1:
                        self.graph.add_edge(tuple(n1), tuple(n2), weight=distances[k])
                    if direction == -1:
                        self.graph.add_edge(tuple(n2), tuple(n1), weight=distances[k])
                    
                    self.not_connected = self.not_connected[~np.isin(self.not_connected, [n2]).all(axis=1)]
                    
    
    def in_gamut(self, l, a, b):
        return Color("lab({}% {} {} / 1)".format(l, a, b)).in_gamut('srgb')


    def hits_obstacle(self, l, a, b):
        # check if a node is in collision with an obstacle, or check if the a* and b* values pass the min_sat threshold
        for i in range(len(self.obstacles)):
            if np.linalg.norm([l, a, b] - self.obstacles[i]) < self.obstacle_rad:
                return True
        
        # Convert a and b values in Euclidean space to c in polar space
        c = np.sqrt(a**2 + b**2)
        if c < self.min_c:
            return True

        return False


    def can_connect(self, n1, n2):
        # check if the midpoint of the line between two nodes is in collision with obstacles
        midpoint = (n1 + n2) / 2
        for i in range(len(self.obstacles)):
            if not (self.in_gamut(*midpoint) and np.linalg.norm(midpoint - self.obstacles[i]) > self.obstacle_rad):
                return False
        return True


    def find_path(self):
        # In this version, implement a reinforcement learning approachs
        # Use a greedy approach to find the shortest path
        pass


    def find_path(self):
        # find a path from start to end through waypoints using A* and return the path

        # Iterate over the waypoints and run A* between each pair of waypoints
        path = [(0.0,0.0,0.0)]

        wps = self.waypoints.copy()

        while len(wps) > 1:
            # Run A* between the first waypoint and the rest of the waypoints
            start = wps[0]
            end = wps[1]
            wps = wps[1:]

            # Run A* to find the shortest path between start and end
            path += astar_path(self.graph, tuple(start), tuple(end), weight='weight')[1:]
        # breakpoint()
        return path


    def get_path(self):
        if self.paths is None:
            self.add_edges()
            self.path = self.find_path()
            
        return self.path
        

# Test the planner
if __name__ == "__main__":
    waypoints = [[20, 0, 0], [70, 0, 0]]
    obstacles = [[50, 0, 0]]
    planner = Planning(waypoints, obstacles, 10, 1000, 30)
    path = planner.get_path()

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # obstacles in red
    ax.scatter(*zip(*planner.obstacles), c='r')

    # waypoints in blue
    ax.scatter(*zip(*planner.waypoints), c='b')

    # get indices of waypoints in samples
    waypoint_indices = [np.where((planner.samples == waypoint).all(axis=1))[0][0] for waypoint in planner.waypoints]

    # mask samples to only include samples that are not waypoints
    samples = planner.samples[~np.isin(np.arange(len(planner.samples)), waypoint_indices)]

    # samples in gray
    ax.scatter(*zip(*samples), c='gray')
    
    ax.plot(*zip(*path), c='g')

    plt.show()