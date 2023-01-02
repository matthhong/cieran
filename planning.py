import numpy as np
from networkx import Graph, astar_path
from coloraide import Color
from scipy.stats.qmc import Halton
from scipy.spatial import KDTree


class Planning:
    # Ramp planner
    def __init__(self, start, waypoints, end, obstacles, obstacle_rad, num_samples=1000):
        self.start = start
        self.waypoints = waypoints
        self.end = end
        self.obstacles = obstacles
        self.obstacle_rad = obstacle_rad
        self.num_samples = num_samples

        self.samples = []
        self.dimensions = [(0,100), (-128,128), (-128,128)]
        
        # Initialize the Halton sampler
        sampler = Halton(len(self.dimensions))

        # Generate samples
        samples = sampler.random(size=self.num_samples)

        # Map the samples of size (num_samples, 3) with values between 0 and 1 to the desired dimensions across the 3 axes
        samples = np.array([np.array([self.dimensions[i][0] + (self.dimensions[i][1] - self.dimensions[i][0]) * samples[:, i] for i in range(len(self.dimensions))]).T])

        # Remove samples that are outside gamut or in collision with obstacles (circles of radius obstacle_rad)
        for i in range(len(samples)):
            for j in range(len(self.obstacles)):
                if self.in_gamut(*samples[i]) and np.linalg.norm(samples[i] - self.obstacles[j]) > self.obstacle_rad:
                    self.samples.append(samples[i])

        self.graph = Graph()
        self.graph.add_node('start', pos=start)
        self.graph.add_node('end', pos=end)

        self.path = None
        

    def add_edges(self):
        # iterate over pairs of nodes and add edges within a distance if their midpoint is collision free
        tree = KDTree(self.samples)
        for n1 in self.samples:
            # for each node connect try to connect to k nearest self.samples
            idxs = tree.query([n1], 16)[0]
            
            for idx in idxs:
                n2 = self.samples[idx]
                if n2 == n1:
                    continue
                    
                if self.can_connect(n1, n2):
                    self.graph.add_edge(n1, n2, weight=1)

    
    def in_gamut(self, l, a, b):
        return Color("lab({}% {} {} / 1)".format(l, a, b)).in_gamut('srgb')


    def can_connect(self, n1, n2):
        # check if the midpoint of the line between two nodes is in collision with obstacles
        midpoint = (n1 + n2) / 2
        for i in range(len(self.obstacles)):
            if self.in_gamut(*midpoint) and np.linalg.norm(midpoint - self.obstacles[i]) > self.obstacle_rad:
                return True
        return False


    def find_path(self):
        # find a path from start to end through waypoints using A* and return the path
        pass

    def get_path(self):
        if self.path is None:
            self.add_samples()
            self.add_edges()
            self.path = self.find_path()
            
        return self.path
        

