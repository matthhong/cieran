import numpy as np
from networkx import Graph, astar_path, dijkstra_path
from coloraide import Color
from scipy.stats.qmc import Halton
from KDTree import KDTree
from colorspace import color_distance, lab_to_rgb

class Planning:
    # Ramp planner
    def __init__(self, waypoints=None, obstacles=None, obstacle_rad=0, num_samples=1000):
        # convert waypoints to a list of lists of floats
        if waypoints is not None:
            self.waypoints = [[float(waypoint[0]), float(waypoint[1]), float(waypoint[2])] for waypoint in waypoints]

        # sort waypoints by first element
        self.waypoints.sort(key=lambda x: x[0])
        
        if obstacles is not None:
            self.obstacles = [[float(obstacle[0]), float(obstacle[1]), float(obstacle[2])] for obstacle in obstacles]

        self.obstacle_rad = obstacle_rad
        self.num_samples = num_samples

        self.samples = []
        self.dimensions = [(0,100), (-128,128), (-128,128)]

        # Load samples from centroid file
        with open('centroids.txt', 'r') as f:
            for line in f:
                self.samples.append([float(x) for x in line.split()])
        
        # Add start, waypoints, and end to the samples
        for i in range(len(self.waypoints)):
            self.samples.append(self.waypoints[i])
        self.samples.insert(0, [0, 0, 0])
        self.samples.append([100, 0, 0])

        # Transform the samples to a numpy array
        self.samples = np.array(self.samples)

        self.graph = Graph()

        self.path = None
        

    def add_edges(self):
        # iterate over pairs of nodes and add edges within a distance if their midpoint is collision free
        tree = KDTree(self.samples, constrained_axis=0)
        for n1 in self.samples[:-1]:
            # for each node connect try to connect to k nearest self.samples
            distances, points = tree.query(n1, 16)
            
            for k, n2 in enumerate(points):
                # check if n2 is a waypoint, otherwise check if the midpoint is collision free
                if np.isin(n2, self.waypoints).any() or self.can_connect(n1, n2):
                    self.graph.add_edge(tuple(n1), tuple(n2), weight=distances[k])
                    
    
    def in_gamut(self, l, a, b):
        return Color("lab({}% {} {} / 1)".format(l, a, b)).in_gamut('srgb')


    def can_connect(self, n1, n2):
        # check if the midpoint of the line between two nodes is in collision with obstacles
        midpoint = (n1 + n2) / 2
        for i in range(len(self.obstacles)):
            if not (self.in_gamut(*midpoint) and np.linalg.norm(midpoint - self.obstacles[i]) > self.obstacle_rad):
                return False
        return True


    def find_path(self):
        # find a path from start to end through waypoints using A* and return the path

        # Iterate over the waypoints and run A* between each pair of waypoints
        path = [(0.0,0.0,0.0)]

        wps = self.waypoints.copy()

        # Add the end node
        wps.insert(0, [0.0, 0.0, 0.0])
        wps.append([100.0, 0.0, 0.0])

        for i in range(len(wps) - 1):
            path += astar_path(self.graph, tuple(wps[i]), tuple(wps[i + 1]))[1:]
        
        # # Use Dijkstra instead
        # for i in range(len(wps) - 1):
        #     path += dijkstra_path(self.graph, tuple(wps[i]), tuple(wps[i + 1]))[1:]
        
        return path


    def get_path(self):
        if self.path is None:
            self.add_edges()
            self.path = self.find_path()
            
        return self.path
        

# Test the planner
if __name__ == "__main__":
    waypoints = [[20, 0, 0], [70, 0, 0]]
    obstacles = [[50, 0, 0]]
    planner = Planning(waypoints, obstacles, 10, 1000)
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