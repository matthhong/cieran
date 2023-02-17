from coloraide import Color
import numpy as np
import networkx as nx

from scipy.stats.qmc import Halton
from typing import Callable, List, Tuple, Union
import random

import pkg_resources


RAMPS_FILE = pkg_resources.resource_filename('cieran', 'basics/ramps.csv')
START = np.array([100, 0, 0])
END = np.array([0, 0, 0])

# from typing import List, Tuple, Union
# import time
# import numpy as np
# from moviepy.editor import VideoFileClip
# import networkx as nx
from geomdl import fitting
from geomdl.operations import tangent
from coloraide import Color

from queue import PriorityQueue

# KDTreeNode: points, depth -> KDTreeNode
class KDTreeNode:
    def __init__(self, points, depth = 0):
        self.points = points
        self.left = None
        self.right = None
        self.mid = None

        self.split_axis = depth % 3

        # sort points by split_axis
        points = points[points[:, self.split_axis].argsort()]

        if len(points) > 1:
            # split points into left and right
            mid = len(points) // 2
            self.mid = points[mid]
            self.left = KDTreeNode(points[:mid], depth + 1)
            self.right = KDTreeNode(points[mid:], depth + 1)


# KDTree: points -> KDTree
class KDTree:
    def __init__(self, points, constrained_axis=None, direction=1, obstacles=None, obstacle_cost_multiplier=1):
        self.constrained_axis = constrained_axis
        self.root = KDTreeNode(points)
        self.direction = direction
        self.obstacles = obstacles
        self.obstacle_cost_multiplier = obstacle_cost_multiplier

    # query: point, k -> (distances, points)
    def query(self, point, k):
        # Create a priority queue
        queue = PriorityQueue(maxsize=k)

        # Recursively search the tree
        self._query(self.root, point, k, queue)
        # Return the k nearest neighbors
        try:
            distances, points = np.array(queue.queue).T
        except:
            raise Exception("No points found in tree")
        return -distances, np.stack(points)


    # _query: node, point, k, queue -> None
    def _query(self, node, point, k, queue):
        # If the node is empty, return
        if node is None:
            return

        # If the node is a leaf, check the points in the node
        if node.left is None and node.right is None:
            for p in node.points:

                # Custom skip
                if self.constrained_axis is not None:
                    # Check if current point has a higher value than the query point on the constrained axis
                    if not (self.direction * (p[self.constrained_axis] - point[self.constrained_axis]) > 0):
                        continue

                # Compute the distance from the point to the query point
                dist = np.linalg.norm(point - p)

                # Compute the average distance from the point to obstacles
                # obstacle_dist = 0
                # for obstacle in self.obstacles:
                #     obstacle_dist += np.linalg.norm(p - obstacle)
                # # breakpoint()
                
                # if len(self.obstacles) > 0:
                #     obstacle_dist /= len(self.obstacles) * self.obstacle_cost_multiplier

                # # Cost is the distance of moving to the point, but there is reward for moving away from obstacles
                # dist -= obstacle_dist

                # If the queue is not full, add the point
                if queue.qsize() < k:
                    queue.put((-dist, p))

                # Otherwise, check if the point is closer than the furthest point in the queue
                else:
                    # If the point is closer, remove the furthest point and add the new point
                    if dist < -queue.queue[0][0]:
                        queue.get()
                        queue.put((-dist, p))

            return
        
        # If the node is not a leaf, check which side of the splitting plane the point is on
        # Compute the distance from the point to the splitting plane
        dist = point[node.split_axis] - node.mid[node.split_axis]

        # Search the side of the splitting plane that is closest to the point
        if dist < 0:
            if node.left is not None:
                self._query(node.left, point, k, queue)
            if node.right is not None:
                self._query(node.right, point, k, queue)
        else:
            if node.right is not None:
                self._query(node.right, point, k, queue)
            if node.left is not None:
                self._query(node.left, point, k, queue)


    

class GraphEnv:

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
                    if l_diff >= 0:
                        continue
                new_ramp.append(nearest_centroid)
                last_centroid = nearest_centroid
        
            self.fitted_ramps.append(new_ramp)
        
        print("num ramps: " + str(len(self.fitted_ramps)))
        print("num out of gamut: " + str(num_out_of_gamut))

        # Visualize the states in 3D LAB space
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for ramp in self.fitted_ramps:
            ramp = np.array(ramp)
            ax.plot(ramp[:,0], ramp[:,1], ramp[:,2])

        # Display all nodes as scatter plot, in gray
        print("Number of nodes:", len(self.graph.nodes))

        print("Number of original colors:" , len(self.fitted_ramps * 9))
        for node in self.graph.nodes:
            # if not in any ramp
            if not any(np.all(node == ramp) for ramp in self.fitted_ramps):
                ax.scatter(node[0], node[1], node[2], c='gray', marker='o')
        
        # Label the axes
        ax.set_xlabel('L')
        ax.set_ylabel('A')
        ax.set_zlabel('B')

        plt.show()

            
        # Add the ramp points to the graph, and edges between adjacent ramp points
        for i, ramp in enumerate(self.fitted_ramps):
            # Put start and end points into the ramp
            self.fitted_ramps[i] = [END] + ramp + [START]
            ramp = self.fitted_ramps[i]
            for i in range(len(ramp)-1, 0, -1):
                distance = np.linalg.norm(ramp[i] - ramp[i-1])
                if np.all(ramp[i] == END) or np.all(ramp[i-1] == START):
                    distance = 0
                self.graph.add_edge(tuple(ramp[i]), tuple(ramp[i-1]), weight=distance)

    

    def load_centroids(self):
        # centroids = np.empty((0,3))
        # with open(self.CENTROID_FILE, 'r') as f:
        #     for line in f:
        #         values = line.replace('\n','').split(' ')
        #         centroids = np.append(centroids, [np.array([float(values[0]), float(values[1]), float(values[2])])], axis=0)
        # return centroids
        dimensions = [(0,100), (-128,127), (-128,127)]
    
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
        with open(RAMPS_FILE, 'r') as f:
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


class QLearning(GraphEnv):

    def __init__(self, color, source, target, weight='weight', epsilon=0.1, feature_func=None):
        super().__init__(color)

        self.state_actions = {}
        for node in self.graph.nodes():
            self.state_actions[node] = list(self.graph.neighbors(node))
        
        self.source = source
        self.target = target
        self.weight = weight
        self.epsilon = epsilon
        self.Q = {}

        self.lr = 1
        self.discount = 1
        self.reward_weights = np.array([-1,-1,-1])

        self.reset()

    def run(self):
        while not self.terminal(self.state):
            self.choose_action(self.state)
            self.Q[(self.state, self.next_state)] = self.state_action_value + self.lr * self.temporal_difference
            self.set_state(self.next_state)

    def random_walk(self):
        while not self.terminal(self.state):
            self.choose_random_action(self.state)
            self.set_state(self.next_state)
        return self.trajectory

    def set_state(self, state):
        self.state = state

    def reset(self):
        self.state = self.source
        self.trajectory = [self.state]
        self.next_state = None

    @property
    def state_action_value(self):
        return self.Q.get((self.state, self.next_state), 0)

    @property
    def reward(self):
        # try:
        #     # return -self.graph[self.state][self.next_state][self.weight]
        # except:
        #     breakpoint()
        return 0

    @property
    def temporal_difference(self):
        return self.reward + self.discount * self.utility(self.next_state) - self.state_action_value

    def terminal(self, state):
        if state is None:
            breakpoint()
        return len(self.state_actions[state]) == 0

    def utility(self, state):
        if self.terminal(state):
            # Dot product of reward weights and feature vector
            return np.dot(self.reward_weights, self.feature_func(self.trajectory[1:-1]))
        else:
            return self.max_Q(state)[0]

    def max_Q(self, state):
        # Find the maximum value in Q for a given (node, neighbor)
        max_q = -float('inf')
        max_neighbor = None
        for neighbor in self.graph.neighbors(state):
            q = self.Q.get((state, neighbor), 0)
            if q > max_q:
                max_q = q
                max_neighbor = neighbor
        return max_q, max_neighbor

    def choose_action(self, state):
        # self.next_state = self.greedy_epsilon(state)
        self.next_state = self.softmax(state)
        self.trajectory.append(self.next_state)

    def choose_random_action(self, state):
        self.next_state = random.choice(self.state_actions[state])
        self.trajectory.append(self.next_state)

    def greedy_epsilon(self, state):
        # Choose a random neighbor
        if random.random() < self.epsilon:
            return random.choice(self.state_actions[state])

        # Choose the neighbor with the highest Q value
        max_neighbor = self.max_Q(state)[1]
        
        return max_neighbor

    def softmax(self, state):
        # Choose a neighbor with a probability proportional to its Q value
        neighbors = self.state_actions[state]
        q_values = [self.Q.get((state, neighbor), 0) for neighbor in neighbors]
        probs = [np.exp(q) / sum(np.exp(q_values)) for q in q_values]
        return random.choices(neighbors, weights=probs)[0]

    def get_best_path(self):
        # Get path that maximizes Q at each step
        path = [self.source]
        state = self.source
        while not self.terminal(state):
            state = self.max_Q(state)[1]
            path.append(state)
        return path


class Environment(QLearning):

    def __init__(self, color, source=tuple(START), target=tuple(END), weight='weight', epsilon=0.1, lambd=0.9, feature_func=None):
        super().__init__(color, source, target, weight, epsilon, feature_func=feature_func)
        self.decay = lambd
        self.render_exists = False
        self.close_exists = False
        self.feature_func = feature_func

    def run(self):
        eligibility = {}

        while self.state != self.target:
            self.choose_action(self.state)
            
            eligibility[(self.state, self.next_state)] = eligibility.get((self.state, self.next_state), 0) + 1

            for a, b in self.graph.edges:
                eligibility[(a,b)] = eligibility.get((a,b), 0) * self.decay * self.discount
                self.Q[(a,b)] = self.Q.get((a,b), 0) + self.lr * self.temporal_difference * eligibility[(a,b)]

            self.state = self.next_state




if __name__ == "__main__":
    states = GraphEnv([26.6128, 37.85, -44.51])

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