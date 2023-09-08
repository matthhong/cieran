"""
Cieran: Designing Sequential Colormaps with a Teachable Robot
Copyright (C) 2023 Matt-Heun Hong

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""


from coloraide import Color
import numpy as np
import networkx as nx

from scipy.stats.qmc import Halton
from scipy.spatial import cKDTree as KDTree
from typing import Callable, List, Tuple, Union
import random

from collections import defaultdict

import pkg_resources

RAMPS_FILE = pkg_resources.resource_filename('cieran', 'basics/ramps.csv')
START = np.array([100, 0, 0])
END = np.array([0, 0, 0])

from coloraide import Color

class GraphEnv:

    def __init__(self, color):
        # self.color = [round(x) for x in color]
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

        # import pdb; pdb.set_trace()
        seed_centroid = self.centroids[tree.query(np.array(self.color).reshape(1, -1), k=1)[1][0]]
        diff = np.array(self.color) - np.array(seed_centroid)
        # breakpoint()

        #time this
        # import time
        # start = time.time()

        num_out_of_gamut = 0
        for ramp in self.ramps:
            translated_ramps = self.fit_ramp_to_color(ramp, self.color)

            for translated_ramp in translated_ramps: 
                new_ramp = []

                out_of_gamut = False
                # Check if every point in the ramp is in gamut
                for point in translated_ramp:
                    if not Color("lab({}% {} {} / 1)".format(*point)).in_gamut('srgb'):
                        out_of_gamut = True
                        num_out_of_gamut += 1
                i = 0
                while not out_of_gamut and i < len(translated_ramp):
                    point = translated_ramp[i]
                    i += 1

                    if np.all(point == self.color):
                        nearest_centroid = seed_centroid
                    else:

                        # Check if the point is valid
                        if point[0] > 100 or point[0] < 0:
                            continue
                        # round the point to the nearest integer
                        # point = [custom_round(x) - diff[i] for i, x in enumerate(point)]

                        # Find the nearest centroid to the point using KDTree
                        tree = KDTree(self.centroids)
                        nearest_centroid = self.centroids[tree.query(np.array(point).reshape(1, -1), k=1)[1][0]]

                        if nearest_centroid[0] > 100:
                            breakpoint()
                    # if not Color("lab({}% {} {} / 1)".format(*point)).in_gamut('srgb'):
                    #     if not out_of_gamut:
                    #         num_out_of_gamut += 1
                    #     out_of_gamut = True
                    # self.graph.add_node(tuple(point))
                    # if not out_of_gamut:
                    # breakpoint()
                    if len(new_ramp) > 0:
                        # Compare the first element between the last centroid and the current centroid
                        l_diff = new_ramp[-1][0] - nearest_centroid[0]

                        if l_diff >= 0 and np.all(point == self.color):
                            new_ramp.pop()
                        elif l_diff >= 0:
                            continue

                    new_ramp.append(nearest_centroid)

                if not out_of_gamut:
                    self.fitted_ramps.append(new_ramp)
        
        # end = time.time()
        # taken = end - start
        # print("Time to fit ramps in seconds: " + str(taken))
        # print("num ramps: " + str(len(self.fitted_ramps)))
        # print("num out of gamut: " + str(num_out_of_gamut))

        #Visualize the states in 3D LAB space
        # import matplotlib.pyplot as plt
        # from mpl_toolkits.mplot3d import Axes3D

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')

        # for ramp in self.fitted_ramps:
        #     ramp = np.array(ramp)
        #     ax.plot(ramp[:,0], ramp[:,1], ramp[:,2])

        # # Display all nodes as scatter plot, in gray
        # print("Number of nodes:", len(self.graph.nodes))

        # print("Number of original colors:" , len(self.fitted_ramps * 9))
        # for node in self.graph.nodes:
        #     # if not in any ramp
        #     if not any(np.all(node == ramp) for ramp in self.fitted_ramps):
        #         ax.scatter(node[0], node[1], node[2], c='gray', marker='o')
        
        # # Label the axes
        # ax.set_xlabel('L')
        # ax.set_ylabel('A')
        # ax.set_zlabel('B')

        # plt.show()

            
        # Add the ramp points to the graph, and edges between adjacent ramp points
        for i in range(len(self.fitted_ramps)):
            # Put start and end points into the ramp
            ramp = [END] + self.fitted_ramps[i] + [START]
            for j in range(len(ramp)-1, 0, -1):
                distance = np.linalg.norm(ramp[j] - ramp[j-1])
                if np.all(ramp[j] == END) or np.all(ramp[j-1] == START):
                    distance = 0
                self.graph.add_edge(tuple(ramp[j]), tuple(ramp[j-1]), weight=distance)
            # Reverse fitted ramp
            self.fitted_ramps[i] = ramp[::-1]

    

    def load_centroids(self):
        # centroids = np.empty((0,3))
        # with open(self.CENTROID_FILE, 'r') as f:
        #     for line in f:
        #         values = line.replace('\n','').split(' ')
        #         centroids = np.append(centroids, [np.array([float(values[0]), float(values[1]), float(values[2])])], axis=0)
        # return centroids
        dimensions = [(0,100), (-128,127), (-128,127)]
    
        # Initialize the Halton sampler
        sampler = Halton(len(dimensions), optimization='lloyd', seed=4)

        # # Generate samples (256)
        # samples = sampler.random(1973)

        # Generate samples (512)
        samples = sampler.random(4074)

        # Map the samples of size (num_samples, 3) with values between 0 and 1 to the desired dimensions across the 3 axes
        samples = np.array([dimensions[i][0] + (dimensions[i][1] - dimensions[i][0]) * samples[:, i] for i in range(len(dimensions))]).T

        points = np.empty((0,3))
        # Remove samples that are outside gamut or in collision with obstacles (circles of radius obstacle_rad)
        for i in range(len(samples)):
            if self.in_gamut(*samples[i]):
                points = np.append(points, [samples[i]], axis=0)
        # print("Number of points:", len(points))
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
                # Sort the ramp by luminance in reverse order
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


        # Compute the difference in angle between the two points with regards to roll axis
        angle = np.arctan2(closest_point[1], closest_point[2]) - np.arctan2(lab_color[1], lab_color[2])

        # Compute the rotation matrix
        rotation_matrix = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])

            # Rotate the ramp
        rotated_ramp = []
        for point in ramp:
            rotated_point = np.dot(rotation_matrix, point)
            rotated_ramp.append(rotated_point)

        # Compute the translation vector
        translation = [0, 0, 0]
        translation[0] = rotated_ramp[min_distance_index][0] - lab_color[0]
        translation[1] = rotated_ramp[min_distance_index][1] - lab_color[1]
        translation[2] = rotated_ramp[min_distance_index][2] - lab_color[2]

        difference_vector = [0, 0, 0]
        new_start_color = [0, 0, 0]
        for i in range(3):
            difference_vector[i] = rotated_ramp[min_distance_index][i] - lab_color[i]
            new_start_color[i] = rotated_ramp[0][i] - difference_vector[i]

        new_ramp1 = self.translate_curve(rotated_ramp, new_start_color)

        # Compute the translation vector
        translation = [0, 0, 0]
        translation[0] = ramp[min_distance_index][0] - lab_color[0]
        translation[1] = ramp[min_distance_index][1] - lab_color[1]
        translation[2] = ramp[min_distance_index][2] - lab_color[2]

        difference_vector = [0, 0, 0]
        new_start_color = [0, 0, 0]
        for i in range(3):
            difference_vector[i] = ramp[min_distance_index][i] - lab_color[i]
            new_start_color[i] = ramp[0][i] - difference_vector[i]

        new_ramp2 = self.translate_curve(ramp, new_start_color)

        return [new_ramp1, new_ramp2]


    def translate_curve(self, curve, starting_point):
        translation_vector_x = curve[0][0] - starting_point[0]
        translation_vector_y = curve[0][1] - starting_point[1]
        translation_vector_z = curve[0][2] - starting_point[2]

        translated_curve = []

        for point in curve:
            translated_point = []
            translated_point.append(round(point[0] - translation_vector_x, 2))
            translated_point.append(round(point[1] - translation_vector_y, 2))
            translated_point.append(round(point[2] - translation_vector_z, 2))

            translated_curve.append(translated_point)

        return translated_curve


class Environment(GraphEnv):

    def __init__(self, color, source=(100,0,0), target=(0,0,0), weight='weight', epsilon=0.1, feature_func=None):
        super().__init__(color)

        self.state_actions = {}
        for node in self.graph.nodes():
            self.state_actions[node] = list(self.graph.neighbors(node))
        
        self.source = source
        self.target = target
        self.weight = weight
        self.epsilon = epsilon

        self.discount= 1
        self.lr = 0.1
        self.Q = defaultdict(float)
        self.N = defaultdict(int)
        self.Q.default_factory = lambda: 100.0
        self.N.default_factory = lambda: 1

        self.feature_func = feature_func
        
        self.set_reward_weights(None)

        self.reset()

    def run(self):
        while not self.terminal(self.state):
            self.choose_action(self.state)
            self.trajectory.append(self.next_state)

            reward = self.reward
            self.Q[(self.state, self.action)] = self.state_action_value + 0.1 * (reward + self.temporal_difference)

            self.N[(self.state, self.action)] += 1
            self.total_reward += reward

            self.set_state(self.next_state)
        return self.total_reward

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
        self.action = None
        self.next_state = None
        self.total_reward = 0

    def set_reward_weights(self, weights):
        self.reward_weights = weights
        self.best_policy = None
        self.best_reward = float("-inf")

    @property
    def state_action_value(self):
        return self.Q[(self.state, self.action)]

    @property
    def reward(self):
        rew = -0.01
        if self.terminal(self.next_state):
            rew += 10
            rew += np.dot(self.reward_weights, self.feature_func(self.trajectory))
        return rew

    @property
    def temporal_difference(self):
        return self.discount * self.state_value(self.next_state) - self.state_action_value

    def terminal(self, state):
        if state is None:
            breakpoint()
        return len(self.state_actions[state]) == 0

    def state_value(self, state):
        if self.terminal(state):
            return 0
        else:
            return self.max_Q(state)[0]

    def max_Q(self, state):
        # Find the maximum value in Q for a given (node, neighbor)
        max_q = -float('inf')
        max_neighbor = None
        neighbors = list(self.graph.neighbors(state))
        # Shuffle neighbors in-place
        random.shuffle(neighbors)

        for neighbor in neighbors:
            q = self.Q[(state, neighbor)]
            if q > max_q:
                max_q = q
                max_neighbor = neighbor
        return max_q, max_neighbor
    
    def accel(self, accessor):
        # Get last three points
        sub_trajectory = self.trajectory[-3:]

        slopes = []
        for i in range(len(sub_trajectory) - 1):
            slopes.append((sub_trajectory[i+1][accessor] - sub_trajectory[i][accessor]) / (sub_trajectory[i+1][0] - sub_trajectory[i][0]))

        accel = 0
        for i in range(len(slopes) - 1):
            accel = slopes[i+1] - slopes[i]

        return accel/127
    
    def max_accel(self, trajectory, accessor):

        slopes = []
        for i in range(len(trajectory) - 1):
            slopes.append((trajectory[i+1][accessor] - trajectory[i][accessor]) / (trajectory[i+1][0] - trajectory[i][0]))

        accels = []
        for i in range(len(slopes) - 1):
            accels.append(slopes[i+1] - slopes[i])

        max_accel = 0
        try:
            max_accel = abs(max(accels))
        except ValueError:
            pass

        return max_accel/127

    def choose_action(self, state):
        self.action = self.greedy_epsilon(state)

        self.next_state = self.action
        if random.random() <= 0.05: # Transition matrix
            self.next_state = random.choice(self.state_actions[state])

    def choose_random_action(self, state):
        self.action = random.choice(self.state_actions[state])
        self.next_state = self.action

    def choose_optimal_action(self, state):
        self.action = self.max_Q(state)[1]
        self.next_state = self.action

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
        q_values = [self.Q[(state, neighbor)] for neighbor in neighbors]
        probs = [np.exp(q) / sum(np.exp(q_values)) for q in q_values]
        return random.choices(neighbors, weights=probs)[0]

    def get_best_path(self):
        # Get path that maximizes Q at each step
        self.reset()
        total_reward = 0
        while not self.terminal(self.state):
            self.choose_optimal_action(self.state)
            self.trajectory.append(self.next_state)
            total_reward += self.reward
            self.set_state(self.next_state)
        return self.trajectory, total_reward



# class Environment(QLearning):

#     def __init__(self, color, source=tuple(START), target=tuple(END), weight='weight', epsilon=0.1, lambd=0.9, feature_func=None):
#         super().__init__(color, source, target, weight, epsilon, feature_func=feature_func)
#         self.decay = lambd
#         self.render_exists = False
#         self.close_exists = False
#         self.feature_func = feature_func

#     def run(self):
#         eligibility = {}

#         while self.state != self.target:
#             self.choose_action(self.state)
            
#             eligibility[(self.state, self.next_state)] = eligibility.get((self.state, self.next_state), 0) + 1

#             for a, b in self.graph.edges:
#                 eligibility[(a,b)] = eligibility.get((a,b), 0) * self.decay * self.discount
#                 self.Q[(a,b)] = self.Q[(a,b), 0) + self.lr * self.temporal_difference * eligibility[(a,b)]

#             self.state = self.next_state




if __name__ == "__main__":
    states = GraphEnv([26.6128, 37.85, -44.51])

    # Visualize the states in 3D LAB space
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import networkx as nx

    # Average number of neighbors per node
    graph = states.graph
    print("Average number of neighbors per node:", sum([len([neighbor for neighbor in graph.neighbors(node)]) for node in graph.nodes]) / len(graph.nodes))

    # Visualize the degree distribution
    degrees = [len([neighbor for neighbor in graph.neighbors(node)]) for node in graph.nodes]
    plt.hist(degrees, bins=range(1, max(degrees) + 1))
    plt.xlabel('Degree')
    plt.ylabel('Number of nodes')
    plt.show()

    # Reset figure
    plt.clf()

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