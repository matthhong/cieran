from coloraide import Color
import numpy as np
import networkx as nx
from KDTree import KDTree
from scipy.stats.qmc import Halton
from typing import Callable, List, Tuple, Union
import random

import pkg_resources

RAMPS_FILE = pkg_resources.resource_filename('aprel', 'basics/ramps.csv')
START = np.array([100, 0, 0])
END = np.array([0, 0, 0])

# from typing import List, Tuple, Union
# import time
# import numpy as np
# from moviepy.editor import VideoFileClip
# import networkx as nx
from geomdl import fitting, operations
from coloraide import Color

"""Modules that are related to environment trajectories."""

class Trajectory:
    """
    A class for keeping trajectories that consist of a sequence of state-action pairs,
    the features and a clip path that keeps a video visualization of the trajectory.
    
    This class supports indexing, such that t^th index returns the state-action pair at time
    step t. However, indices cannot be assigned, i.e., a specific state-action pair cannot be
    changed, because that would enable infeasible trajectories.
    
    Parameters:
        env (Environment): The environment object that generated this trajectory.
        trajectory (List[Tuple[numpy.array, numpy.array]]): The sequence of state-action pairs.
        clip_path (str): The path to the video clip that keeps the visualization of the trajectory.
    
    Attributes:
        trajectory (List[Tuple[numpy.array, numpy.array]]): The sequence of state-action pairs.
        features (numpy.array): Features of the trajectory.
        clip_path (str): The path to the video clip that keeps the visualization of the trajectory.
    """
    def __init__(self, trajectory: List[np.array], clip_path: str = None):
        # Remove first and last points of trajectory
        self.trajectory = trajectory[1:-1]
        self.curve = None
        self.clip_path = clip_path
        self._features = None

    def __getitem__(self, t: int) -> Tuple[np.array, np.array]:
        """Returns the state-action pair at time step t of the trajectory."""
        return self.trajectory[t]
        
    @property
    def length(self) -> int:
        """The length of the trajectory, i.e., the number of time steps in the trajectory."""
        return len(self.trajectory)
        
    def visualize(self):
        """
        Visualizes the trajectory with a video if the clip exists. Otherwise, prints the trajectory information.
        
        :Note: FPS is fixed at 25 for video visualizations.
        """
        if self.clip_path is not None:
            # clip = VideoFileClip(self.clip_path)
            # clip.preview(fps=30)
            # clip.close()
            pass
        else:
            print('Headless mode is on. Printing the trajectory information.')
            #print(self.trajectory)
            print('Features for this trajectory are: ' + str(self.features))

    def interpolate(self):
        # Interpolate the ramp
        self.curve = fitting.interpolate_curve(self.control_points, 3, centripetal=True)

    def ramp(self):
        t = np.linspace(0, 1, 1000)
        at = np.linspace(0, 1, 1000)
        points = self.curve.evaluate_list(at)

        # Get the arc length of the ramp at each point using distance function
        arc_lengths = [0]
        for i in range(1, len(points)):
            arc_lengths.append(arc_lengths[i-1] + self.distance(points[i-1], points[i]))

        # Normalize the arc lengths
        arc_lengths = np.array(arc_lengths) / arc_lengths[-1]

        # Invert the arc lengths to get the parameterization
        at_t = np.interp(at, arc_lengths, t)

        # Get the points from the ramp using the parameterization
        self.ramp = self.curve.evaluate_list(at_t)

    def distance(self, p1, p2):
        return Color("lab({}% {} {} / 1)".format(*p1)).delta_e(Color("lab({}% {} {} / 1)".format(*p2)), method='2000')

    @property
    def features(self):
        if self._features is None:
            self.interpolate()
            self._features = []
            self._features.extend(self.a_range())
            self._features.extend(self.b_range())
            self._features.append(self.max_curvature())
            self._features = np.array(self._features)
        return self._features

    def arc_length(self, path: List[np.array], env) -> float:
        # Get the edge weights of the path, where an edge is an array of two nodes
        # edge_weights = [env.graph.get_edge_data(tuple(path[i]), tuple(path[i+1]))['weight'] for i in range(len(path) - 1)]
        # return sum(edge_weights) / env.longest_path_length
        pass

    def a_range(self) -> float:
        #min is either 0 or the actual min
        min = 0
        if min([point[1] for point in self.curve]) < 0:
            min = min([point[1] for point in self.curve])
        return [abs(min/255)/128, max([point[1] for point in self.curve])/127]

    def b_range(self) -> float:
        min = 0
        if min([point[2] for point in self.curve]) < 0:
            min = min([point[2] for point in self.curve])
        return [abs(min/255)/128, max([point[2] for point in self.curve])/127]

    def max_curvature(self) -> float:
        # Tangent vectors
        curvetan = []
        delta = 0.01

        # For each delta
        for u in np.arange(0, 1, delta):
            curvetan.append(operations.tangent(self.curve, u, normalize=True))
        
        # Get the derivative of the tangent vectors
        curvetan = np.array(curvetan)
        curvetan = np.diff(curvetan, axis=0)

        # Get the magnitude of the derivative
        curvetan = np.linalg.norm(curvetan, axis=1)

        curvatures = curvetan / (delta ** 2)

        return max(curvatures)




class TrajectorySet:
    """
    A class for keeping a set of trajectories, i.e. :class:`.Trajectory` objects.
    
    This class supports indexing, such that t^th index returns the t^th trajectory in the set.
    Similarly, t^th trajectory in the set can be replaced with a new trajectory using indexing.
    Only for reading trajectories with indexing, list indices are also allowed.
    
    Parameters:
        trajectories (List[Trajectory]): The list of trajectories to be stored in the set.
    
    Attributes:
        trajectories (List[Trajectory]): The list of trajectories in the set.
        features_matrix (numpy.array): n x d array of features where each row consists of the d features
            of the corresponding trajectory.
    """
    def __init__(self, trajectories: List[Trajectory]):
        self.trajectories = trajectories
        self.features_matrix = np.array([trajectory.features for trajectory in self.trajectories])

    def __getitem__(self, idx: Union[int, List[int], np.array]):
        if isinstance(idx, list) or type(idx).__module__ == np.__name__:
            return TrajectorySet([self.trajectories[i] for i in idx])
        return self.trajectories[idx]
        
    def __setitem__(self, idx: int, new_trajectory: Trajectory):
        self.trajectories[idx] = new_trajectory

    @property
    def size(self) -> int:
        """The number of trajectories in the set."""
        return len(self.trajectories)
        
    def append(self, new_trajectory: Trajectory):
        """Appends a new trajectory to the set."""
        self.trajectories.append(new_trajectory)
        if self.size == 1:
            self.features_matrix = new_trajectory.features.reshape((1,-1))
        else:
            self.features_matrix = np.vstack((self.features_matrix, new_trajectory.features))

    

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

        # breakpoint()
        # print(max_angular_velocity(nx.dag_longest_path(self.graph)))
        # self.longest_path_length = nx.dag_longest_path_length(self.graph)
        self.longest_trajectory = Trajectory(nx.dag_longest_path(self.graph))
    

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

    def __init__(self, color, source, target, weight='weight', epsilon=0.1):
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
            return np.dot(self.reward_weights, self.feature_vec)
        else:
            return self.max_Q(state)[0]

    @property
    def feature_vec(self):
        trajectory = Trajectory(self.trajectory[1:-1])
        return trajectory.features

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
        self.next_state = self.greedy_epsilon(state)
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

    def get_best_path(self):
        # Get path that maximizes Q at each step
        path = [self.source]
        state = self.source
        while not self.terminal(state):
            state = self.max_Q(state)[1]
            path.append(state)
        return path


class Environment(QLearning):

    def __init__(self, color, source=tuple(START), target=tuple(END), weight='weight', epsilon=0.1, lambd=0.9):
        super().__init__(color, source, target, weight, epsilon)
        self.decay = lambd
        self.render_exists = False
        self.close_exists = False

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