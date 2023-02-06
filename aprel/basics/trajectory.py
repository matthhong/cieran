"""Modules that are related to environment trajectories."""

from typing import List, Tuple, Union
import time
import numpy as np
# from moviepy.editor import VideoFileClip
import networkx as nx


def path_length(path: List[np.array], env) -> float:
    # Get the edge weights of the path, where an edge is an array of two nodes
    edge_weights = [env.graph.get_edge_data(tuple(path[i]), tuple(path[i+1]))['weight'] for i in range(len(path) - 1)]
    return sum(edge_weights) / env.longest_path_length

def path_a_range(path: List[np.array]) -> float:
    return (max([point[1] for point in path]) - min([point[1] for point in path])) / 255

def path_b_range(path: List[np.array]) -> float:
    return (max([point[2] for point in path]) - min([point[2] for point in path])) / 255


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
    def __init__(self, env, trajectory: List[np.array], clip_path: str = None):
        # Remove first and last points of trajectory
        self.trajectory = trajectory[1:-1]
        self.features = []
        self.features.append(path_length(self.trajectory, env))
        self.features.append(path_a_range(self.trajectory))
        self.features.append(path_b_range(self.trajectory))
        self.features = np.array(self.features)

        self.clip_path = clip_path
        
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
