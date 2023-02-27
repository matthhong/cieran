"""Modules that are related to environment trajectories."""

from typing import List, Tuple, Union
import time
import numpy as np
# from moviepy.editor import VideoFileClip
import networkx as nx
from geomdl import fitting, operations
from coloraide import Color

from matplotlib.colors import ListedColormap

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
    def __init__(self, env, trajectory: List[np.array], clip_path: str = None):
        # Remove first and last points of trajectory
        self.trajectory = trajectory
        self._curve = None
        self.clip_path = clip_path
        self._ramp = None
        self.env = env

        self.features = None
        if env:
            self.features = env.feature_func(trajectory)

        self.interpolate()

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

    # def get_curve(self, num_points):
    #     t = np.linspace(0, 1, num_points)
    #     at = np.linspace(0, 1, num_points)
    #     points = self._curve.evaluate_list(at)    
    #     return points

    def interpolate(self):
        # Interpolate the ramp
        try:
            if len(self.trajectory) > 3:
                self._curve = fitting.interpolate_curve(self.trajectory, 3, centripetal=True)
            elif len(self.trajectory) == 3:
                self._curve = fitting.interpolate_curve(self.trajectory, 2, centripetal=True)
            else:
                self._curve = fitting.interpolate_curve(self.trajectory, 1, centripetal=True)
        except:
            breakpoint()
        # try:
        #     controls = [Color("lab({}% {} {} / 1)".format(*p)) for p in self.trajectory]
        #     self._curve = Color.interpolate(controls, method='monotone')
        # except:
        #     pass

    @property
    def ramp(self):
        if self._ramp is None:
            controls = [Color("lab({}% {} {} / 1)".format(*p)) for p in self.trajectory]
            # self._curve = Color.interpolate(controls, method='natural')
        
            t = np.linspace(0, 1, 256)
            at = np.linspace(0, 1, 256)
            points = self._curve.evaluate_list(at)
            # points = [self._curve(index) for index in at]

            # Get the arc length of the ramp at each point using distance function
            arc_lengths = [0]
            for i in range(1, len(points)):
                arc_lengths.append(arc_lengths[i-1] + self.distance_lab(points[i-1], points[i]))

            # Normalize the arc lengths
            arc_lengths = np.array(arc_lengths) / arc_lengths[-1]

            # Invert the arc lengths to get the parameterization
            at_t = np.interp(at, arc_lengths, t)

            # Get the points from the ramp using the parameterization
            points = self._curve.evaluate_list(at_t)
            colors = [self.lab_to_rgb(p).to_string(hex=True) for p in points]
            # colors = [self._curve(index).convert('srgb').to_string(hex=True) for index in at_t]

            # convert to ListedColormap
            self._ramp = ListedColormap(colors)

        return self._ramp

    def lab_to_rgb(self, lab):
        # Convert a CIELAB value to an RGB value
        return Color("lab({}% {} {} / 1)".format(*lab)).convert("srgb")

    def distance_lab(self, p1, p2):
        return Color("lab({}% {} {} / 1)".format(*p1)).delta_e(Color("lab({}% {} {} / 1)".format(*p2)), method='2000')

    def distance(self, c1, c2):
        return c1.delta_e(c2, method='2000')



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
