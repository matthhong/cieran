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


Modules that are related to environment trajectories.
"""

from typing import List, Tuple, Union
import numpy as np
from coloraide import Color


from matplotlib.colors import ListedColormap


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
    def __init__(self, env, trajectory: List[np.array], clip_path: str = None, draw=None):
        # Remove first and last points of trajectory
        self.trajectory = trajectory
        self._curve = None
        self.clip_path = clip_path
        self._ramp = None
        self.env = env
        self._points = None
        self._draw = draw

        self.features = None
        if env:
            self.features = env.feature_func(trajectory)

        # self.interpolate()

    def __getitem__(self, t: int) -> Tuple[np.array, np.array]:
        """Returns the state-action pair at time step t of the trajectory."""
        return self.trajectory[t]
        
    @property
    def length(self) -> int:
        """The length of the trajectory, i.e., the number of time steps in the trajectory."""
        return len(self.trajectory)
        
    def draw(self):
        if self._draw:
            self._draw(self.ramp)

    @property
    def ramp(self):
        controls = [Color("lab({}% {} {} / 1)".format(*p)) for p in self.trajectory]
        self._curve = Color.interpolate(controls, method='bspline')
    
        t = np.linspace(0, 1, 256)
        at = np.linspace(0, 1, 256)
        # points = self._curve.evaluate_list(at)
        points = [self._curve(index) for index in at]

        # Get the arc length of the ramp at each point using distance function
        arc_lengths = [0]
        for i in range(1, len(points)):
            arc_lengths.append(arc_lengths[i-1] + self.distance(points[i-1], points[i]))

        # Normalize the arc lengths
        arc_lengths = np.array(arc_lengths) / arc_lengths[-1]

        # Invert the arc lengths to get the parameterization
        at_t = np.interp(at, arc_lengths, t)
        self._points = [self._curve(index) for index in at_t]

        # Filter all points where the first (L*) value is less than 30
        self._points = [p for p in self._points if p._coords[0] > 30 and p._coords[0] < 100]

        # Get the points from the ramp using the parameterization
        # points = self._curve.evaluate_list(at_t)
        colors = [p.convert('srgb').to_string(hex=True) for p in self._points]
        # colors = [self._curve(index).convert('srgb').to_string(hex=True) for index in at_t]

        # convert to ListedColormap
        self._ramp = ListedColormap(colors)

        return self._ramp
    
    @property
    def in_gamut(self):
        if self._curve is None:
            controls = [Color("lab({}% {} {} / 1)".format(*p)) for p in self.trajectory]
            self._curve = Color.interpolate(controls, method='bspline')

        t = np.linspace(0, 1, 256)
        at = np.linspace(0, 1, 256)
        # points = self._curve.evaluate_list(at)
        points = [self._curve(index) for index in at]

        in_gamut = True
        for point in points:
            color = Color("lab({}% {} {} / 1)".format(*point))
            if not color.in_gamut('srgb'):
                # print("Out of gamut: {}".format(color))
                in_gamut = False
                break
        return in_gamut
    
    def plot_all(self):
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        fig = plt.figure(figsize=(15, 5))

        # Create a 3x3 grid of subplots using GridSpec
        gs = gridspec.GridSpec(3, 3,
                            width_ratios=[1, 1, 1],
                            height_ratios=[1, 1, 1]
                            )
        
        # Remove padding
        fig.subplots_adjust(wspace=0, hspace=0)

        ax1 = fig.add_subplot(gs[0:, 0])

        # Create 3 subplots stacked vertically in the second column
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 1])
        ax4 = fig.add_subplot(gs[2, 1])

        ax5 = fig.add_subplot(gs[0:, 2])

        fig.suptitle("Cieran's plot")

        # Compute distances between each point in self.ramper.path and plot the distances with a line chart
        distances = np.array([self.distance_lab(self._points[i], self._points[i-1]) for i in range(1, len(self._points))])
        distances = len(distances) * distances
        arclength = np.sum(distances)   
        rmse = np.std(distances)
        ax5.plot(distances)

        ax5.set_title("Flatness of perceptual differences: %0.2f%%"
                % (100 - (100 * rmse / arclength)))
        ax5.set_xlabel("Point")
        ax5.set_ylabel("Distance")

        # Set y axes from 0 to max distance
        ax5.set_ylim(0, max(distances)*2)
        
        # Plot the L* values ranging from 0 to 100
        l_values = np.array([self._points[i][0] for i in range(0, len(self._points))])
        ax2.plot(l_values)

        ax2.set_title("L* values")
        ax2.set_xlabel("Point")
        ax2.set_ylabel("L* value")

        # Set y axes from 0 to 100
        ax2.set_ylim(0, 100)

        a_values = np.array([self._points[i][1] for i in range(0, len(self._points))])
        b_values = np.array([self._points[i][2] for i in range(0, len(self._points))])

        # Convert a and b values to polar coordinates
        c_values = np.sqrt(a_values**2 + b_values**2)
        h_values = np.arctan2(b_values, a_values)

        # Plot the c values ranging from 0 to 150
        ax3.plot(c_values)

        ax3.set_title("c* values")
        ax3.set_xlabel("Point")
        ax3.set_ylabel("c* value")

        # Set y axes from -150 to 150
        ax3.set_ylim(0, 100)

        # Plot the h values
        ax4.plot(h_values)

        ax4.set_title("h* values")
        ax4.set_xlabel("Point")
        ax4.set_ylabel("h* value")

        # Set y axes from  -pi to pi
        ax4.set_ylim(-np.pi, np.pi)

        # Plot the interpolated curve in matplotlib, projecting into 2D, showing only y and z
        # obstacles in red

        # waypoints in their color values in cielab
        waypoints_proj = np.array(self.trajectory)[:, [1, 2]]
        ax1.scatter(
            *zip(*waypoints_proj), 
            c=[Color("lab({}% {} {} / 1)".format(*centroid)).convert('srgb')[:3] for centroid in self.trajectory]
        )

        # path in green
        path_proj = np.array(self._points)[:, [1, 2]]
        ax1.plot(*zip(*path_proj), c='green')

        ax1.set_title("Interpolated color ramp")
        ax1.set_xlabel("a*")
        ax1.set_ylabel("b*")

        # Set x and y axes from -128 to 128
        ax1.set_xlim(-100, 100) 
        ax1.set_ylim(-100, 100)

        plt.savefig("test.svg")

        plt.show()

    def lab_to_rgb(self, lab):
        # Convert a CIELAB value to an RGB value
        return Color("lab({}% {} {} / 1)".format(*lab)).convert("srgb")

    def distance_lab(self, p1, p2):
        return Color("lab({}% {} {} / 1)".format(*p1)).delta_e(Color("lab({}% {} {} / 1)".format(*p2)), method='2000')

    def distance(self, c1, c2):
        return c1.delta_e(c2, method='2000')



"""
MIT License

Copyright (c) 2021 Stanford Intelligent and Interactive Autonomous Systems Group

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


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

    def display(self, shuffle=True):
        # Create two columns with jupyter widgets, where the first coulmn contains four rows
        # and the second column contwins two rows, the first row taking up 1/4 of the second column,
        # and the second row taking up 3/4 of the second column

        pass
