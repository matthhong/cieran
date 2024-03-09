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


from cieran.basics.environment import Environment
from cieran.basics.trajectory import Trajectory, TrajectorySet
from cieran.basics.features import feature_func

from cieran.learning.data_types import WeakComparisonQuery, WeakComparison
from cieran.learning.user_models import SoftmaxUser, HumanUser
from cieran.learning.belief_models import SamplingBasedBelief

from cieran.querying.query_optimizer import QueryOptimizerDiscreteTrajectorySet


from coloraide import Color


import numpy as np
import hashlib

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.cm import ScalarMappable

import ipywidgets as widgets
from IPython.display import display



class Cieran:
    def __init__(self, draw):
        """Cieran is a module for designing sequential colormaps from pairwise choices.

        Examples:
            >>> from cieran import Cieran
            >>> import matplotlib.pyplot as plt
            >>> import numpy as np
            >>> data2d = np.random.rand(10, 10)
            >>> def draw(cmap): plt.imshow(data2d, cmap=cmap); plt.show()
            >>> cieran = Cieran(draw)

        Args:
            draw (function): A callback function that takes a ListedColormap as input and displays a plot. 
                
        """
            
        self.draw = draw
        self._search_result = None

        self.hex_color = None

        self._env = None
        self._trajectories = None
        self._query_optimizer = None
        self._user_model = None
        self._belief_model = None
        self._query = None
        self._ranked_results = None
        self.candidates = []
        self.cmap = None


    def set_color(self, color):
        """Sets the color that should be included as a control point in output colormap trajectories.

        Examples:
            >>> cieran.set_color([50, 0, 0])
            >>> cieran.set_color("#ff0000")

        Args:
            color (str or list): A color that should be included as a control point in output colormap trajectories. 
                If a list, it should be a 3-element list of L*, a*, and b* values. 
                If a string, it should be a hexadecimal string.
        """
        if isinstance(color, str):
            self.color = Color(color).convert('lab')._coords[:-1]
            self.hex_color = color
        elif isinstance(color, list) or isinstance(color, np.ndarray):
            self.color = color
            self.hex_color = Color("lab({}% {} {} / 1)".format(*color)).convert('srgb').to_string(hex=True)

        self.block += 1
        self.data['color'] = self.hex_color

        self._env = Environment(self.color, feature_func=feature_func)

        self._trajectories = TrajectorySet([])
        self.candidates = []
        for traj in self._env.fitted_ramps:
            traj_id = hashlib.sha256(np.array(traj)).hexdigest()
            if traj_id not in self.candidates:
                traj_obj = Trajectory(self._env, traj)
                self._trajectories.append(traj_obj)
                self.candidates.append(traj_id)
        features_dim = len(self._trajectories[0].features)

        self._query_optimizer = QueryOptimizerDiscreteTrajectorySet(self._trajectories)

        params = {'weights': np.zeros(features_dim)}
        self._env.reward_weights = params['weights']

        self._user_model = SoftmaxUser(params)
        self._belief_model = SamplingBasedBelief(self._user_model, [], params)
                                            
        self._query = WeakComparisonQuery(self._trajectories[:2], chart=self.draw)
        self._search_result = None
        self._ranked_results = None

    def _ranker(self):
        """Ranks the trajectories in the colormap corpus according to the user belief model.

        The ranking is based on the expected reward of each trajectory, which is the dot product of its features and the user belief model's sample mean.
        """
        ranked = np.argsort(np.dot(self._trajectories.features_matrix, self._env.reward_weights))[::-1]
        results = TrajectorySet([])

        for i in ranked:
            results.append(self._trajectories.trajectories[i])

        self._ranked_results = results

    def teach(self, N=15):
        """Teaches the robot to rank sequential colormaps by asking the user for pairwise comparisons.

        Args:
            N (int): The number of colormap pairs to be queried for modeling user preferences.
        """
        true_user = HumanUser(delay=0.5)

        bar = widgets.IntProgress(min=0, max=N-1, layout=widgets.Layout(width='auto', height='36px', margin='8px'))
        bar.style.bar_color = self.hex_color

        for query_no in range(N):
            bar.value = query_no
            display(bar)

            queries, objective_values = self._query_optimizer.optimize('disagreement', self._belief_model, self._query)
            
            responses = true_user.respond(queries[0])
            self._belief_model.update(WeakComparison(queries[0], responses[0]))
            self._env.set_reward_weights(self._belief_model.mean['weights'])
        
        bar.value = query_no
        display(bar)

        self._ranker()


    def search(self, epochs=10000):
        """Searches for a new colormap using Optimistic Q-Learning and the user belief model.

        Args:
            epochs (int): The number of iterations to run the search algorithm (i.e., the number of colormaps to be sampled).
        """

        self.reward_history = []
        
        bar = widgets.IntProgress(min=0, max=epochs, layout=widgets.Layout(width='auto', height='36px', margin='8px'))
        bar.style.bar_color = self.hex_color
        display(bar)

        candidates = []
        for i in range(epochs):
            if i % 100 == 99:
                bar.value = i

            _ = self._env.run()

            self.reward_history.append(_)

            if self._env.total_reward > self._env.best_reward and len(self._env.trajectory) > 3:
                traj_id = hashlib.sha256(np.array(self._env.trajectory)).hexdigest()

                if traj_id not in candidates:
                    candidates.append(traj_id)
                    traj = Trajectory(self._env, self._env.trajectory)
                    in_gamut = traj.in_gamut
                    if in_gamut:
                        self._env.best_reward = self._env.total_reward
                        self._env.best_policy = self._env.trajectory

            self._env.reset()

        self._search_result = Trajectory(self._env, self._env.best_policy)

    def select(self):
        """Displays a slider for selecting a colormap from the results.
        
        The output of the slider will be stored in the `cmap` attribute of the Cieran object.
        """
        search_result = self._search_result
        min_value = 0

        if search_result:
            cmaps = [search_result] + self._ranked_results.trajectories
        else:
            cmaps = self._ranked_results.trajectories
        
        cmaps = cmaps[::-1]
        max_value = len(cmaps) - 1
        
        slider = widgets.IntSlider(
            value=max_value,
            min=min_value,
            max=max_value,
            step=1,
            description='',
            disabled=False,
            orientation='vertical',
            readout=False
        )

        out = widgets.Output()
        label = widgets.Label(layout=widgets.Layout(margin='0 auto'))

        def update_label(val):
            if search_result is None:
                label.value = 'Option ' + str(max_value - val + 1)
            else:
                if val == max_value:
                    label.value = 'New colormap'
                else:
                    label.value = 'Option ' + str(max_value - val)

        def update(val):
            cmap = cmaps[val]
            with out:
                out.clear_output(wait=True)
                self.draw(cmap.ramp)
            update_label(val)
            self.cmap = cmap.ramp

        update_label(slider.value)
        slider.observe(lambda change: update(change['new']), names='value')

        slider_with_label = widgets.VBox([label, slider])
        layout = widgets.HBox([slider_with_label, out])

        update(slider.value)

        display(layout)

    def plot_3d(self):
        """Plots a 3D visualization of the preference learning process and the model outputs.

        This recreates Figure 3 in the paper.
        """
        fig, axes = plt.subplots(1, 5, subplot_kw={'projection': '3d'}, figsize=(20, 4))

        fig.subplots_adjust(wspace=-0.35)

        for ax in axes:
            ax.margins(0)

        data1 = self._env.centroids
        data2 = self._env.fitted_ramps
        _data3 = [self._env.fitted_ramps[10], self._env.fitted_ramps[20]]
        data3 = []

        for ramp in _data3:
            controls = [Color("lab({}% {} {} / 1)".format(*p)) for p in ramp]
            curve = Color.interpolate(controls, method='bspline')
            at = np.linspace(0, 1, 256)
            data3.append([curve(index) for index in at])

        data4 = self._ranked_results
        _data5 = self._search_result
        data5 = []
        controls = [Color("lab({}% {} {} / 1)".format(*p)) for p in _data5]
        curve = Color.interpolate(controls, method='bspline')
        at = np.linspace(0, 1, 256)
        data5 = [curve(index) for index in at]


        axes[0].scatter([centroid[0] for centroid in data1], [centroid[1] for centroid in data1], [centroid[2] for centroid in data1], c=[np.array(Color("lab({}% {} {} / 1)".format(*centroid)).convert('srgb')[:3]) for centroid in data1], marker='o', s=40, alpha=1)
        axes[0].set_title('512 quantized colors')

        for ramp in data2:
            axes[1].plot([point[0] for point in ramp], [point[1] for point in ramp], [point[2] for point in ramp], c='gray', alpha=0.2)
        axes[1].set_title('Expert-designed colormaps')


        axes[2].plot([point[0] for point in data3[0][1:-1]], [point[1] for point in data3[0][1:-1]], [point[2] for point in data3[0][1:-1]], c='black')
        axes[2].scatter([point[0] for point in _data3[0][1:-1]], [point[1] for point in _data3[0][1:-1]], [point[2] for point in _data3[0][1:-1]], 
                        c=[np.array(Color("lab({}% {} {} / 1)".format(*point)).convert('srgb')[:3]) for point in _data3[0][1:-1]], marker='o', s=40, alpha=1)
        axes[2].plot([point[0] for point in data3[1]], [point[1] for point in data3[1]], [point[2] for point in data3[1]], c='gray', alpha=0.5)
        axes[2].scatter([point[0] for point in _data3[1]], [point[1] for point in _data3[1]], [point[2] for point in _data3[1]], c='gray', alpha=0.5, s=5)

        axes[2].scatter([point[0] for point in _data5][0], [point[1] for point in _data5][0], [point[2] for point in _data5][0], c='white', marker='o', s=40, alpha=1, edgecolors='black', linewidths=1)
        axes[2].scatter([point[0] for point in _data5][-1], [point[1] for point in _data5][-1], [point[2] for point in _data5][-1], c='black', marker='o', s=40, alpha=1, edgecolors='black', linewidths=1)

        axes[2].set_title('Pairwise comparison queries')

        rewards = [np.dot(result.features, self._env.reward_weights) for result in data4]
        norm = plt.Normalize(min(rewards), max(rewards))
        mapper = ScalarMappable(norm=norm, cmap='Greys')

        for i, result in enumerate(data4):
            axes[3].plot([point[0] for point in result.trajectory], [point[1] for point in result.trajectory], [point[2] for point in result.trajectory], c=mapper.to_rgba(rewards[i]), alpha=0.7)
        axes[3].set_title('Ranking expert colormaps')

        # Plot the trajectory of the search result
        axes[4].scatter([point[0] for point in _data5][0], [point[1] for point in _data5][0], [point[2] for point in _data5][0], c='white', marker='o', s=40, alpha=1, edgecolors='black', linewidths=1)
        axes[4].scatter([point[0] for point in _data5][-1], [point[1] for point in _data5][-1], [point[2] for point in _data5][-1], c='black', marker='o', s=40, alpha=1, edgecolors='black', linewidths=1)
        
        axes[4].plot([point[0] for point in data5], [point[1] for point in data5], [point[2] for point in data5], c='black')
        axes[4].scatter([point[0] for point in _data5[1:-1]], [point[1] for point in _data5[1:-1]], [point[2] for point in _data5[1:-1]], c=[np.array(Color("lab({}% {} {} / 1)".format(*point)).convert('srgb')[:3]) for point in _data5[1:-1]], marker='o', s=40, alpha=1)
        
        axes[4].text(_data5[0][0], _data5[0][1], _data5[0][2] + 30, "Initial state", color='black', fontsize=10, ha='center', va='center', style='italic')
        axes[4].text(_data5[-1][0], _data5[-1][1], _data5[-1][2] - 30, "End state", color='black', fontsize=10, ha='center', va='center', style='italic')

        axes[4].set_title('Generating a new colormap')

        for ax in axes[:-1]:
            ax.set_xticklabels([])

        for ax in axes[1:-1]:
            ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            ax.tick_params(axis='x', colors=(1.0, 1.0, 1.0, 0.0))

        # Move the tick labels in axes[4] 10 pixels in the x direction
        axes[4].xaxis.set_tick_params(pad=2, labelsize=10, labeltop=True, labelbottom=False, labelright=False, labelleft=False)

        for ax in axes:
            ax.set_xlim3d(100, 0)
            ax.set_ylim3d(-100, 100)
            ax.set_zlim3d(-100, 100)
        
            ax.set_yticklabels([])
            ax.set_zticklabels([])

            # Rotate 45 degrees clockwise
            ax.view_init(azim=0, elev=30)

            # Make the plot longer in the x direction
            ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1.5, 1, 1, 1]))

            ax.xaxis._axinfo['grid']['color'] = (0, 0, 0, 0.15)
            ax.yaxis._axinfo['grid']['color'] = (0, 0, 0, 0.15)
            ax.zaxis._axinfo['grid']['color'] = (0, 0, 0, 0.15)

            ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

        for ax in axes[1:]:
            # Remove the black outline and tickmarks
            ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

            ax.tick_params(axis='y', colors=(1.0, 1.0, 1.0, 0.0))
            ax.tick_params(axis='z', colors=(1.0, 1.0, 1.0, 0.0))


        # Label the axes
        axes[0].set_xlabel('L*')
        axes[0].set_ylabel('a*')
        axes[0].set_zlabel('b*')

        # Move the labels closer to the axes
        axes[0].xaxis.labelpad = -10
        axes[0].yaxis.labelpad = -10
        axes[0].zaxis.labelpad = -10

        plt.figure(figsize=(688/192, 384/192), dpi=192)




