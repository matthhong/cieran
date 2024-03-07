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

import csv
import os

from coloraide import Color

from collections import defaultdict
from functools import partial

import numpy as np
import hashlib

import ipywidgets as widgets
from IPython.display import display, clear_output
from random import randint

# define the colors you want to use
colors = []

COLOR_FILE = 'hex_values.txt'




class Cieran:
    def __init__(self, draw, color=None):
            
        self.draw = draw
        self._search_result = None

        self.lab_color = None
        self.hex_color = None

        self.data = {
            'id': randint(1, 999999)
            # 'choice2': {}
        }
        self.block = 0
        if color == 'tableau10':
            self._tableau10()
        elif color is not None:
            self.set_color(color)

        self._env = None
        self._trajectories = None
        self._query_optimizer = None
        self._user_model = None
        self._belief_model = None
        self._query = None
        self._ranked_results = None
        self.candidates = []


    def set_color(self, color):
        if isinstance(color, str):
            self.color = Color(color).convert('lab')._coords[:-1]
            self.hex_color = color
        elif isinstance(color, list) or isinstance(color, np.ndarray):
            self.color = color
            self.hex_color = Color("lab({}% {} {} / 1)".format(*color)).convert('srgb').to_string(hex=True)

        # self.data['choice'] = {}
        self.block += 1
        self.data['color'] = self.hex_color
        # self.data['block'] = self.block

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

        # params = {'weights': util_functions.get_random_normalized_vector(features_dim)}
        params = {'weights': np.zeros(features_dim)}
        self._env.reward_weights = params['weights']

        self._user_model = SoftmaxUser(params)
        self._belief_model = SamplingBasedBelief(self._user_model, [], params)
        # print('Estimated user parameters: ' + str(belief.mean))
                                            
        self._query = WeakComparisonQuery(self._trajectories[:2], chart=self.draw)
        self._search_result = None
        self._ranked_results = None

    def _tableau10(self):
        clear_output()
        # import color ramps
        with open(COLOR_FILE, 'r') as f:
            for line in f:
                # Remove the newline character
                line = line.strip()
                colors.append(line)

        # create a list to store the buttons
        buttons = []

        out = widgets.Output()
        display(out)

        def on_button_clicked(b):
            # clear_output()
            # destroy buttons
            # buttons = []
            self.set_color(b.style.button_color)

        # create a loop to generate the buttons
        for color in colors:
            button = widgets.Button(description='', layout=widgets.Layout(width='30px', height='30px'))
            button.style.button_color = color
            # Initialize the environment with the color on click
            button.on_click(on_button_clicked)
            buttons.append(button)

        # create a grid box to display the buttons
        grid = widgets.GridBox(buttons, layout=widgets.Layout(grid_template_columns='repeat(10, 30px)',
                                                            grid_template_rows='repeat(1, 32px)',
                                                            margin='8px'))
        # display the grid box
        with out:
            display(grid)

    def _ranker(self):
        ranked = np.argsort(np.dot(self._trajectories.features_matrix, self._env.reward_weights))[::-1]
        results = TrajectorySet([])

        for i in ranked:
            results.append(self._trajectories.trajectories[i])

        self._ranked_results = results

    def set_reward_weights(self, weights):
        self._env.set_reward_weights(weights)
        # self._belief_model.user_model.params['weights'] = weights

    def teach(self, n_queries=15):
        true_user = HumanUser(delay=0.5)

        # Visualize epochs as a progress bar
        bar = widgets.IntProgress(min=0, max=n_queries-1, layout=widgets.Layout(width='auto', height='36px', margin='8px'))
        bar.style.bar_color = self.hex_color
        # bar.style.margin = '8px'

        for query_no in range(n_queries):
            bar.value = query_no
            display(bar)

            queries, objective_values = self._query_optimizer.optimize('disagreement', self._belief_model, self._query)

            # print('Trajectory 1: ' + str(queries[0].slate[0].features))
            # print('Trajectory 2: ' + str(queries[0].slate[1].features))

            # print('Objective Value: ' + str(objective_values[0]))
            
            responses = true_user.respond(queries[0])
            self._belief_model.update(WeakComparison(queries[0], responses[0]))
            self._env.set_reward_weights(self._belief_model.mean['weights'])
            # print('Estimated user parameters: ' + str(self._belief.mean))
        
        bar.value = query_no
        display(bar)

        self._ranker()

        # best_traj = self._query_optimizer.planner(self._user_model)

    def search(self, weights=None, epochs=10000):
        if weights is not None:
            self._env.set_reward_weights(weights)

        self.reward_history = []
        
        # Visualize epochs as a progress bar
        bar = widgets.IntProgress(min=0, max=epochs, layout=widgets.Layout(width='auto', height='36px', margin='8px'))
        bar.style.bar_color = self.hex_color
        display(bar)

        candidates = []
        for i in range(epochs):
            if i % 100 == 99:
                bar.value = i

            # self._env.lr = max(self._env.lr * lr_decay_rate, min_lr)

            _ = self._env.run()

            # path, total_reward = self._env.get_best_path()
            # path = self._env.best_policy
            # total_reward = self._env.best_reward
            self.reward_history.append(_)

            # if total_reward > best_reward:
            #     best_reward = total_reward
            #     best_path = path
            if self._env.total_reward > self._env.best_reward and len(self._env.trajectory) > 3:
                # print(self._env.trajectory)
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

    def search_result(self):
        if self._search_result is not None:
            return self._search_result.ramp

    def ranked_results(self, N=4):
        if self._ranked_results is not None:
            ramps = []
            for i in range(N):
                ramps.append(self._ranked_results[i].ramp)
            return ramps
        
    @property
    def options(self):
        return self._options()

    def _options(self, M=1):
        if M % 2 == 0:
            return self.options_display
        else:
            return self.options_display2
        
    @property
    def options_display(self, shuffle=False):
        import ipywidgets as widgets
        results = [self._search_result, self._ranked_results[0]]

        # Get the middle element of _ranked_results
        middle = len(self._ranked_results.trajectories) // 2
        median = self._ranked_results[middle]

        # Get the last element of _ranked_results
        last = self._ranked_results[-1]
        results = results + [median, last]

        # self.data['weights'] = 
        for i, weight in enumerate(self._env.reward_weights):
            self.data['weight_' + str(i)] = weight
        # self.data['trajectories'] = [res.trajectory for res in results]

        # for result in self._ranked_results[:3]:
        #     self.data['trajectories'].append(result.trajectory)

        for i, result in enumerate(results):
            self.data[str(i) + '_reward'] = np.dot(result.features, self._env.reward_weights)

        def on_rank_change(change, id):
            # get new value
            new_value = change['new']
            
            self.data[id] = new_value

        outputs = []
        for i, result in enumerate(results):
            output = widgets.Output(description=str(i))

            bound = partial(on_rank_change, id=i)
            
            button = widgets.Button(description='Option ' + str(i), 
                                    layout=widgets.Layout(
                width='auto', height='auto', margin='8px',
                # Center the button
                display='flex', align_items='center', justify_content='center'
                ))
            text_box = widgets.IntText(
                value=None,
                description='Rank ',
                disabled=False
            )
            # Bind a callback to the text_box when the value changes
            text_box.observe(bound, names='value')

            output.append_display_data(text_box)

            display(output)
            outputs.append(output)
            with output:
                self.draw(result.ramp)

        if shuffle:
            np.random.shuffle(outputs)
        # Make a gridbox
        grid = widgets.GridBox(children=outputs, layout=widgets.Layout(grid_template_columns="repeat(2, 50%)"))
        display(grid)

    @property
    def options_display2(self, shuffle=False):
        import ipywidgets as widgets
        results = [self.search_result()] + self.ranked_results(3)

        outputs = []
        def on_rank_change2(change, id):
            # get new value
            new_value = change['new']
            
            self.data['choice2'][id] = new_value
            # self.data['weights'] = self._env.reward_weights
            self.data['trajectories2'] = [self._search_result.trajectory]

            for result in self._ranked_results[:3]:
                self.data['trajectories2'].append(result.trajectory)

            self.data['rewards2'] = [self._belief_model.reward(result) for result in [self._search_result] + self._ranked_results[:3]]

        for i, result in enumerate(results):
            output = widgets.Output(description=str(i))

            bound = partial(on_rank_change2, id=i)
            
            button = widgets.Button(description='Option ' + str(i), 
                                    layout=widgets.Layout(
                width='auto', height='auto', margin='8px',
                # Center the button
                display='flex', align_items='center', justify_content='center'
                ))
            text_box = widgets.IntText(
                value=None,
                description='Rank ',
                disabled=False
            )
            # Bind a callback to the text_box when the value changes
            text_box.observe(bound, names='value')

            # output.append_display_data(text_box)
            display(output)
            outputs.append(output)
            with output:
                self.draw(result)

        if shuffle:
            np.random.shuffle(outputs)
        # Make a gridbox
        grid = widgets.GridBox(children=outputs, layout=widgets.Layout(grid_template_columns="repeat(2, 50%)"))
        display(grid)

    
    def save_data(self):

        filename = './cieran' + str(self.data['id']) + '.csv'
        if not os.path.exists(filename):
            with open('./cieran' + str(self.data['id']) + '.csv', 'w') as f:
                writer = csv.DictWriter(f, fieldnames=self.data.keys())
                writer.writeheader()
                writer.writerow(self.data)
        else:
            with open('./cieran' + str(self.data['id']) + '.csv', 'a') as f:
                writer = csv.DictWriter(f, fieldnames=self.data.keys())
                writer.writerow(self.data)

    @property
    def results(self):
        # Throw an error if _search_result is None
        if self._search_result is None:
            raise ValueError('No search results found. Run the search() method first.')
        
        # Create a vertical slider on the right of the screen, and show the selected colormap on the left
        slider = widgets.IntSlider(
            value=5,
            min=0,
            max=10,
            description='Test:',
            disabled=False,
            continuous_update=True,
            orientation='vertical',
            readout=True,
            readout_format='d'
        )

        output = widgets.Output()
        output_list = [slider, output]

        # When the slider changes, draw the colormap next to it
        def on_value_change(change):
            with output:
                output.clear_output()
                self.draw(self._ranked_results[change['new']].ramp)
        slider.observe(on_value_change, names='value')

        #Display the slider and the output together
        display(widgets.HBox(output_list))


    def plot_3d(self):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib.cm import ScalarMappable

        # Create a row of five 3D subplots
        fig, axes = plt.subplots(1, 5, subplot_kw={'projection': '3d'}, figsize=(20, 4))

        # Remove padding between subplots
        fig.subplots_adjust(wspace=-0.35)

        # Remove margins from subplots
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

        # Plot the centroids while coloring each point according to its CIELAB values, without point opacity
        axes[0].scatter([centroid[0] for centroid in data1], [centroid[1] for centroid in data1], [centroid[2] for centroid in data1], c=[np.array(Color("lab({}% {} {} / 1)".format(*centroid)).convert('srgb')[:3]) for centroid in data1], marker='o', s=40, alpha=1)
        axes[0].set_title('512 quantized colors')

        # Plot the fitted ramps as 3D lines
        for ramp in data2:
            axes[1].plot([point[0] for point in ramp], [point[1] for point in ramp], [point[2] for point in ramp], c='gray', alpha=0.2)
        axes[1].set_title('Expert-designed colormaps')

        # Plot data 3

        axes[2].plot([point[0] for point in data3[0][1:-1]], [point[1] for point in data3[0][1:-1]], [point[2] for point in data3[0][1:-1]], c='black')
        axes[2].scatter([point[0] for point in _data3[0][1:-1]], [point[1] for point in _data3[0][1:-1]], [point[2] for point in _data3[0][1:-1]], 
                        c=[np.array(Color("lab({}% {} {} / 1)".format(*point)).convert('srgb')[:3]) for point in _data3[0][1:-1]], marker='o', s=40, alpha=1)
        axes[2].plot([point[0] for point in data3[1]], [point[1] for point in data3[1]], [point[2] for point in data3[1]], c='gray', alpha=0.5)
        axes[2].scatter([point[0] for point in _data3[1]], [point[1] for point in _data3[1]], [point[2] for point in _data3[1]], c='gray', alpha=0.5, s=5)

        axes[2].scatter([point[0] for point in _data5][0], [point[1] for point in _data5][0], [point[2] for point in _data5][0], c='white', marker='o', s=40, alpha=1, edgecolors='black', linewidths=1)
        # Give the last circle a stroke
        axes[2].scatter([point[0] for point in _data5][-1], [point[1] for point in _data5][-1], [point[2] for point in _data5][-1], c='black', marker='o', s=40, alpha=1, edgecolors='black', linewidths=1)
        

        axes[2].set_title('Pairwise comparison queries')

        # Compute the reward for each result
        rewards = [np.dot(result.features, self._env.reward_weights) for result in data4]
        norm = plt.Normalize(min(rewards), max(rewards))
        mapper = ScalarMappable(norm=norm, cmap='Greys')

        # # Invert the colormap so that the best results are blue
        # mapper.set_array(rewards)
        # mapper.set_clim(min(rewards), max(rewards))


        # Plot data 4 and color according to reward (orange is bad, blue is good)
        for i, result in enumerate(data4):
            axes[3].plot([point[0] for point in result.trajectory], [point[1] for point in result.trajectory], [point[2] for point in result.trajectory], c=mapper.to_rgba(rewards[i]), alpha=0.7)
        axes[3].set_title('Ranking expert colormaps')

        # Plot the trajectory of the search result
                # Give the first circle a stroke
        axes[4].scatter([point[0] for point in _data5][0], [point[1] for point in _data5][0], [point[2] for point in _data5][0], c='white', marker='o', s=40, alpha=1, edgecolors='black', linewidths=1)
        # Give the last circle a stroke
        axes[4].scatter([point[0] for point in _data5][-1], [point[1] for point in _data5][-1], [point[2] for point in _data5][-1], c='black', marker='o', s=40, alpha=1, edgecolors='black', linewidths=1)
        
        axes[4].plot([point[0] for point in data5], [point[1] for point in data5], [point[2] for point in data5], c='black')
        axes[4].scatter([point[0] for point in _data5[1:-1]], [point[1] for point in _data5[1:-1]], [point[2] for point in _data5[1:-1]], c=[np.array(Color("lab({}% {} {} / 1)".format(*point)).convert('srgb')[:3]) for point in _data5[1:-1]], marker='o', s=40, alpha=1)
        
        # Label the first point as "Start" 30 pixels above the point in the z-direction
        axes[4].text(_data5[0][0], _data5[0][1], _data5[0][2] + 30, "Initial state", color='black', fontsize=10, ha='center', va='center', style='italic')
        # Label the last point as "End" 30 pixels below the point in the z-direction
        axes[4].text(_data5[-1][0], _data5[-1][1], _data5[-1][2] - 30, "End state", color='black', fontsize=10, ha='center', va='center', style='italic')

        axes[4].set_title('Generating a new colormap')

        for ax in axes[:-1]:
            ax.set_xticklabels([])

        for ax in axes[1:-1]:
            ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            ax.tick_params(axis='x', colors=(1.0, 1.0, 1.0, 0.0))

        # Move the tick labels in axes[4] 10 pixels in the x direction
        axes[4].xaxis.set_tick_params(pad=2, labelsize=10, labeltop=True, labelbottom=False, labelright=False, labelleft=False)

        # Set all axes to the same scale
        for ax in axes:
            ax.set_xlim3d(100, 0)
            ax.set_ylim3d(-100, 100)
            ax.set_zlim3d(-100, 100)
        
            # ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])

            # Zoom into the plot
            # ax.view_init(azim=-90, elev=30)

            # Rotate 45 degrees clockwise
            ax.view_init(azim=0, elev=30)

            # Make the plot longer in the x direction
            ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1.5, 1, 1, 1]))

            # Make the reference lines lighter
            ax.xaxis._axinfo['grid']['color'] = (0, 0, 0, 0.15)
            ax.yaxis._axinfo['grid']['color'] = (0, 0, 0, 0.15)
            ax.zaxis._axinfo['grid']['color'] = (0, 0, 0, 0.15)

            # # Make all three panels transparent
            # ax.patch.set_alpha(0)

            # Make the background black
            # ax.set_facecolor((0, 0, 0, 1))

            ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

            # Make the whole plot bigger
            # ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1.5, 1.5, 1.5, 1]))

        for ax in axes[1:]:
            # Remove the black outline of the plot
            # ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

            # Remove the tick marks too
            # ax.tick_params(axis='x', colors=(1.0, 1.0, 1.0, 0.0))
            ax.tick_params(axis='y', colors=(1.0, 1.0, 1.0, 0.0))
            ax.tick_params(axis='z', colors=(1.0, 1.0, 1.0, 0.0))

            # Draw another xz-plane across the plot
            

        

        # Label the axes
        axes[0].set_xlabel('L*')
        axes[0].set_ylabel('a*')
        axes[0].set_zlabel('b*')

        # Move the labels closer to the axes
        axes[0].xaxis.labelpad = -10
        axes[0].yaxis.labelpad = -10
        axes[0].zaxis.labelpad = -10

        # plt.figure(figsize=(688/192, 384/192), dpi=192)

        # Save as svg
        plt.savefig('colormap.svg', bbox_inches='tight', pad_inches=0)

        # # Plot the points array
        # # ax.scatter([point[0] for point in points], [point[1] for point in points], [point[2] for point in points], c='b', marker='o', s=10)

        # # Plot the centroids while coloring each point according to its CIELAB values, without point opacity
        # ax.scatter([centroid[0] for centroid in points], [centroid[1] for centroid in points], [centroid[2] for centroid in points], c=[np.array(Color("lab({}% {} {} / 1)".format(*centroid)).convert('srgb')[:3]) for centroid in points], marker='o', s=40, alpha=1)

        # # path = [np.array([0, 0, 0]), np.array([ 19.2989826 ,  41.74382203, -38.76011605]), np.array([ 32.28726385,  27.86865056, -41.53451605]), np.array([ 37.48745916,  -2.38207341, -36.01835605]), np.array([ 50.01187323, -21.99646146, -15.25931605]), np.array([ 51.50113104, -34.32994721,   6.33204395]), np.array([ 62.27993963, -45.39380943,  27.61332395]), np.array([ 70.80044744, -44.6164889 ,  49.12308395]), np.array([ 82.76333807, -28.52595392,  72.16692395]), np.array([85.80288885, -7.81036178, 78.41748395]), np.array([100,   0,   0])]

        # # # traj = Trajectory(None, path)
        # # # curve = traj.get_curve(1000)
        # # # Plot the continuous curve
        # # # breakpoint()
        # # # ax.plot(*zip(*curve), c='r', linewidth=2)

        # # # Plot the discrete path
        # # ax.scatter([point[0] for point in path], [point[1] for point in path], [point[2] for point in path], c='r', marker='o')

        # # label axes
        # ax.set_xlabel('L*')
        # ax.set_ylabel('a*')
        # ax.set_zlabel('b*')

        # # set axis ranges
        # ax.set_xlim(0, 100)
        # ax.set_ylim(-128, 128)
        # ax.set_zlim(-128, 128)
        # plt.show()




