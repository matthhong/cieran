from cieran.basics.environment import Environment
from cieran.basics.trajectory import Trajectory, TrajectorySet
from cieran.basics.features import feature_func

from cieran.learning.data_types import WeakComparisonQuery, WeakComparison
from cieran.learning.user_models import SoftmaxUser, HumanUser
from cieran.learning.belief_models import SamplingBasedBelief

from cieran.querying.query_optimizer import QueryOptimizerDiscreteTrajectorySet

from cieran.utils import util_functions

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
            'id': randint(1, 9999)
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
        ranked = np.argsort(self._belief_model.reward(self._trajectories))[::-1]
        results = TrajectorySet([])

        for i in ranked:
            results.append(self._trajectories.trajectories[i])

        self._ranked_results = results

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

    def search(self, weights=None, epochs=20000):
        if weights is not None:
            self._env.set_reward_weights(weights)

        self.reward_history = []
        
        # Visualize epochs as a progress bar
        bar = widgets.IntProgress(min=0, max=epochs, layout=widgets.Layout(width='auto', height='36px', margin='8px'))
        bar.style.bar_color = self.hex_color
        display(bar)

        self._env.discount= 1
        self._env.lr = 1
        self._env.Q = defaultdict(float)
        self._env.epsilon = 0.1
        self._env.Q.default_factory = lambda: 100.0

        lr_decay_rate = 0.9995
        min_lr = 0.01

        best_reward = float("-inf")
        best_path = None
        candidates = []
        for i in range(epochs):
            if i % 100 == 99:
                bar.value = i

            self._env.lr = max(self._env.lr * lr_decay_rate, min_lr)

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
        
    def options(self, M=0):
        if M % 2 == 0:
            return self.options_display
        else:
            return self.options_display2
        
    @property
    def options_display(self, shuffle=True):
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
            self.data[str(i) + '_reward'] = self._belief_model.reward(result)

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

            outputs.append(output)
            with output:
                self.draw(result.ramp)

        if shuffle:
            np.random.shuffle(outputs)
        # Make a gridbox
        grid = widgets.GridBox(children=outputs, layout=widgets.Layout(grid_template_columns="repeat(2, 50%)"))
        display(grid)

    @property
    def options_display2(self, shuffle=True):
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

            output.append_display_data(text_box)

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




