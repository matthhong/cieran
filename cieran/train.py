from cieran.basics.environment import Environment
from cieran.basics.trajectory import Trajectory, TrajectorySet
from cieran.basics.features import feature_func

from cieran.learning.data_types import WeakComparisonQuery, WeakComparison
from cieran.learning.user_models import SoftmaxUser, HumanUser
from cieran.learning.belief_models import SamplingBasedBelief

from cieran.querying.query_optimizer import QueryOptimizerDiscreteTrajectorySet

from cieran.utils.generate_trajectories import generate_trajectories_randomly
from cieran.utils import util_functions

import pickle

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from coloraide import Color

from collections import defaultdict

import numpy as np

import ipywidgets as widgets
from IPython.display import display, clear_output

# define the colors you want to use
colors = []

COLOR_FILE = 'hex_values.txt'


from coloraide import Color


def initialize_color(button):
    global env
    env = cieran.initialize(button.style.button_color)

def tableau10():
    # import color ramps
    with open(COLOR_FILE, 'r') as f:
        for line in f:
            # Remove the newline character
            line = line.strip()
            colors.append(line)

    # create a list to store the buttons
    buttons = []

    # create a loop to generate the buttons
    for color in colors:
        button = widgets.Button(description='', layout=widgets.Layout(width='30px', height='30px'))
        button.style.button_color = color
        # Initialize the environment with the color on click
        button.on_click(initialize_color)
        buttons.append(button)

    # create a grid box to display the buttons
    grid = widgets.GridBox(buttons, layout=widgets.Layout(grid_template_columns='repeat(10, 30px)',
                                                        grid_template_rows='repeat(1, 30px)'))

    # display the grid box
    display(grid)


class Cieran:
    def __init__(self, draw, color=None, palette=None):
            
        self.draw = draw
        self.search_result = None

        self._env = None
        self._trajectories = None
        self._query_optimizer = None
        self._user_model = None
        self._belief = None
        self._query = None

        if palette == 'tableau10':
            self._tableau10()

    def set_color(self, color):
        if isinstance(color, str):
            color = Color(color).convert('lab')._coords[:-1]

        self._env = Environment(color, feature_func=feature_func)

        self._trajectories = TrajectorySet([])
        for traj in self._env.fitted_ramps:
            self._trajectories.append(Trajectory(self._env, traj))
        features_dim = len(self._trajectories[0].features)

        self._query_optimizer = QueryOptimizerDiscreteTrajectorySet(self._trajectories)

        params = {'weights': util_functions.get_random_normalized_vector(features_dim)}
        self._env.reward_weights = params['weights']

        self._user_model = SoftmaxUser(params)
        self._belief = SamplingBasedBelief(self._user_model, [], params)
        # print('Estimated user parameters: ' + str(belief.mean))
                                            
        self._query = WeakComparisonQuery(self._trajectories[:2], chart=self.draw)
        self.search_result = None

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


    def teach(self, n_queries=15):
        true_user = HumanUser(delay=0.5)

        # Visualize epochs as a progress bar
        bar = widgets.IntProgress(min=0, max=n_queries-1, layout=widgets.Layout(width='auto', height='36px', margin='8px'))
        bar.style.bar_color = 'black'
        # bar.style.margin = '8px'

        for query_no in range(n_queries):
            bar.value = query_no
            display(bar)

            queries, objective_values = self._query_optimizer.optimize('disagreement', self._belief, self._query)

            # print('Trajectory 1: ' + str(queries[0].slate[0].features))
            # print('Trajectory 2: ' + str(queries[0].slate[1].features))

            # print('Objective Value: ' + str(objective_values[0]))
            
            responses = true_user.respond(queries[0])
            self._belief.update(WeakComparison(queries[0], responses[0]))
            self._env.reward_weights = self._belief.mean['weights']
            # print('Estimated user parameters: ' + str(self._belief.mean))

        # best_traj = self._query_optimizer.planner(self._user_model)


    def search(self, weights=None, epochs=20000):
        if weights is None:
            weights = self._env.reward_weights

        self.reward_history = []
        
        # Visualize epochs as a progress bar
        bar = widgets.IntProgress(min=0, max=epochs, layout=widgets.Layout(width='auto', height='36px', margin='8px'))
        display(bar)

        self._env.discount= 1
        self._env.lr = 1
        self._env.Q = defaultdict(float)
        self._env.epsilon = 0.0
        self._env.Q.default_factory = lambda: 100.0

        lr_decay_rate = 0.999
        min_lr = 0.01

        best_reward = -99999
        best_path = None
        for i in range(epochs):
            if i % 100 == 99:
                bar.value = i

            self._env.lr = max(self._env.lr * lr_decay_rate, min_lr)

            _ = self._env.run()

            path, total_reward = self._env.get_best_path()
            self.reward_history.append(total_reward)

            if total_reward > best_reward:
                best_reward = total_reward
                best_path = path

            self._env.reset()

        self.search_result = Trajectory(self._env, best_path)

    def results(self, N=4):

        # top = np.argmax(self._user_model.reward(self.trajectories)).item()
        # Get top N trajectories
        top_n = np.argpartition(self._user_model.reward(self.trajectories), -N)[-N:]

        results = TrajectorySet()

        if self.search_result is not None:
            results.append(self.search_result)

        for i in top_n:
            results.append(self.trajectories[i])

        return results





