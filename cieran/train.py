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

import numpy as np




def visualize_path(ramp):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ramp = np.array(ramp)
    ax.plot(ramp[:,0], ramp[:,1], ramp[:,2])
    
    # Label the axes
    ax.set_xlabel('L')
    ax.set_ylabel('A')
    ax.set_zlabel('B')

    # Set the limits of the axes
    ax.set_xlim(0, 100)
    ax.set_ylim(-127, 127)
    ax.set_zlim(-127, 127)

    # Give the plot a title
    title = ax.text(0.5, 1.05, 'Cieran', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

    plt.show()

def load_environment(color):
    # Load environment from pickle file
    with open(str(color) + '.pkl', 'rb') as f:
        env = pickle.load(f)

    return env

def query(color, render=None):
    # Need to be able to capture hexcode, lab, rgb, or cmyk

    # TODO: Maybe we can just precompute the trajectories and save them to a file by using some gray color as a

    # Save environment as a pickle file

    
    env = Environment(color, feature_func=feature_func)

    # with open(str(color) + '.pkl', 'wb') as f:
    #     pickle.dump(env, f)

    # Generate trajectories here as opposed to the above
    # trajectory_set = generate_trajectories_randomly(env, num_trajectories=100, max_episode_length=300, file_name='Cieran', seed=0)
    trajectory_set = TrajectorySet([])
    for traj in env.fitted_ramps:
        trajectory_set.append(Trajectory(env, traj))
    features_dim = len(trajectory_set[0].features)

    query_optimizer = QueryOptimizerDiscreteTrajectorySet(trajectory_set)

    true_user = HumanUser(delay=0.5)

    params = {'weights': util_functions.get_random_normalized_vector(features_dim)}
    # params = {'weights': [-1.0] * features_dim}
    user_model = SoftmaxUser(params)
    belief = SamplingBasedBelief(user_model, [], params)
    print('Estimated user parameters: ' + str(belief.mean))
                                        
    query = WeakComparisonQuery(trajectory_set[:2], chart=render)

    for query_no in range(10):
        queries, objective_values = query_optimizer.optimize('disagreement', belief, query)
        # queries, objective_values = query_optimizer.optimize('disagreement', belief, query, optimization_method='medoids', batch_size=6)

        # Print trajectory features for each query
        print('Trajectory 1: ' + str(queries[0].slate[0].features))
        print('Trajectory 2: ' + str(queries[0].slate[1].features))

        print('Objective Value: ' + str(objective_values[0]))
        
        responses = true_user.respond(queries[0])
        belief.update(WeakComparison(queries[0], responses[0]))
        print('Estimated user parameters: ' + str(belief.mean))

    env.reward_weights = belief.mean['weights']
    best_traj = query_optimizer.planner(user_model)

    return env, best_traj

def train(env):
    epochs = 20000
    path_history = []
    reward_history = []
    print("Learning...")

    epsilon = 1.0
    eps_decay_rate = 0.999
    min_eps = 0.001
    for i in range(epochs):
        # Decay epsilon
        epsilon = max(epsilon * eps_decay_rate, min_eps)
        env.epsilon = epsilon

        env.run()

        path, total_reward = env.get_best_path()
        path_history.append(path)
        reward_history.append(total_reward)

        # if i > 0 and i % 500 == 0:
        #     # Compare the reward of the current path to the reward of the path 499 epochs ago, and stop if they are the same
        #     if reward_history[-1] == reward_history[-500]:
        #         print("Converged at epoch " + str(i))
        #         break

        env.reset()

    return Trajectory(env, path_history[-1]), path_history, reward_history

