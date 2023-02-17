from cieran.basics.environment import Environment
from cieran.basics.trajectory import Trajectory
from cieran.basics.features import feature_func

from cieran.learning.data_types import WeakComparisonQuery, WeakComparison
from cieran.learning.user_models import SoftmaxUser, HumanUser
from cieran.learning.belief_models import SamplingBasedBelief

from cieran.querying.query_optimizer import QueryOptimizerDiscreteTrajectorySet

from cieran.utils.generate_trajectories import generate_trajectories_randomly

import pickle


def train(color, render=None):
    # Need to be able to capture hexcode, lab, rgb, or cmyk

    # TODO: Maybe we can just precompute the trajectories and save them to a file by using some gray color as a

    # Save environment as a pickle file
    import pickle

    env_name = 'Cieran'

    with open(env_name + '.pkl', 'rb') as f:
        env = pickle.load(f)
    
    # env = Environment(color, feature_func=feature_func)

    # with open(env_name + '.pkl', 'wb') as f:
    #     pickle.dump(env, f)

    # Generate trajectories here as opposed to the above
    trajectory_set = generate_trajectories_randomly(env, num_trajectories=100, max_episode_length=300, file_name='Cieran', seed=0)
    features_dim = len(trajectory_set[0].features)

    query_optimizer = QueryOptimizerDiscreteTrajectorySet(trajectory_set)

    true_user = HumanUser(delay=0.5)

    # params = {'weights': util_funs.get_random_normalized_vector(features_dim)}
    params = {'weights': [-1.0] * features_dim}
    user_model = SoftmaxUser(params)
    belief = SamplingBasedBelief(user_model, [], params)
    print('Estimated user parameters: ' + str(belief.mean))
                                        
    query = WeakComparisonQuery(trajectory_set[:2], chart=render)

    for query_no in range(10):
        queries, objective_values = query_optimizer.optimize('disagreement', belief, query, optimization_method='medoids', batch_size=6)

        print('Objective Value: ' + str(objective_values[0]))
        
        responses = true_user.respond(queries[0])
        belief.update(WeakComparison(queries[0], responses[0]))
        print('Estimated user parameters: ' + str(belief.mean))

    env.reward_weights = belief.mean['weights']

    epochs = 1000
    print("Learning...")
    for i in range(epochs):
        env.run()
        env.reset()

    path = env.get_best_path()
    return Trajectory(env, path).ramp