# Making APReL work for networkx graphs

# Ideal workflow:
# trajectory_set = crowdsourced corpus
# features_func = feature function

import aprel
import matplotlib as mpl


from matplotlib.colors import ListedColormap

from geomdl import fitting

from coloraide import Color
import numpy as np

def max_slope_difference(trajectory):
    a_slopes = []
    b_slopes = []
    for i in range(len(trajectory) - 1):
        a_slopes.append((trajectory[i+1][1] - trajectory[i][1]) / (trajectory[i+1][0] - trajectory[i][0]))
        b_slopes.append((trajectory[i+1][2] - trajectory[i][2]) / (trajectory[i+1][0] - trajectory[i][0]))
    
    # Compute differences in slope
    a_diffs = []
    b_diffs = []
    for i in range(len(a_slopes) - 1):
        a_diffs.append(abs(a_slopes[i+1] - a_slopes[i]))
        b_diffs.append(abs(b_slopes[i+1] - b_slopes[i]))

    try:
        return max([max(a_diffs), max(b_diffs)]) / 127
    except ValueError:
        return 0

def a_range(trajectory):
    #min is either 0 or the actual min
    min_val = 0
    if min([point[1] for point in trajectory]) < 0:
        min_val = min([point[1] for point in trajectory])
    return [abs(min_val)/128, max([point[1] for point in trajectory])/127]

def b_range(trajectory):
    min_val = 0
    if min([point[2] for point in trajectory]) < 0:
        min_val = min([point[2] for point in trajectory])
    return [abs(min_val)/128, max([point[2] for point in trajectory])/127]


def feature_func(trajectory):
    a_list = a_range(trajectory)
    b_list = b_range(trajectory)
    return np.array([max_slope_difference(trajectory), a_list[0], a_list[1], b_list[0], b_list[1]])
    

def main():
    # env = aprel.Environment([69.33, -10.77, -24.79], feature_func=feature_func)
    env_name = 'Cieran'

    env = None

    # Save environment as a pickle file
    import pickle
    # with open(env_name + '.pkl', 'wb') as f:
    #     pickle.dump(env, f)
    
    # Load environment from pickle file
    with open(env_name + '.pkl', 'rb') as f:
        env = pickle.load(f)

    trajectory_set = aprel.generate_trajectories_randomly(env, num_trajectories=100,max_episode_length=300,file_name=env_name, seed=0)
    features_dim = len(trajectory_set[0].features)

    # breakpoint()

    query_optimizer = aprel.QueryOptimizerDiscreteTrajectorySet(trajectory_set)

    true_user = aprel.HumanUser(delay=0.5)

    # params = {'weights': aprel.util_funs.get_random_normalized_vector(features_dim)}
    params = {'weights': [-1.0] * features_dim}
    user_model = aprel.SoftmaxUser(params)
    belief = aprel.SamplingBasedBelief(user_model, [], params)
    print('Estimated user parameters: ' + str(belief.mean))
                                        
    query = aprel.WeakComparisonQuery(trajectory_set[:2])

    for query_no in range(10):
        queries, objective_values = query_optimizer.optimize('disagreement', belief, query, optimization_method='medoids', batch_size=6)

        print('Objective Value: ' + str(objective_values[0]))
        
        responses = true_user.respond(queries[0])
        belief.update(aprel.Preference(queries[0], responses[0]))
        print('Estimated user parameters: ' + str(belief.mean))

    # Qlearning
    env.reward_weights = belief.mean['weights']

    epochs = 1000
    Q = env.Q.copy()
    for i in range(epochs):
        env.run()
        # Test for convergence on Q table values
        print("Epoch {}".format(i))
        if i % 100 == 0:
            # print("Q table: {}".format(env.Q))
            # print("Q table diff: {}".format({k: env.Q[k] - Q[k] for k in env.Q.keys() & Q.keys()}))
            # print("Best path: {}".format(env.get_best_path()))
            Q = env.Q.copy()
        env.reset()

    # Get the path
    path = env.get_best_path()
    return aprel.Trajectory(env, path).ramp



if __name__=="__main__":
    ramp = main()
    # [(72.45506299680495, 19.451625626588225, -14.244807378939939), (61.419906746804955, 20.967400660881793, -22.32320737893994), (54.876937996804955, 21.550391058686984, -26.076807378939947), (46.057357918679955, 23.985995387295446, -33.355527378939954), (37.909164559304955, 32.588342590465686, -32.49056737893994), (27.118148934304955, 36.90247153422425, -40.568967378939945)]

    # Visualize the cmap using matplotlib using a simple colorbar
    fig, ax = mpl.pyplot.subplots(figsize=(6, 1))
    fig.subplots_adjust(bottom=0.5)

    norm = mpl.colors.Normalize(vmin=0, vmax=1)

    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=ramp,

                                    norm=norm,

                                    orientation='horizontal')

    mpl.pyplot.show()



