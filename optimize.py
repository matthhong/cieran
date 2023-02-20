# Parameter optimization for Q-learning

import cieran
import numpy as np
import matplotlib.pyplot as plt

import networkx as nx
from cieran.utils import util_functions

def draw_chart(cmap):

    t = np.linspace(0, 2 * np.pi, 1024)
    data2d = np.sin(t)[:, np.newaxis] * np.cos(t)[np.newaxis, :]
    # Draw a chart of data2d with the given colormap
    fig, ax = plt.subplots()
    ax.imshow(data2d, cmap=cmap)

    plt.show()

env = cieran.load_environment([48.325, -19.2993, -16.5717])
env.reward_weights = np.array([ -0,  0.84481028,  0.96703832])

def train_model(env, reward_weights, discount, lr, default_Q=3.0, epsilon=0.1):
    env.reward_weights = reward_weights

    path = nx.shortest_path(env.graph, source=(100, 0, 0), target=(0,0,0))

    env.discount = discount
    env.lr = lr
    env.Q = {}

    for i in range(len(path)-1):
        env.Q[(path[i], path[i+1])] = default_Q
        
    env.epsilon = epsilon

    cmap, history, reward_history = cieran.train(env)

    return reward_history[-1]

# Q: how many hours does it take to run 5^8 trials if each trial takes 3 seconds?
# A: (5^8) * 3 / 3600 = 



def experiment1():
    # Experiment with 5 different values of each parameter
    num_weight_vals = 3
    num_vals = 5

    w_1 = np.linspace(-1, 1, 3)
    w_2 = np.linspace(-1, 1, 3)
    w_3 = np.linspace(-1, 1, 3)
    discount = np.linspace(0.1, 1, num_vals)
    lr = np.linspace(0.1, 1, num_vals)
    default_Q = 3.0
    epsilon = 0.1

    # Iterate through all combinations of parameters
    combos = np.array(np.meshgrid(w_1, w_2, w_3, discount, lr)).T.reshape(-1,5)

    # Remove combinations where w_1, w_2, and w_3 are all 0
    combos = combos[~np.all(combos[:, :3] == 0, axis=1)]

    model_output = np.zeros((num_weight_vals**3 * num_vals**2 - num_vals**2, 8))

    # Should take 30 minutes to run
    # time it
    
    import time
    start = time.time()

    for i in range(len(combos)):
        reward_weights = combos[i, :3]
        discount = combos[i, 3]
        lr = combos[i, 4]

        reward = train_model(env, reward_weights, discount, lr, default_Q, epsilon)

        model_output[i] = np.array([reward, *reward_weights, discount, lr, default_Q, epsilon])

    np.save('experiment1.npy', model_output)
    # print in minutes
    print('Took ' + (time.time() - start) / 60 + ' minutes')



if __name__=='__main__':
    experiment1()
    # # Test different values of discount
    # model_output = np.zeros((10, 9))
    
    # # Get 10 values of discount
    # num_vals = 10
    # discount = np.linspace(0.1, 1, num_vals)
    # for i in range(num_vals):
    #     reward_weights = util_functions.get_random_normalized_vector(len(env.reward_weights))
    #     lr = np.random.uniform(0.1, 1)
    #     default_Q = np.random.uniform(0, 5)
    #     epsilon = np.random.uniform(0.05, 0.75)

    #     reward = train_model(env, reward_weights, discount[i], lr, default_Q, epsilon)

    #     # breakpoint()
    #     model_output[i] = np.array([reward, *reward_weights, discount[i], lr, default_Q, epsilon])

    # # np.save('model_output.npy', model_output)

    # # Plot the results
    # fig, ax = plt.subplots()
    # ax.scatter(model_output[:, 5], model_output[:, 0], s=1)
    # ax.set_xlabel('Discount')
    # ax.set_ylabel('Reward')
    # plt.show()

        # # Get 10 values of lr
        # model_output = np.zeros((10, 9))

        # num_vals = 10
        # lr = np.linspace(0.1, 1, num_vals)
        # for i in range(num_vals):
        #     reward_weights = util_functions.get_random_normalized_vector(len(env.reward_weights))
        #     discount = np.random.uniform(0.1, 1)
        #     default_Q = np.random.uniform(0, 5)
        #     epsilon = np.random.uniform(0.05, 0.75)

        #     reward = train_model(env, reward_weights, discount, lr[i], default_Q, epsilon)

        #     # breakpoint()
        #     model_output[i] = np.array([reward, *reward_weights, discount, lr[i], default_Q, epsilon])

        # # np.save('model_output.npy', model_output)

        # # Plot the results
        # fig, ax = plt.subplots()
        # ax.scatter(model_output[:, 6], model_output[:, 0], s=1)
        # ax.set_xlabel('Learning Rate')
        # ax.set_ylabel('Reward')
        # plt.show()

    # Get 10 values of default_Q
    # model_output = np.zeros((10, 9))
    
    # num_vals = 10
    # default_Q = np.linspace(0, 10, num_vals)
    # for i in range(num_vals):
    #     reward_weights = util_functions.get_random_normalized_vector(len(env.reward_weights))
    #     discount = np.random.uniform(0.1, 1)
    #     lr = np.random.uniform(0.1, 1)
    #     epsilon = np.random.uniform(0.05, 0.75)

    #     reward = train_model(env, reward_weights, discount, lr, default_Q[i], epsilon)

    #     # breakpoint()
    #     model_output[i] = np.array([reward, *reward_weights, discount, lr, default_Q[i], epsilon])

    # # np.save('model_output.npy', model_output)

    # # Plot the results
    # fig, ax = plt.subplots()
    # ax.scatter(model_output[:, 7], model_output[:, 0], s=1)
    # ax.set_xlabel('Default Q')
    # ax.set_ylabel('Reward')
    # plt.show()

