import networkx as nx
import numpy as np
import random

def eps_greedy_action(proposed_action, epsilon, available_actions, state):
    # Choose a random action with probability epsilon
    # Otherwise, choose the action with the highest expected reward
    if np.random.rand() < epsilon or state not in proposed_action:
        return np.random.choice(available_actions)
    else:
        return proposed_action[state]

def trip(graph, rewards, proposed_action, epsilon, returns, V, Q):
    state = 'A'
    history = []
    while state != 'E':
        available_actions = list(graph.neighbors(state))
        action = eps_greedy_action(proposed_action, epsilon, available_actions, state)
        history.append((state, action))
        new_state = action
        state = new_state
    G = 0
    for (state, action) in list(reversed(history)):
        G = G + rewards[(state, action)]
        if state not in [h[0] for h in history]:
            print("hi")
            returns[state].append(G)
            V[state] = np.average(returns[state])
        new_state = action
        Q[(state, action)] = V[new_state] + rewards[(state, action)]
    return Q

# Example usage
# Create a graph and add edges with weights
G = nx.DiGraph()
G.add_edge("A", "B", weight=1)
G.add_edge("A", "C", weight=2)
G.add_edge("B", "D", weight=1)
G.add_edge("B", "C", weight=2)
G.add_edge("C", "D", weight=1)
G.add_edge("D", "E", weight=4)
G.add_edge("C", "E", weight=1)
#Shortest path is A -> C -> E with a total reward of -3
#Longest path is A -> B -> D -> E with a total reward of -6

# Define the rewards for each (state, action) pair
rewards = dict(((u, v), G[u][v]["weight"]) for u, v in G.edges())

# Define the initial Q-values
Q = dict(((u, v), 0) for u, v in G.edges())

# Define the state-value function
V = dict((node, 0) for node in G.nodes())

# Define the returns for each state
returns = dict((node, []) for node in G.nodes())
proposed_action = {}

# Define the epsilon value for exploration-exploitation trade-off
epsilon = 0.5

# Define the maximum number of iterations
max_iter = 10000

for t in range(max_iter):
    Q = trip(G, rewards, proposed_action, epsilon, returns, V, Q)
    for node in G.nodes():
        actions_rewards = {}
        for neighbor in G.neighbors(node):
            actions_rewards[neighbor] = Q[(node, neighbor)]
        
        if len(actions_rewards) > 0:
            proposed_action[node] = max(actions_rewards, key=actions_rewards.get)

# breakpoint()
print(proposed_action)