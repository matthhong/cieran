
import random
import numpy
# Find shortest path in a graph using reinforcement learning
# QLearning: graph, source, target, weight, alpha, gamma, epsilon, lambd -> path

def total_variation_3d(lst):
    # Find the total variation of a list of 3d points
    return sum([abs(lst[i][0] - lst[i+1][0]) + abs(lst[i][1] - lst[i+1][1]) + abs(lst[i][2] - lst[i+1][2]) for i in range(len(lst)-1)])

def min_a(lst):
    return min([i[1] for i in lst])

def max_a(lst):
    return max([i[1] for i in lst])

def min_b(lst):
    return min([i[2] for i in lst])

def max_b(lst):
    return max([i[2] for i in lst])

def feature_func(traj):
    return [total_variation_3d(traj), min_a(traj), max_a(traj), min_b(traj), max_b(traj)]

class QLearning:

    def __init__(self, graph, source, target, weight='weight', epsilon=0.1):
        self.graph = graph
        self.source = source
        self.target = target
        self.weight = weight
        self.epsilon = epsilon
        self.Q = {}

        self.lr = 1
        self.discount = 1

        self.state = self.source
        self.trajectory = [self.state]
        self.next_state = None

    def run(self):
        while self.state != self.target:
            self.choose_action(self.state)
            self.Q[(self.state, self.next_state)] = self.state_action_value + self.lr * self.temporal_difference
            self.state = self.next_state

    def reset(self):
        self.state = self.source
        self.trajectory = [self.state]
        self.next_state = None

    @property
    def state_action_value(self):
        return self.Q.get((self.state, self.next_state), 0)

    @property
    def reward(self):
        # return -self.graph[self.state][self.next_state][self.weight]
        return 0

    @property
    def temporal_difference(self):
        return self.reward + self.discount * self.utility(self.next_state) - self.state_action_value

    def utility(self, state):
        if state == self.target:
            u = {
                'ABDE': 100,
                'ACDE': 20,
                'ABCE': 10,
                'ACE': 20,
                'ABCE': 40,
                'ABCDE': 50,
            }
            return u[''.join(self.trajectory)]
            # return 10
        else:
            return self.max_Q(state)[0]

    def max_Q(self, state):
        # Find the maximum value in Q for a given (node, neighbor)
        max_q = -float('inf')
        max_neighbor = None
        for neighbor in self.graph.neighbors(state):
            q = self.Q.get((state, neighbor), 0)
            if q > max_q:
                max_q = q
                max_neighbor = neighbor
        return max_q, max_neighbor

    def choose_action(self, state):
        self.next_state = self.greedy_epsilon(state)
        if self.next_state is None:
            # Should never happen
            raise Exception("Target cannot be reached")
        self.trajectory.append(self.next_state)

    def greedy_epsilon(self, state):
        neighbors = list(self.graph.neighbors(state))
        if len(neighbors) == 0:
            return None
        
        # Choose a random neighbor
        if random.random() < self.epsilon:
            return random.choice(neighbors)

        # Choose the neighbor with the highest Q value
        max_neighbor = self.max_Q(state)[1]
        
        return max_neighbor

    def get_best_path(self):
        # Get path that maximizes Q at each step
        path = [self.source]
        state = self.source
        while state != self.target:
            state = self.max_Q(state)[1]
            path.append(state)
        return path


class QLambdaLearning(QLearning):

    def __init__(self, graph, source, target, weight='weight', epsilon=0.1, lambd=0.9):
        super().__init__(graph, source, target, weight, epsilon)
        self.decay = lambd

    def run(self):
        eligibility = {}

        while self.state != self.target:
            self.choose_action(self.state)
            
            eligibility[(self.state, self.next_state)] = eligibility.get((self.state, self.next_state), 0) + 1

            for a, b in self.graph.edges:
                eligibility[(a,b)] = eligibility.get((a,b), 0) * self.decay * self.discount
                self.Q[(a,b)] = self.Q.get((a,b), 0) + self.lr * self.temporal_difference * eligibility[(a,b)]

            self.state = self.next_state


# Iterate 10000 epochs
if __name__=='__main__':
    import networkx as nx

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

    # Learn
    epochs = 1000
    env = QLambdaLearning(G, "A", "E")
    for i in range(epochs):
        env.run()
        env.reset()

    # Get the path
    path = env.get_best_path()
    print("Best path: {}".format(path))
    breakpoint()