
import random
# Find shortest path in a graph using reinforcement learning
# QLearning: graph, source, target, weight, alpha, gamma, epsilon, lambd -> path


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

    def run(self):
        state = self.source

        while state != self.target:
            next_state = self.choose_action(state)
            if next_state is None:
                # Should never happen
                breakpoint()
                raise Exception("Target cannot be reached")

            reward = -self.graph[state][next_state][self.weight]

            # Update Q
            Q_value = self.Q.get((state, next_state), 0) 

            if next_state == self.target:
                utility = 10
            else:
                utility = self.max_Q(next_state)[0]

            Q_update = self.lr * (reward + self.discount * utility - self.Q.get((state, next_state), 0))
            self.Q[(state, next_state)] = Q_value + Q_update

            state = next_state

        return state

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
        return self.greedy_epsilon(state)

    def greedy_epsilon(self, state):
        # Choose a neighbor
        neighbors = list(self.graph.neighbors(state))
        if len(neighbors) == 0:
            return None
        
        # Choose a random neighbor
        if random.random() < self.epsilon:
            return random.choice(neighbors)

        # Choose the neighbor with the highest Q value
        max_q = -float('inf')
        max_neighbor = None
        for neighbor in neighbors:
            q = self.Q.get((state, neighbor), 0)
            if q > max_q:
                max_q = q
                max_neighbor = neighbor
        
        return max_neighbor

    def get_best_path(self):
        # Get path that maximizes Q at each step
        path = [self.source]
        state = self.source
        while state != self.target:
            state = self.max_Q(state)[1]
            path.append(state)
        return path


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
    env = QLearning(G, "A", "E")
    for i in range(epochs):
        env.run()

    # Get the path
    path = env.get_best_path()
    print("Best path: {}".format(path))