
import cieran
import numpy as np
import matplotlib.pyplot as plt

from cieran import Environment
from coloraide import Color

def draw_chart(cmap):

    t = np.linspace(0, 2 * np.pi, 1024)
    data2d = np.sin(t)[:, np.newaxis] * np.cos(t)[np.newaxis, :]
    # Draw a chart of data2d with the given colormap
    fig, ax = plt.subplots()
    ax.imshow(data2d, cmap=cmap)

    plt.show()
    
if __name__ == "__main__":

    color = "#bb8bb0"
    env = Environment(color = Color(color).convert('lab')._coords[:-1])

    # import pickle
    # with open('test.pkl', 'wb') as f:

    breakpoint()

    import networkx as nx

    # Average number of neighbors per node
    graph = env.graph
    print("Average number of neighbors per node:", sum([len([neighbor for neighbor in graph.neighbors(node)]) for node in graph.nodes]) / len(graph.nodes))

    # Visualize the degree distribution
    degrees = [len([neighbor for neighbor in graph.neighbors(node)]) for node in graph.nodes]
    plt.hist(degrees, bins=range(1, max(degrees) + 1))
    plt.xlabel('Degree')
    plt.ylabel('Number of nodes')
    plt.show()

    # Reset figure
    plt.clf()

    # Visualize the states in 3D LAB space
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for ramp in env.fitted_ramps:
        ax.plot(ramp[:,0], ramp[:,1], ramp[:,2])

    # Display all nodes as scatter plot, in gray
    print("Number of nodes:", len(env.graph.nodes))

    print("Number of original colors:" , len(env.fitted_ramps * 9))
    for node in env.graph.nodes:
        # if not in any ramp
        if not any(np.all(node == ramp) for ramp in env.fitted_ramps):
            ax.scatter(node[0], node[1], node[2], c='gray', marker='o')
    
    # Label the axes
    ax.set_xlabel('L')
    ax.set_ylabel('A')
    ax.set_zlabel('B')

    plt.show()