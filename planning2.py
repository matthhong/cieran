import numpy as np
from networkx import DiGraph, astar_path, dijkstra_path
from coloraide import Color
from scipy.stats.qmc import Halton
from KDTree import KDTree
from colorspace import color_distance, lab_to_rgb

# This planning algorithm optimizes a motion plan for a virtual robot using gradient descent
# This robot has a 3D position and 2D orientation
# The robot can move in any direction, but cannot move backwards in x
# The robot can rotate in any direction, but cannot rotate backwards

# Planning: waypoints -> path
class AnglePlanning:
    def __init__(self, waypoints, zeta=3, learning_rate=0.1, num_samples=20000):
        self.waypoints = waypoints
        self.learning_rate = learning_rate
        self.zeta = zeta
        self.path = None

        self.current_plan = None

        ## Initialize configuration space using Halton sequence
        # dimensions = [(0,100), (-128,128), (-128,128), (-np.pi/2, np.pi/2), (-np.pi/2, np.pi/2)]

        # Create a discretized and valid configuration space
        # Looping through each angular dimension
        # Robot moves a unit distance in the direction of its orientation

        self.distance_rate = 20

        # Create a graph
        self.graph = DiGraph()
        self.create_configurations()

        self.configuration_space = np.array([x for x in self.graph.nodes])

        #visualize the configuration space
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # config space as points
        ax.scatter(self.configuration_space[:,0], self.configuration_space[:,1], self.configuration_space[:,2], c='b', marker='o')

        # label axes
        ax.set_xlabel('L*')
        ax.set_ylabel('a*')
        ax.set_zlabel('b*')

        # set axis limits
        ax.set_xlim(0, 100)
        ax.set_ylim(-128, 128)
        ax.set_zlim(-128, 128)

        plt.show()

        # Find the shortest path
        self.initialize_plan()

    def create_configurations(self):

        stack = [np.zeros(5)]

        while stack:
            current = stack.pop()
            # Randomly sample an angle
            tried = 0
            found = 0

            while True:
                theta, phi = np.random.uniform(-np.pi/2, np.pi/2, size=2)

                # Create a new configuration
                new_config = np.zeros(5)

                # Compute orientation by step size, which is pi/12, and the range is -np.pi/2, np.pi/2
                new_config[3] = theta
                new_config[4] = phi

                # Set the new current to a unit distance in the direction of the orientation
                new_config[2] = current[2] + self.distance_rate * np.sin(theta) * np.cos(phi)
                new_config[1] = current[1] + self.distance_rate * np.sin(theta) * np.sin(phi)
                new_config[0] = current[0] + self.distance_rate * np.cos(theta)

                if self.in_gamut(*new_config[:3]) is False:
                    tried += 1
                    if tried > 100:
                        break
                    continue
                # Add the new configuration to the configuration space
                self.graph.add_edge(tuple(current), tuple(new_config), weight=self.cost_function(current, new_config))
                stack.append(new_config)
                found += 1

                if found > 1:
                    break

            print(len(stack))

            # if len(stack) > 250:
            #     breakpoint()

    def cost_function(self, fro, to):
        # Convert from LAB to LCH
        lab = to[:3]
        lch = np.zeros(3)
        # polar coordinates
        lch[0] = lab[0]
        lch[1] = np.sqrt(lab[1]**2 + lab[2]**2)
        lch[2] = np.arctan2(lab[2], lab[1])

        # Get angular velocity between from and to
        angular_velocity_cost = np.linalg.norm(to[3:] - fro[3:])**2

        return self.distance_rate + angular_velocity_cost + self.zeta * lch[1]

        
    def in_gamut(self, l, a, b):
        return Color("lab({}% {} {} / 1)".format(l, a, b)).in_gamut('srgb')

                    
    def initialize_plan(self):
        # Initialize the plan with the shortest path
        self.current_plan = dijkstra_path(self.graph, tuple(self.config_space[0]), tuple(self.config_space[-1]))
    

# Test
if __name__ == "__main__":
    waypoints = [[20, 0, 0], [70, 0, 0]]
    planner = AnglePlanning(waypoints)

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # waypoints in blue
    ax.scatter(*zip(*planner.waypoints), c='b')
    
    ax.plot(*zip(*planner.current_plan), c='g')

    plt.show()