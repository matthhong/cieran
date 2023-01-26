from planning import Planning
from ramping import Ramping
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from coloraide import Color
from matplotlib.colors import ListedColormap

class Cieran:
    # Uses the Planning and Rampling classes to draw a curve through waypoints, avoiding obstacles with radius r, and truncating the front and back of the curve
    def __init__(self, waypoints, obstacles=[], rad_obstacles=0, truncate_front=20, truncate_back=20, min_c=0):
        self.waypoints = waypoints
        self.obstacles = obstacles
        self.rad_obstacles = rad_obstacles
        self.truncate_front = truncate_front
        self.truncate_back = truncate_back
        self.min_c = min_c

        # Create a planner and ramper
        self.planner = Planning(self.waypoints, self.obstacles, self.rad_obstacles, 1000, min_c=self.min_c)
        self.ramper = Ramping(self.planner.get_path(), self.truncate_front, self.truncate_back)

        # Execute the ramper
        self.ramper.execute()

        # Convert to hex using Coloraide
        path = [self.lab_to_rgb(p).to_string(hex=True) for p in self.ramper.path]

        # convert to ListedColormap
        self.cmap = ListedColormap(path)
        

    def lab_to_rgb(self, lab):
        # Convert a CIELAB value to an RGB value
        return Color("lab({}% {} {} / 1)".format(*lab)).convert("srgb")

    
    def plot_distances(self):
        # Compute distances between each point in self.ramper.path and plot the distances with a line chart
        distances = np.array([self.color_distance(self.ramper.path[i], self.ramper.path[i-1]) for i in range(1, len(self.ramper.path))])
        distances = len(distances) * distances
        arclength = np.sum(distances)
        rmse = np.std(distances)
        plt.plot(distances)

        plt.title("Length: %0.1f\nRMS deviation from flat: %0.1f (%0.1f%%)"
              % (arclength, rmse, 100 * rmse / arclength))
        plt.xlabel("Point")
        plt.ylabel("Distance")

        # Set y axes from 0 to max distance
        plt.ylim(0, max(distances))
        plt.show()

    def plot_l_diff(self):
        # Compute the difference between the L values of each point in self.ramper.path and plot the differences with a line chart
        l_diff = np.array([self.ramper.path[i][0] - self.ramper.path[i-1][0] for i in range(1, len(self.ramper.path))])
        l_diff = len(l_diff) * l_diff
        total_l_diff = np.sum(l_diff)
        rmse = np.std(l_diff)
        plt.plot(l_diff)

        plt.title("Total L* difference: %0.1f\nRMS deviation from flat: %0.2f (%0.2f%%)"
                % (total_l_diff, rmse, 100 * rmse / total_l_diff))
        plt.xlabel("Point")
        plt.ylabel("L* difference")

        # Set y axes from 0 to max distance
        plt.ylim(0, max(l_diff) * 2)
        plt.show()

    def plot_l_values(self):
        # Plot the L* values ranging from 0 to 100
        l_values = np.array([self.ramper.path[i][0] for i in range(0, len(self.ramper.path))])
        plt.plot(l_values)

        plt.title("L* values")
        plt.xlabel("Point")
        plt.ylabel("L* value")

        # Set y axes from 0 to 100
        plt.ylim(0, 100)
        plt.show()
        

    def color_distance(self, p1, p2):
        # breakpoint()
        return Color("lab({}% {} {} / 1)".format(*p1)).delta_e(Color("lab({}% {} {} / 1)".format(*p2)), method='2000')


    def plot_all(self):
        import matplotlib.gridspec as gridspec

        fig = plt.figure(figsize=(15, 5))

        # Create a 3x3 grid of subplots using GridSpec
        gs = gridspec.GridSpec(3, 3,
                            width_ratios=[1, 1, 1],
                            height_ratios=[1, 1, 1]
                            )

        ax1 = fig.add_subplot(gs[0:, 0])

        # Create 3 subplots stacked vertically in the second column
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 1])
        ax4 = fig.add_subplot(gs[2, 1])

        ax5 = fig.add_subplot(gs[0:, 2])

        fig.suptitle("Cieran's plot")

        # Compute distances between each point in self.ramper.path and plot the distances with a line chart
        distances = np.array([self.color_distance(self.ramper.path[i], self.ramper.path[i-1]) for i in range(1, len(self.ramper.path))])
        distances = len(distances) * distances
        arclength = np.sum(distances)   
        rmse = np.std(distances)
        ax5.plot(distances)

        ax5.set_title("Flatness of perceptual differences: %0.2f%%"
                % (100 - (100 * rmse / arclength)))
        ax5.set_xlabel("Point")
        ax5.set_ylabel("Distance")

        # Set y axes from 0 to max distance
        ax5.set_ylim(0, max(distances)*2)
        
        # Plot the L* values ranging from 0 to 100
        l_values = np.array([self.ramper.path[i][0] for i in range(0, len(self.ramper.path))])
        ax2.plot(l_values)

        ax2.set_title("L* values")
        ax2.set_xlabel("Point")
        ax2.set_ylabel("L* value")

        # Set y axes from 0 to 100
        ax2.set_ylim(0, 100)

        a_values = np.array([self.ramper.path[i][1] for i in range(0, len(self.ramper.path))])
        b_values = np.array([self.ramper.path[i][2] for i in range(0, len(self.ramper.path))])

        # Convert a and b values to polar coordinates
        c_values = np.sqrt(a_values**2 + b_values**2)
        h_values = np.arctan2(b_values, a_values)

        # Plot the c values ranging from 0 to 150
        ax3.plot(c_values)

        ax3.set_title("c* values")
        ax3.set_xlabel("Point")
        ax3.set_ylabel("c* value")

        # Set y axes from -150 to 150
        ax3.set_ylim(0, 150)

        # Plot the h values
        ax4.plot(h_values)

        ax4.set_title("h* values")
        ax4.set_xlabel("Point")
        ax4.set_ylabel("h* value")

        # Set y axes from  -pi to pi
        ax4.set_ylim(-np.pi, np.pi)

        # Plot the interpolated curve in matplotlib, projecting into 2D, showing only y and z
        # obstacles in red
        if len(self.planner.obstacles) > 0:
            obstacles_proj = np.array(self.planner.obstacles)[:, [1, 2]]
            ax1.scatter(*zip(*obstacles_proj), c='red')

            # Draw a circle of radius rad_obstacles around each obstacle, with a dashed black line
            for obstacle in self.planner.obstacles:
                circle = plt.Circle((obstacle[1], obstacle[2]), self.rad_obstacles, color='black', fill=False, linestyle='dashed')
                ax1.add_artist(circle)

        if len(self.planner.waypoints) > 0:
            # waypoints in their color values in cielab
            waypoints_proj = np.array(self.planner.path)[:, [1, 2]]
            ax1.scatter(
                *zip(*waypoints_proj), 
                c=[Color("lab({}% {} {} / 1)".format(*centroid)).convert('srgb')[:3] for centroid in self.planner.path]
            )

        # path in green
        path_proj = np.array(self.ramper.path)[:, [1, 2]]
        ax1.plot(*zip(*path_proj), c='green')

        ax1.set_title("Interpolated color ramp")
        ax1.set_xlabel("a*")
        ax1.set_ylabel("b*")

        # Set x and y axes from -128 to 128
        ax1.set_xlim(-100, 100) 
        ax1.set_ylim(-100, 100)

        plt.show()

    def plot(self):
        # Plot the interpolated curve in matplotlib 3D
        
        fig = plt.figure()

        ax = fig.add_subplot(111, projection='3d')

        # obstacles in red
        ax.scatter(*zip(*self.planner.obstacles), c='r')

        # get indices of waypoints in samples
        waypoint_indices = [np.where((self.planner.samples == waypoint).all(axis=1))[0][0] for waypoint in self.planner.path]

        # mask samples to only include samples that are not waypoints
        samples = self.planner.samples[~np.isin(np.arange(len(self.planner.samples)), waypoint_indices)]

        # samples in gray
        # ax.scatter(*zip(*samples), c='gray', opacity=0.1)

        # waypoints in their color values in cielab
        ax.scatter(
            *zip(*self.planner.path), 
            c=[Color("lab({}% {} {} / 1)".format(*centroid)).convert('srgb')[:3] for centroid in self.planner.path]
        )

        # path in green
        ax.plot(*zip(*self.ramper.path), c='g')
        
        plt.show()    