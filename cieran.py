from planning import Planning
from ramping import Ramping
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from coloraide import Color
from matplotlib.colors import ListedColormap

class Cieran:
    # Uses the Planning and Rampling classes to draw a curve through waypoints, avoiding obstacles with radius r, and truncating the front and back of the curve
    def __init__(self, waypoints, obstacles, rad_obstacles, truncate_front=20, truncate_back=20):
        self.waypoints = waypoints
        self.obstacles = obstacles
        self.rad_obstacles = rad_obstacles
        self.truncate_front = truncate_front
        self.truncate_back = truncate_back

        # Create a planner and ramper
        self.planner = Planning(self.waypoints, self.obstacles, self.rad_obstacles, 1000)
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
        # In a 1x3 grid, plot the distances, L* values, and the interpolated curve in a 2D plot
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        # Make sure first subplot is a square
        axs[0].set_aspect('equal')

        axs[1].set_aspect(2)
        axs[2].set_aspect(1.5)

        fig.suptitle("Cieran's plot")

        # Compute distances between each point in self.ramper.path and plot the distances with a line chart
        distances = np.array([self.color_distance(self.ramper.path[i], self.ramper.path[i-1]) for i in range(1, len(self.ramper.path))])
        distances = len(distances) * distances
        arclength = np.sum(distances)   
        rmse = np.std(distances)
        axs[2].plot(distances)

        axs[2].set_title("RMS deviation from flat: %0.2f (%0.2f%%)"
                % (rmse, 100 * rmse / arclength))
        axs[2].set_xlabel("Point")
        axs[2].set_ylabel("Distance")

        # Set y axes from 0 to max distance
        axs[2].set_ylim(0, max(distances)*2)
        
        # Plot the L* values ranging from 0 to 100
        l_values = np.array([self.ramper.path[i][0] for i in range(0, len(self.ramper.path))])
        axs[1].plot(l_values)

        axs[1].set_title("L* values")
        axs[1].set_xlabel("Point")
        axs[1].set_ylabel("L* value")

        # Set y axes from 0 to 100
        axs[1].set_ylim(0, 100)
        # axs[1].show()

        # Plot the interpolated curve in matplotlib, projecting into 2D, showing only y and z
        # obstacles in red
        if len(self.planner.obstacles) > 0:
            obstacles_proj = np.array(self.planner.obstacles)[:, [1, 2]]
            axs[0].scatter(*zip(*obstacles_proj), c='red')

            # Draw a circle of radius rad_obstacles around each obstacle, with a dashed black line
            for obstacle in self.planner.obstacles:
                circle = plt.Circle((obstacle[1], obstacle[2]), self.rad_obstacles, color='black', fill=False, linestyle='dashed')
                axs[0].add_artist(circle)

        if len(self.planner.waypoints) > 0:
            # waypoints in their color values in cielab
            waypoints_proj = np.array(self.planner.path)[:, [1, 2]]
            axs[0].scatter(
                *zip(*waypoints_proj), 
                c=[Color("lab({}% {} {} / 1)".format(*centroid)).convert('srgb')[:3] for centroid in self.planner.path]
            )

        # path in green
        path_proj = np.array(self.ramper.path)[:, [1, 2]]
        axs[0].plot(*zip(*path_proj), c='green')

        axs[0].set_title("Interpolated color ramp")
        axs[0].set_xlabel("a*")
        axs[0].set_ylabel("b*")

        # Set x and y axes from -128 to 128
        axs[0].set_xlim(-100, 100) 
        axs[0].set_ylim(-100, 100)

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