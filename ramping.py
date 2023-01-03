
from geomdl import fitting
from geomdl.visualization import VisMPL as vis
import numpy as np
import math
    
class Ramping:

    def __init__(self, control_points, truncate_front=20, truncate_back=20):
        self.truncate_front = truncate_front
        self.truncate_back = truncate_back

        self.control_points = control_points
        self.ramp = None
        self.path = []

    def generate_control_points(self):
        # add intermediate points along the path given prop_control_points
        # add start, waypoints, and end
        pass

    def interpolate(self):
        # interpolate between points using cubic splines with centripetal parameterization
        # save the ramp
        self.generate_control_points()

        # Do global curve interpolation
        self.ramp = fitting.interpolate_curve(self.control_points, 3, centripetal=True)

    # def truncate(self):
    #     # truncate the ramp at the start and end given truncate_front and truncate_back (in percent)
        
    #     # truncate the front
    #     # breakpoint()
    #     self.ramp.knotvector = self.ramp.knotvector[round(self.truncate_front * len(self.ramp.knotvector)):len(self.ramp.knotvector) - round(self.truncate_back * len(self.ramp.knotvector))]

    def execute(self):
        # execute the ramp
        self.interpolate()
        # self.truncate()
    
        # Get points from the interpolated geomdl ramp
        for i in range(len(self.ramp.evalpts)):
            self.path.append(self.ramp.evalpts[i][0:2])

        # Truncate the front and back of the path
        self.path = self.path[round(self.truncate_front * len(self.path)):len(self.path) - round(self.truncate_back * len(self.path))]


# Test the Ramping class using points from planning.py
if __name__ == '__main__':
    from planning import Planning
    from ramping import Ramping

    # Test the Ramping class with the Planning class
    waypoints = [[20, 0, 0], [70, 0, 0]]
    obstacles = [[50, 0, 0]]
    planner = Planning(waypoints, obstacles, 10, 1000)
    path = planner.get_path()
    ramper = Ramping(path, truncate_front=0, truncate_back=0)
    
    ramper.execute()

    # Plot the interpolated curve in matplotlib 3D
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')

    # obstacles in red
    ax.scatter(*zip(*planner.obstacles), c='r')

    # waypoints in blue
    ax.scatter(*zip(*planner.waypoints), c='b')

    # get indices of waypoints in samples
    waypoint_indices = [np.where((planner.samples == waypoint).all(axis=1))[0][0] for waypoint in planner.waypoints]

    # mask samples to only include samples that are not waypoints
    samples = planner.samples[~np.isin(np.arange(len(planner.samples)), waypoint_indices)]

    # samples in gray
    ax.scatter(*zip(*samples), c='gray')

    # path in green
    ax.plot(*zip(*ramper.path), c='g')
    
    plt.show()
