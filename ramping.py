
from geomdl import fitting
from geomdl.visualization import VisMPL as vis
import numpy as np
import math
from coloraide import Color
    
class Ramping:

    def __init__(self, control_points, truncate_front=0.2, truncate_back=0.2):
        self.truncate_front = truncate_front
        self.truncate_back = truncate_back

        self.control_points = control_points
        self.ramp = None
        self.path = []

    def generate_control_points(self):
        pass

    def interpolate(self, mode='cubic'):
        # interpolate between points using cubic splines with centripetal parameterization
        # save the ramp
        self.generate_control_points()

        if mode == 'cubic':
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
        # However, the distance between points is not constant,
        # so we need to normalize the distance between points using arc length
        # We want to parameterize by t', which measures normalized arclength.

        # Get the points from the ramp
        t = np.linspace(self.truncate_front, 1-self.truncate_back, 1000)
        at = np.linspace(self.truncate_front, 1-self.truncate_back, 1000)
        points = self.ramp.evaluate_list(at)

        # Get the arc length of the ramp at each point using distance function
        arc_lengths = [0]
        for i in range(1, len(points)):
            arc_lengths.append(arc_lengths[i-1] + self.distance(points[i-1], points[i]))

        # Normalize the arc lengths
        arc_lengths = np.array(arc_lengths) / arc_lengths[-1]

        # Invert the arc lengths to get the parameterization
        at_t = np.interp(at, arc_lengths, t)

        # Get the points from the ramp using the parameterization
        self.path = self.ramp.evaluate_list(at_t)

        # Truncate the front and back of the path
        # self.path = self.path[round(self.truncate_front * len(self.path)):len(self.path) - round(self.truncate_back * len(self.path))]

    def distance(self, p1, p2):
        return Color("lab({}% {} {} / 1)".format(*p1)).delta_e(Color("lab({}% {} {} / 1)".format(*p2)), method='2000')


# Test the Ramping class using points from planning.py
if __name__ == '__main__':
    from planning import Planning
    from ramping import Ramping

    # [69.2, -7.569, -24.114]
    # Test the Cieran class
    # waypoints = [[26.6128, 37.85, -44.51], [90, -20, -13]]
    # waypoints = [[26.6128, 37.85, -44.51]]
    # obstacles = [[76, -9, -20]]
    waypoints = [[26.6128, 37.85, -44.51], [69.2, -7.569, -24.114]]
    obstacles = []

    planner = Planning(waypoints, obstacles, 20, 1000, 0)
    path = planner.get_path()
    ramper = Ramping(path, truncate_front=0, truncate_back=0)
    
    ramper.execute()

    # Plot the interpolated curve in matplotlib 3D
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')

    # obstacles in red
    if len(planner.obstacles) > 0:
        ax.scatter(*zip(*planner.obstacles), c='r')

    # # path in blue
    # ax.plot(*zip(*planner.path), c='b')

    # get indices of waypoints in samples
    waypoint_indices = [np.where((planner.samples == waypoint).all(axis=1))[0][0] for waypoint in planner.path]

    # mask samples to only include samples that are not waypoints
    samples = planner.samples[~np.isin(np.arange(len(planner.samples)), waypoint_indices)]

    # samples in gray
    ax.scatter(*zip(*samples), c='gray')

    # waypoints in blue
    ax.scatter(*zip(*planner.path), c='b')

    # path in green
    ax.plot(*zip(*ramper.path), c='g')

    # Label the axes
    ax.set_xlabel('L*')
    ax.set_ylabel('a*')
    ax.set_zlabel('b*')
    
    plt.show()
