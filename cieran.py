from planning import Planning
from ramping import Ramping

class Cieran:
    # Plan then ramp
    def __init__(self, start, waypoints, end, obstacles, obstacle_rad, num_samples, prop_control_points, truncate_front, truncate_back):
        self.start = start
        self.waypoints = waypoints
        self.end = end
        self.obstacles = obstacles
        self.obstacle_rad = obstacle_rad
        self.num_samples = num_samples

        self.prop_control_points = prop_control_points
        self.truncate_front = truncate_front
        self.truncate_back = truncate_back

        self.planner = Planning(start, waypoints, end, obstacles, obstacle_rad, num_samples)
        self.ramp = Ramping(self.planner.get_path(), start, waypoints, end, prop_control_points, truncate_front, truncate_back)


    def add_obstacle(self, obstacle):
        # add an obstacle to the planner
        self.obstacles.append(obstacle)


    def render(self, obstacle_rad, prop_control_points, truncate_front, truncate_back):
        # render the ramp
        return self.ramp.render()