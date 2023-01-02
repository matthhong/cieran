

    
class Ramping:

    def __init__(self, start, waypoints, end, prop_control_points, truncate_front, truncate_back):
        self.start = start
        self.waypoints = waypoints
        self.end = end
        self.prop_control_points = prop_control_points
        self.truncate_front = truncate_front
        self.truncate_back = truncate_back

        self.control_points = []
        self.ramp = None

    def generate_control_points(self):
        # add intermediate points along the path given prop_control_points
        # add start, waypoints, and end
        pass

    def interpolate(self):
        # interpolate between points using cubic splines with centripetal parameterization
        # save the ramp
        self.generate_control_points()
        pass

    def truncate(self):
        # truncate the ramp at the start and end
        pass

    def render(self):
        # render the ramp
        if self.ramp is None:
            self.interpolate()
            self.truncate()

        return self.ramp