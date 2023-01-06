import numpy as np
from matplotlib.lines import Line2D

class SingleBezierCurveModel:
    def __init__(self, control_points):
        self.method = CatmulClark
        self.control_points = np.array(control_points).T
        x, y = self.get_bezier_points()
        self.bezier_curve = Line2D(x, y)

    def get_bezier_points(self, num=200):
        return self.get_bezier_points_at(np.linspace(0, 1, num))

    def get_bezier_points_at(self, at, grid=1000):
        xp, yp = self.control_points
        return compute_bezier_points(xp, yp, at, self.method, grid=grid)

    def _refresh(self):
        x, y = self.get_bezier_points()
        self.bezier_curve.set_data(x, y)


def CatmulClark(points, at):
    points = np.asarray(points)

    fixed = list(range(0, len(points)*2, 2))  # fix all control points in position

    while len(points) < len(at):
        new_p = np.zeros((2 * len(points), 2))
        new_p[0] = points[0]
        new_p[-1] = points[-1]
        for i in fixed:
            new_p[i] = points[i//2]  # fixed control points are not modified
            
        # breakpoint()
        for i in range(1, len(new_p)-1):
            if i not in fixed:
                new_p[i] = 3/4. * points[(i-1)//2] + 1/4. * points[(i+1)//2]
        points = new_p

    xp, yp = zip(*points)
    xp = np.interp(at, np.linspace(0, 1, len(xp)), xp)
    yp = np.interp(at, np.linspace(0, 1, len(yp)), yp)
    return np.asarray(list(zip(xp, yp)))

def compute_bezier_points(xp, yp, at, method, grid=256):
    at = np.asarray(at)
    # The Bezier curve is parameterized by a value t which ranges from 0
    # to 1. However, there is a nonlinear relationship between this value
    # and arclength. We want to parameterize by t', which measures
    # normalized arclength. To do this, we have to calculate the function
    # arclength(t), and then invert it.
    t = np.linspace(0, 1, grid)

    arclength = compute_arc_length(xp, yp, method, t=t)   
    arclength /= arclength[-1]
    # Now (t, arclength) is a lookup table describing the t -> arclength
    # mapping. Invert it to get at -> t
    at_t = np.interp(at, arclength, t)
    # And finally look up at the Bezier values at at_t
    # (Might be quicker to np.interp againts x and y, but eh, doesn't
    # really matter.)

    return method(list(zip(xp, yp)), at_t).T

def compute_arc_length(xp, yp, method, t=None, grid=256):
    if t is None:
        t = np.linspace(0, 1, grid)
    x, y = method(list(zip(xp, yp)), t).T
    x_deltas = np.diff(x)
    y_deltas = np.diff(y)
    arclength_deltas = np.empty(len(x))
    if t.size == 0:
        return np.asarray([0])
    arclength_deltas[0] = 0
    np.hypot(x_deltas, y_deltas, out=arclength_deltas[1:])
    return np.cumsum(arclength_deltas)


# Test code
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal')

    control_points = [(-.5, 0), (-.5, .5), (.5, .5), (.5, 0)]
    model = SingleBezierCurveModel(control_points)

    # Draw the control points
    x, y = zip(*control_points)
    ax.plot(x, y, 'o')

    # Draw the Bezier curve
    ax.add_line(model.bezier_curve)

    plt.show()

    