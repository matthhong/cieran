import numpy as np
from scipy.interpolate import splprep, splev

class Bspline:
    def __init__(self, points, alpha=0.5, degree=3):
        self.points = points
        self.alpha = alpha
        self.degree = degree

        # Compute the knots and control points
        tck, u = splprep(points, k=degree, s=alpha)[:3]
        self.knots, self.control_points, self.degree = tck

    def evaluate(self, t):
        """Evaluate the B-spline curve at a given set of parameter values."""
        return splev(t, (self.knots, self.control_points, self.degree))

class CustomBspline:
    def __init__(self, points, alpha=0.5, degree=3):
        self.points = points
        self.alpha = alpha
        self.degree = degree

        # Compute the knots and control points
        self.knots, self.control_points = self._interpolate()

    def _interpolate(self):
        """Interpolate the B-spline curve through the given points."""
        num_points = self.points.shape[0]
        num_knots = num_points + self.degree + 1

        breakpoint()

        # Compute the knots using exponential parametrization
        knots = np.zeros(num_knots)
        knots[:self.degree+1] = 0
        knots[-self.degree-1:] = 1
        knots[self.degree+1:-self.degree-1] = self._centripetal_parameterization()

        breakpoint()
        # Compute the coefficient matrix using the De Boor algorithm
        C = self._de_boor(self.degree, knots, self.points)

        # Solve for the control points
        control_points = np.linalg.solve(C, self.points)

        return knots, control_points

    def _centripetal_parameterization(self):
        """Compute the knots of the B-spline curve using the centripetal parametrization method."""
        num_points = len(self.points)
        knots = np.zeros(num_points)
        knots[0] = 0
        for i in range(1, num_points):
            knots[i] = knots[i-1] + ((self.points[i] - self.points[i-1]) ** self.alpha) ** 0.5
        knots /= knots[-1]
        return knots

    def _de_boor(p, knots, data_points):
        # Initialize the coefficient matrix
        m = len(data_points) - 1
        n = len(knots) - 1
        C = np.zeros((m+1, n+1))
        
        # Compute the coefficient vectors for the data points and interior knots
        for i in range(m+1):
            c = np.zeros(n+1)
            u = data_points[i][0] if i < m else knots[i]
            for j in range(p+1):
                c[j] = 1
            for i in range(p+1, n+1):
                for j in range(p, i):
                    c[j] += c[j-1] * (u - knots[j]) / (knots[j+p-i+1] - knots[j])
            C[i] = c

    def evaluate(self, t):
        """Evaluate the B-spline curve at a given set of parameter values."""
        # Initialize the curve to zero
        curve = np.zeros((len(t), 2))

        # Evaluate the curve using the control points and the de Boor algorithm
        for i in range(len(self.knots) - self.degree - 1):
            curve += self._de_boor(i, self.degree, self.knots, self.control_points)

        return curve


if __name__ == '__main__':
    # Create a B-spline curve
    points = np.random.rand(2, 6)

    # for a in [0, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100]:
    curve = Bspline(points, degree=5, alpha=0.5)

    # Evaluate the curve at 100 points
    t = np.linspace(0, 1, 100)
    x, y = curve.evaluate(t)

    # Plot the results
    import matplotlib.pyplot as plt
    plt.plot(x, y, 'r')
    plt.plot(points[0], points[1], 'bo')
    plt.show()