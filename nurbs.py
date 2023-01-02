from geomdl import fitting
from geomdl.visualization import VisMPL as vis


if __name__ == '__main__':
    # The NURBS Book Ex9.1
    points = ((0, 0), (3, 4), (-1, 4), (-4, 0), (-4, -3))
    degree = 4  # cubic curve

    # Do global curve interpolation
    curve = fitting.interpolate_curve(points, degree, centripetal=True)

    # Plot the interpolated curve
    curve.delta = 0.01
    curve.vis = vis.VisCurve2D()
    curve.render()