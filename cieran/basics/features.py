import numpy as np


def max_chroma(trajectory):
    chroma = []
    for point in trajectory:
        chroma.append(np.sqrt(point[1]**2 + point[2]**2))
    # Divide by 150
    return max([c/150 for c in chroma])

def mean_and_slope_chroma(trajectory):
    chroma = []
    for point in trajectory:
        chroma.append(np.sqrt(point[1]**2 + point[2]**2))

    # Fit a linear regression to the chroma values without using external libraries
    x = np.array([l for l,a,b in trajectory])
    y = np.array(chroma)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    numerator = sum([(x[i] - x_mean) * (y[i] - y_mean) for i in range(len(x))])
    denominator = sum([(x[i] - x_mean)**2 for i in range(len(x))])
    slope = numerator / denominator

    return y_mean/150, slope

def lightness_max_c(trajectory):
    chroma = []
    for point in trajectory:
        chroma.append(np.sqrt(point[1]**2 + point[2]**2))

    ind = np.argmax(chroma)
    return trajectory[ind][0]/100
    

def max_accel(trajectory, accessor):

    slopes = []
    for i in range(len(trajectory) - 1):
        slopes.append(abs(
            (trajectory[i+1][1] - trajectory[i][1]) / (trajectory[i+1][0] - trajectory[i][0])))

    max_slope = 0
    try:
        max_slope = max(slopes)
    except ValueError:
        pass

    accels = []
    for i in range(len(slopes) - 1):
        accels.append(slopes[i+1] - slopes[i])

    max_accel = 0
    try:
        max_accel = max(accels)
    except ValueError:
        pass

    return max_accel/127

def mean_stdev_theta(trajectory):
    # COnvert to polar coordinates
    theta = []
    for point in trajectory[1:-1]:
        theta.append(np.arctan2(point[2], point[1]))

    # Find the circular mean using arctan2
    a = np.mean([np.cos(t) for t in theta])
    b = np.mean([np.sin(t) for t in theta])
    mean_theta = np.arctan2(b, a)

    # Find the circular stdev using Yamartino method
    epsilon = np.sqrt(1-a**2-b**2)
    stdev_theta = np.arcsin(epsilon) * (1 + (((2 / np.sqrt(3) - 1)) * epsilon**3))
    return mean_theta/np.pi, 2*stdev_theta/np.pi


def distance(trajectory):
    # Compute euclidean distance between points
    dist = sum([np.sqrt((trajectory[i+1][0] - trajectory[i][0])**2 + (trajectory[i+1][1] - trajectory[i][1])**2 + (trajectory[i+1][2] - trajectory[i][2])**2) for i in range(len(trajectory) - 1)])
    # return dist/414.48168477836606
    return dist/255


# def min_max(trajectory, accessor):
#     # min is either 0 or the actual min
#     min_val = abs(min([accessor(point) for point in trajectory]))
#     max_val = max([accessor(point) for point in trajectory])
#     if min_val > max_val:
#         return -min_val/128
#     else:
#         return max_val/127
#     # return [abs(min_val)/128, max([accessor(point) for point in trajectory])/127]

# def value_range(trajectory, degree):
def value_range(trajectory, accessor):

    min_val = min([accessor(point) for point in trajectory])
    max_val = max([accessor(point) for point in trajectory])
    return (max_val - min_val) / 127

def value_range2(trajectory, degree):

    ab_values = [[point[1], point[2]] for point in trajectory]
    # Rotate ab_values by an amount of degree (in radians)
    ab_values = np.array(ab_values)
    ab_values = np.dot(ab_values, np.array([[np.cos(degree), -np.sin(degree)], [np.sin(degree), np.cos(degree)]]))
    return (max(ab_values[:, 0]) - min(ab_values[:, 0]))/127


def distance2(trajectory, degree):

    ab_values = [[point[1], point[2]] for point in trajectory]
    # Rotate ab_values by an amount of degree (in radians)
    ab_values = np.array(ab_values)
    ab_values = np.dot(ab_values, np.array([[np.cos(degree), -np.sin(degree)], [np.sin(degree), np.cos(degree)]]))
    
    corner = (-128, 127)
    dist = []
    for point in ab_values:
        dist.append(np.sqrt((corner[0] - point[0])**2 + (corner[1] - point[1])**2))
    return (min(dist)) / 180


def distance_from_corner(trajectory, corner):
    # Compute the distance from the corner (e.g. (-128, 127)) to each point in the trajectory and return the min
    dist = []
    for point in trajectory:
        dist.append(np.sqrt((corner[0] - point[1])**2 + (corner[1] - point[2])**2))
    return min(dist)/255


def feature_func(trajectory):
    # a_accel = max_accel(trajectory, accessor=lambda point: point[1])
    # b_accel = max_accel(trajectory, accessor=lambda point: point[2])
    # dist = distance(trajectory)
    # a_list = min_max(trajectory, accessor=lambda point: point[1])
    # b_list = min_max(trajectory, accessor=lambda point: point[2])
    # max_c = max_chroma(trajectory)
    mean_c, slope_c = mean_and_slope_chroma(trajectory)
    # mean_theta, stdev_theta = mean_stdev_theta(trajectory)
    # a_range = value_range(trajectory[1:-1], accessor=lambda point: point[1])
    # b_range = value_range(trajectory[1:-1], accessor=lambda point: point[2])
    # range1 = value_range2(trajectory, degree=0)
    # range2 = value_range2(trajectory, degree=np.pi/4)
    # range3 = value_range2(trajectory, degree=np.pi/2)
    # range4 = value_range2(trajectory, degree=3*np.pi/4)

    # return np.array([slope_c, range1, range2, range3, range4])

    # corner1 = distance_from_corner(trajectory, (127, 127))
    # corner2 = distance_from_corner(trajectory, (127, -128))
    # corner3 = distance_from_corner(trajectory, (-128, -128))
    # corner4 = distance_from_corner(trajectory, (-128, 127))
    # corner5 = distance_from_corner(trajectory, (0, 127))
    # corner6 = distance_from_corner(trajectory, (128, 0))
    # corner7 = distance_from_corner(trajectory, (0, -128))
    # corner8 = distance_from_corner(trajectory, (-128, 127))
    # return np.array([dist, max_c, mean_theta, stdev_theta])
    # return np.array([slope_c, stdev_theta, max_c, corner1, corner2, corner3, corner4])
    # return np.array([slope_c, a_range, b_range, corner1, corner2, corner3, corner4])

    corner1 = distance2(trajectory, 0)
    corner2 = distance2(trajectory, np.pi/4)
    corner3 = distance2(trajectory, np.pi/2)
    corner4 = distance2(trajectory, 3*np.pi/4)
    corner5 = distance2(trajectory, np.pi)
    corner6 = distance2(trajectory, 5*np.pi/4)
    corner7 = distance2(trajectory, 3*np.pi/2)
    corner8 = distance2(trajectory, 7*np.pi/4)

    return np.array([slope_c, corner1, corner2, corner3, corner4, corner5, corner6, corner7, corner8])
