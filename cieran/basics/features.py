import numpy as np


def max_chroma(trajectory):
    chroma = []
    for point in trajectory:
        chroma.append(np.sqrt(point[1]**2 + point[2]**2))
    # Divide by 150
    return max([c/150 for c in chroma])


def max_accel(trajectory, accessor):

    slopes = []
    for i in range(len(trajectory) - 1):
        slopes.append(abs((trajectory[i+1][1] - trajectory[i][1]) / (trajectory[i+1][0] - trajectory[i][0])))

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

def distance(trajectory):
    # Compute euclidean distance between points
    return sum([np.sqrt((trajectory[i+1][0] - trajectory[i][0])**2 + (trajectory[i+1][1] - trajectory[i][1])**2 + (trajectory[i+1][2] - trajectory[i][2])**2) for i in range(len(trajectory) - 1)])


def min_max(trajectory, accessor):
    #min is either 0 or the actual min
    min_val = 0
    if min([accessor(point) for point in trajectory]) < 0:
        min_val = min([accessor(point) for point in trajectory])
    return [abs(min_val)/128, max([accessor(point) for point in trajectory])/127]


def feature_func(trajectory):
    # a_accel = max_accel(trajectory, accessor=lambda point: point[1])
    # b_accel = max_accel(trajectory, accessor=lambda point: point[2])
    dist = distance(trajectory)
    a_list = min_max(trajectory, accessor=lambda point: point[1])
    b_list = min_max(trajectory, accessor=lambda point: point[2])
    # max_c = max_chroma(trajectory)

    return np.array([dist, a_list[0], a_list[1], b_list[0], b_list[1]])

    