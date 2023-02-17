import numpy as np

def max_slope_difference(trajectory):
    a_slopes = []
    b_slopes = []
    for i in range(len(trajectory) - 1):
        a_slopes.append((trajectory[i+1][1] - trajectory[i][1]) / (trajectory[i+1][0] - trajectory[i][0]))
        b_slopes.append((trajectory[i+1][2] - trajectory[i][2]) / (trajectory[i+1][0] - trajectory[i][0]))
    
    # Compute differences in slope
    a_diffs = []
    b_diffs = []
    for i in range(len(a_slopes) - 1):
        a_diffs.append(abs(a_slopes[i+1] - a_slopes[i]))
        b_diffs.append(abs(b_slopes[i+1] - b_slopes[i]))

    try:
        return max([max(a_diffs), max(b_diffs)]) / 127
    except ValueError:
        return 0

def rate_of_change_polar(trajectory):
    # convert first and second coordinates to polar
    polar_traj = []
    for point in trajectory:
        chroma = np.sqrt(point[1]**2 + point[2]**2)
        polar_traj.append([point[0], chroma])

    slopes = []
    for i in range(len(trajectory) - 1):
        slopes.append((polar_traj[i+1][1] - polar_traj[i][1]) / (polar_traj[i+1][0] - polar_traj[i][0]))
    
    return slopes

def acceleration_polar(trajectory):
    slopes = rate_of_change_polar(trajectory)
    accels = []
    for i in range(len(slopes) - 1):
        accels.append(slopes[i+1] - slopes[i])
    
    return accels

def max_derivatives_polar(trajectory):
    polar_traj = []
    for point in trajectory:
        chroma = np.sqrt(point[1]**2 + point[2]**2)
        polar_traj.append([point[0], chroma])

    slopes = []
    for i in range(len(trajectory) - 1):
        slopes.append((polar_traj[i+1][1] - polar_traj[i][1]) / (polar_traj[i+1][0] - polar_traj[i][0]))

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

    return [max_slope/127, max_accel/127]

def chroma(trajectory):
    chroma = []
    for point in trajectory:
        chroma.append(np.sqrt(point[1]**2 + point[2]**2))
    # Divide by 150
    return [c/150 for c in chroma]

def max_derivatives(trajectory):
    # FIX: This is not working, only capturing the a* values

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

    return [max_slope/127, max_accel/127]

def min_max(trajectory, accessor):
    #min is either 0 or the actual min
    min_val = 0
    if min([accessor(point) for point in trajectory]) < 0:
        min_val = min([accessor(point) for point in trajectory])
    return [abs(min_val)/128, max([accessor(point) for point in trajectory])/127]

def feature_func(trajectory):
    a_list = min_max(trajectory, accessor=lambda point: point[1])
    b_list = min_max(trajectory, accessor=lambda point: point[2])
    derivs = max_derivatives(trajectory)
    c_list = chroma(trajectory)

    return np.array([derivs[0], derivs[1], a_list[0], a_list[1], b_list[0], b_list[1], np.mean(c_list), max(c_list)])

    