import numpy as np
from coloraide import Color

RAMPS_FILE = 'ramps.csv'

def load_ramps():
    # Each line is a ramp, where each triplet is a 3D point of LAB color, and there are 9 points per ramp
    # load into np array
    ramps = np.empty((0,9,3))
    with open(RAMPS_FILE, 'r') as f:
        for line in f:
            values =  line.replace('\n','').split(',')
            ramp = np.empty((0,3))
            for i in range(0, len(values), 3):
                point = np.array([float(values[i]), float(values[i+1]), float(values[i+2])])
                ramp = np.append(ramp, [point], axis=0)
            # Sort the ramp by luminance in reverse order
            ramp = ramp[ramp[:,0].argsort()]
            ramps = np.concatenate((ramps, ramp[np.newaxis, ...]), axis=0)
    return ramps


def fit_ramp_to_color(ramp, lab_color):
    min_distance = 99999
    min_distance_index = -1
    distance = 0
    given_luminance = lab_color[0]

    for i in range(len(ramp)):
        distance = abs(given_luminance - ramp[i][0])
        if distance < min_distance:
            min_distance = distance
            min_distance_index = i

    closest_point = ramp[min_distance_index]

    # Compute translation vector to move the closest point to the given color
    translation = [0, 0, 0]
    translation[0] = closest_point[0] - lab_color[0]
    translation[1] = closest_point[1] - lab_color[1]
    translation[2] = closest_point[2] - lab_color[2]

    # difference_vector = [0, 0, 0]
    # new_start_color = [0, 0, 0]
    # for i in range(3):
    #     difference_vector[i] = closest_point[i] - lab_color[i]
    #     new_start_color[i] = ramp[0][i] - difference_vector[i]

    # new_ramp = translate_curve(ramp, new_start_color)

    new_ramp = []
    for point in ramp:
        new_ramp.append([point[0] - translation[0], point[1] - translation[1], point[2] - translation[2]])

    return new_ramp

 
def translate_curve(curve, starting_point):
    translation_vector_x = curve[0][0] - starting_point[0]
    translation_vector_y = curve[0][1] - starting_point[1]
    translation_vector_z = curve[0][2] - starting_point[2]

    translated_curve = []

    for point in curve:
        translated_point = []
        translated_point.append(round(point[0] - translation_vector_x, 2))
        translated_point.append(round(point[1] - translation_vector_y, 2))
        translated_point.append(round(point[2] - translation_vector_z, 2))

        translated_curve.append(translated_point)

    return translated_curve


def is_valid(curve):
    is_valid = True
    for color in curve:
        if color[0] < 0 or color[0] > 100:
            is_valid = False
        if color[1] < -128 or color[1] > 128:
            is_valid = False
        if color[2] < -128 or color[2] > 128:
            is_valid = False
    return is_valid

def in_gamut(curve):
    in_gamut = True
    for point in curve:
        color = Color("lab({}% {} {} / 1)".format(*point))
        if not color.in_gamut('srgb'):
            in_gamut = False
    return in_gamut


if __name__=='__main__':
    ramps = load_ramps()

    fitted_ramps = []
    out_of_gamut = []
    for i, ramp in enumerate(ramps):
        new_ramp = np.array(fit_ramp_to_color(ramp, [55.49, 74.81, 52.36]))
        if not is_valid(new_ramp) or not in_gamut(new_ramp):
            out_of_gamut.append(i)
        fitted_ramps.append(new_ramp)

    # print lengths of each array
    print('Ramps length: ' + str(len(ramps)))
    print('Fitted ramps length: ' + str(len(fitted_ramps)))

    # Visualize the states in 3D LAB space
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Add two figures side by side, one for the original ramps, and one for the fitted ramps
    fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={'projection': '3d'})

    for ramp in fitted_ramps:
        ax1.plot(ramp[:,0], ramp[:,1], ramp[:,2])

    # filter fitted ramps by indices in out_of_gamut
    out_ramps = np.array(fitted_ramps)[out_of_gamut]

    for ramp in ramps:
        ax2.plot(ramp[:,0], ramp[:,1], ramp[:,2])

    # Fix axes scales

    ax1.set_xlim(0, 100)
    ax1.set_ylim(-127, 127)
    ax1.set_zlim(-127, 127)

    ax2.set_xlim(0, 100)
    ax2.set_ylim(-127, 127)
    ax2.set_zlim(-127, 127)

    # Set labels
    ax1.set_xlabel('L')
    ax1.set_ylabel('A')
    ax1.set_zlabel('B')


    plt.show()