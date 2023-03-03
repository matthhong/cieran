
import numpy as np
from coloraide import Color
from scipy.stats.qmc import Halton
from sklearn.neighbors import KDTree


def in_gamut(l, a, b):
    return Color("lab({}% {} {} / 1)".format(l, a, b)).in_gamut('srgb')

def hex_to_lab(hex):
    return Color(hex).convert('lab')._coords[:3]

dimensions = [(0,100), (-128,127), (-128,127)]

# Initialize the Halton sampler
sampler = Halton(len(dimensions), optimization='lloyd', seed=4)

# Generate samples
samples = sampler.random(8074)

# Map the samples of size (num_samples, 3) with values between 0 and 1 to the desired dimensions across the 3 axes
samples = np.array([dimensions[i][0] + (dimensions[i][1] - dimensions[i][0]) * samples[:, i] for i in range(len(dimensions))]).T

points = np.empty((0,3))
# Remove samples that are outside gamut or in collision with obstacles (circles of radius obstacle_rad)
for i in range(len(samples)):
    if in_gamut(*samples[i]):
        points = np.append(points, [samples[i]], axis=0)

CAT_COLORS_FILE = 'categorical_colors.txt'

def load_cat_colors():
    # Each line is an array of hex values
    # convert them to LAB and load into np array
    cat_colors = np.empty((0,3))
    with open(CAT_COLORS_FILE, 'r') as f:
        for line in f:
            values =  line.replace('\n','').split(',')
            for hex in values:
                lab = hex_to_lab(hex.replace('"', '').replace('[', '').replace(']', '')) # remove quotes and brackets
                cat_colors = np.append(cat_colors, [lab], axis=0)
    
    # Find the closest point in the points array for each color in cat_colors
    tree = KDTree(points)
    dist, ind = tree.query(cat_colors)

    return ind

ind = load_cat_colors()

NUM_RAMPS_FILE = 'ramps_per_color2.txt'

# Load the number of ramps and filter by ind
def load_num_ramps():
    num_ramps = []
    with open(NUM_RAMPS_FILE, 'r') as f:
        for i, line in enumerate(f):
            if i in ind:
                num_ramps.append([i, int(line.split(' ')[3].strip('\n'))])

    # breakpoint()
    # # Filter num_ramps if the number of ramps is more than 20
    # num_ramps = list(set([x[0] for x in num_ramps if x[1] >= 20]))

    colors = points[num_ramps]
    hex_values = [Color("lab({}% {} {} / 1)".format(*c[0])).convert('srgb').to_string(hex=True) for c in colors]
    breakpoint()
    
    # Write the hex values to a file
    with open('hex_values.txt', 'w') as f:
        for hex in hex_values:
            f.write(hex + '\n')
    
    # Draw a histogram of the number of ramps
    import matplotlib.pyplot as plt
    plt.hist(num_ramps, bins=range(1, 200))
    plt.xlabel('Number of ramps')
    plt.ylabel('Number of colors')
    plt.show()

load_num_ramps()
