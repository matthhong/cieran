
FILE_NAME = 'ramps_per_color2.txt'
import numpy as np
from coloraide import Color

def load_data():
    with open(FILE_NAME, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        lines = [line.split(' ') for line in lines]
        lines = [[(float(l), float(a), float(b)), int(count)] for l, a, b, count in lines]
        lines = np.array(lines)
        return lines
    
def load_data_hsl():
    with open(FILE_NAME, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        lines = [line.split(' ') for line in lines]
        lines = [Color("lab({}% {} {} / 1)".format(l, a, b)).convert('hsl')._coords[:3] for l, a, b, count in lines]
        lines = np.array(lines)
        return lines
        
    
OUTPUT = 'seed_colors.txt'
def select_seed_colors():
    data = load_data()
    # Sort the data by the number of ramps
    # data = data[data[:,1].argsort()]
    # Select the top 512 colors with the most ramps
    # data = data[-512:]
    # Select only colors with at least 20 ramps
    data = data[data[:,1] >= 20]
    # Sort by luminance, then a, then b
    data = data[data[:,0].argsort(kind='mergesort')]

    # Convert to hex
    # data[:,0] = [Color("lab({}% {} {} / 1)".format(l, a, b)).convert('srgb').to_string(hex=True) for l, a, b in data[:,0]]

    # Write the data to a file
    with open(OUTPUT, 'w') as f:
        for line in data:
            f.write(str(line[0]).strip('(').strip(')') + '\n')
    
    
if __name__ == '__main__':
    hsl_data = load_data_hsl()

    # Visualize three histograms
    # In the first one, x-axis is hue value and y-axis is the number of colors with that hue value
    # In the second one, x-axis is saturation value and y-axis is the number of colors with that saturation value
    # In the third one, x-axis is lightness value and y-axis is the number of colors with that lightness value
    # import matplotlib.pyplot as plt
    # fig, axs = plt.subplots(3)
    # axs[0].hist(hsl_data[:,0], bins=range(0, 360 + 1, 10))
    # axs[0].set_xlabel('Hue')
    # axs[0].set_ylabel('Number of colors')

    # axs[1].hist(hsl_data[:,1], bins=np.arange(0, 1, 0.1))
    # axs[1].set_xlabel('Saturation')
    # axs[1].set_ylabel('Number of colors')

    # axs[2].hist(hsl_data[:,2], bins=np.arange(0, 1, 0.1))
    # axs[2].set_xlabel('Lightness')
    # axs[2].set_ylabel('Number of colors')
    # plt.show()

    # data = load_data()
    # select_seed_colors()
    # print(data.shape)
    # print(data)

    # # Visualize a histogram where x-axis is the count of the number of ramps (second column) and y-axis is the number of colors with that number of ramps
    # # Each bin width is 20
    # import matplotlib.pyplot as plt
    # plt.hist(data[:,1], bins=range(0, int(max(data[:,1])) + 1, 10))
    # plt.xlabel('Number of ramps')
    # plt.ylabel('Number of colors')

    # # Draw a reference line where half of the histogram is above and half is below
    # plt.axvline(x=np.median(data[:,1]), color='r', linestyle='--')
    
    # # Annotate the median
    # plt.annotate('Median: ' + str(np.median(data[:,1])), xy=(np.median(data[:,1]), 0), xytext=(np.median(data[:,1]), 0), color='r')
    # plt.show()