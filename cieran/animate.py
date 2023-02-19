import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Generate initial data
x = np.random.rand(100)
y = np.random.rand(100)
z = np.random.rand(100)
c = np.random.rand(100)

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(x, y, z, c=c)

# Define the update function
def update(i):
    # Generate new data
    x = np.random.rand(100)
    y = np.random.rand(100)
    z = np.random.rand(100)
    c = np.random.rand(100)

    # Update the scatter plot data
    scatter._offsets3d = (x, y, z)
    scatter.set_array(c)

    # Return the updated plot objects
    return scatter,

# Create the animation
ani = FuncAnimation(fig, update, frames=10, interval=1000, blit=True)

# Show the plot
plt.show()