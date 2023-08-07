# Creating 3D animated trajectory of the Earth as it orbits the Sun
# Using Semi-Implicit Euler Integration Method looping for calculation over 1000 data points
# Output file saved in "earth_trajectory.gif"

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Constants
G = 39.478  # Gravitational constant in (AU^3)/(Msun * yr^2)
M_SUN = 1.0  # Mass of the Sun in solar masses

def F_ES(r_E):
    # Calculate the gravitational force on Earth exerted by the Sun
    r_norm = np.linalg.norm(r_E)
    return -G * M_SUN / r_norm**3 * r_E

def integrate_earth(tmax, dt=1e-3):
    # Initial conditions
    r_E = np.array([1.0, 0.0, 0.0])  # Initial position of Earth (distance from Sun to Earth in AU)
    v_E = np.array([0.0, 2.0 * np.pi, 0.0])  # Initial velocity of Earth (perpendicular to the position vector in AU/yr)

    # Lists to store the trajectory
    trajectory_r_E = [r_E]

    # Integration loop
    t = 0
    while t < tmax:
        F = F_ES(r_E)
        v_E_next = v_E + dt * F
        r_E_next = r_E + dt * v_E_next
        trajectory_r_E.append(r_E_next)
        r_E = r_E_next
        v_E = v_E_next
        t += dt

    return np.array(trajectory_r_E)

# Time interval and time step
tmax = 1.0  # One year in years (since we are using AU and solar masses)
dt = 1e-3  # 1 millisecond in years

# Calculate the trajectory of Earth
trajectory = integrate_earth(tmax, dt)

# Extract x, y, and z coordinates from the trajectory
x_coords = trajectory[:, 0]
y_coords = trajectory[:, 1]
z_coords = trajectory[:, 2]

# Create a function to update the plot in each animation frame
def update(frame):
    ax.cla()  # Clear the current axis without clearing the whole figure
    ax.set_xlabel('x (AU)')
    ax.set_ylabel('y (AU)')
    ax.set_zlabel('z (AU)')
    ax.set_title('Trajectory of Earth')

    # Plot the Sun
    ax.scatter(0, 0, 0, c='orange', marker='o', label='Sun')

    # Plot the trajectory
    ax.plot(x_coords[:frame], y_coords[:frame], z_coords[:frame], label='Earth')

    # Plot the starting point (red triangle)
    ax.scatter(x_coords[frame-1], y_coords[frame-1], z_coords[frame-1], c='red', marker='^', label='Starting Point (Earth)')
    ax.legend(loc='upper right')  # Move legend to the top right

# Create the 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ani = FuncAnimation(fig, update, frames=len(x_coords), interval=50, repeat=False)

# Save the animation as a series of images using Pillow
ani.save('earth_trajectory.gif', writer='pillow')

# Display the animation
plt.show()

