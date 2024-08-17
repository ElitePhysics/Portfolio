# This is to be the 3D animated trajectory of the Earth as it orbits the Sun, and the Moon as it orbits the Earth around the Sun.
# This is intended to add the trajectory of the Moon to the Earth-Sun system as programmed in the file
# titled Earth-Sun.py
# This is WIP

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

def F_EM(r_M, r_E):
    # Calculate the gravitational force on the Moon exerted by the Earth
    r_norm = np.linalg.norm(r_M - r_E)
    return G * M_EARTH * M_MOON / r_norm**3 * (r_E - r_M)

def integrate_earth_moon(tmax, dt=1e-3):
    # Initial conditions for the Earth
    r_E = np.array([1.0, 0.0, 0.0])  # Initial position of Earth (distance from Sun to Earth in AU)
    v_E = np.array([0.0, 2.0 * np.pi, 0.0])  # Initial velocity of Earth (perpendicular to the position vector in AU/yr)

    # Initial conditions for the Moon (relative to the Earth)
    r_M_relative = np.array([R_EARTH_MOON, 0.0, 0.0])  # Initial position of the Moon relative to Earth in AU
    v_M_relative = np.array([0.0, V_EARTH_MOON, 0.0])  # Initial velocity of the Moon relative to Earth in AU/yr

    # Lists to store the trajectory of Earth and the Moon
    trajectory_r_E = [r_E]
    trajectory_r_M = [r_E + r_M_relative]  # Initialize the Moon's position relative to the Sun

    # Integration loop
    t = 0
    while t < tmax:
        # Calculate forces on the Earth and the Moon
        F_E = F_ES(r_E)
        F_M = F_EM(trajectory_r_M[-1], r_E)

        # Update Earth's position and velocity
        v_E_next = v_E + dt * F_E
        r_E_next = r_E + dt * v_E_next

        # Update Moon's position and velocity relative to Earth
        v_M_relative_next = v_M_relative + dt * F_M / M_MOON
        r_M_relative_next = r_M_relative + dt * v_M_relative_next

        # Update Moon's absolute position (relative to the Sun)
        r_M_next = r_E_next + r_M_relative_next

        # Append positions to the trajectories
        trajectory_r_E.append(r_E_next)
        trajectory_r_M.append(r_M_next)

        # Update variables for the next iteration
        r_E = r_E_next
        v_E = v_E_next
        r_M_relative = r_M_relative_next
        v_M_relative = v_M_relative_next

        t += dt

    return np.array(trajectory_r_E), np.array(trajectory_r_M)


# Time interval and time step
tmax = 1.0  # One year in years (since we are using AU and solar masses)
dt = 1e-3  # 1 millisecond in years

# Calculate the trajectory of Earth and Moon
trajectory_earth, trajectory_moon = integrate_earth_moon(tmax, dt)

# Extract x, y, and z coordinates from the trajectories
x_coords = trajectory_earth[:, 0]
y_coords = trajectory_earth[:, 1]
z_coords = trajectory_earth[:, 2]

moon_x_coords = trajectory_moon[:, 0]
moon_y_coords = trajectory_moon[:, 1]
moon_z_coords = trajectory_moon[:, 2]

# ... (The rest of your animation code remains the same)

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
