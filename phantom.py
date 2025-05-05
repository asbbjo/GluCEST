import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Main cylinder
theta = np.linspace(0, 2*np.pi, 100)
z_main = np.linspace(0, 126, 100)
theta_grid, z_grid = np.meshgrid(theta, z_main)
x_main = 15 * np.cos(theta_grid)
y_main = 15 * np.sin(theta_grid)
ax.plot_surface(x_main, y_main, z_grid, alpha=0.2, color='blue')


# Small cylinder 
back_angle = 2*np.pi/3  
x_back = 10 * np.cos(back_angle)
y_back = 10 * np.sin(back_angle)
z_small = np.linspace(3, 113, 126)
theta_small = np.linspace(0, 2*np.pi, 100)
theta_grid_small, z_grid_small = np.meshgrid(theta_small, z_small)
x_small = 2.5 * np.cos(theta_grid_small) + x_back
y_small = 2.5 * np.sin(theta_grid_small) + y_back
ax.plot_surface(x_small, y_small, z_grid_small, alpha=0.6, color='white')

# Centered disk
z_disk = np.linspace(75, 77, 10)
theta_grid_disk, z_grid_disk = np.meshgrid(theta, z_disk)
x_disk = 15 * np.cos(theta_grid_disk)
y_disk = 15 * np.sin(theta_grid_disk)
ax.plot_surface(x_disk, y_disk, z_grid_disk, alpha=0.5, color='black')

# 6 Holes in disk - now shown in the plane
hole_angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
for angle in hole_angles:
    x_hole = 10 * np.cos(angle)  # 10mm radius for hole centers
    y_hole = 10 * np.sin(angle)
    # Draw each hole as a small cylinder
    z_hole = np.linspace(75, 77, 10)
    theta_hole = np.linspace(0, 2*np.pi, 20)
    theta_grid_hole, z_grid_hole = np.meshgrid(theta_hole, z_hole)
    x_hole_cyl = 2.5 * np.cos(theta_grid_hole) + x_hole
    y_hole_cyl = 2.5 * np.sin(theta_grid_hole) + y_hole
    ax.plot_surface(x_hole_cyl, y_hole_cyl, z_grid_hole, alpha=0.8, color='black')

# Centered disk
z_disk = np.linspace(0, 2, 10)
theta_grid_disk, z_grid_disk = np.meshgrid(theta, z_disk)
x_disk = 15 * np.cos(theta_grid_disk)
y_disk = 15 * np.sin(theta_grid_disk)
ax.plot_surface(x_disk, y_disk, z_grid_disk, alpha=0.5, color='black')

# 6 Holes in disk - now shown in the plane
hole_angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
for angle in hole_angles:
    x_hole = 10 * np.cos(angle)  # 10mm radius for hole centers
    y_hole = 10 * np.sin(angle)
    # Draw each hole as a small cylinder
    z_hole = np.linspace(0, 2, 10)
    theta_hole = np.linspace(0, 2*np.pi, 20)
    theta_grid_hole, z_grid_hole = np.meshgrid(theta_hole, z_hole)
    x_hole_cyl = 2.5 * np.cos(theta_grid_hole) + x_hole
    y_hole_cyl = 2.5 * np.sin(theta_grid_hole) + y_hole
    ax.plot_surface(x_hole_cyl, y_hole_cyl, z_grid_hole, alpha=0.8, color='black')

# Centered disk
z_disk = np.linspace(24, 26, 10)
theta_grid_disk, z_grid_disk = np.meshgrid(theta, z_disk)
x_disk = 15 * np.cos(theta_grid_disk)
y_disk = 15 * np.sin(theta_grid_disk)
ax.plot_surface(x_disk, y_disk, z_grid_disk, alpha=0.5, color='black')

# 6 Holes in disk - now shown in the plane
hole_angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
for angle in hole_angles:
    x_hole = 10 * np.cos(angle)  # 10mm radius for hole centers
    y_hole = 10 * np.sin(angle)
    # Draw each hole as a small cylinder
    z_hole = np.linspace(24, 26, 10)
    theta_hole = np.linspace(0, 2*np.pi, 20)
    theta_grid_hole, z_grid_hole = np.meshgrid(theta_hole, z_hole)
    x_hole_cyl = 2.5 * np.cos(theta_grid_hole) + x_hole
    y_hole_cyl = 2.5 * np.sin(theta_grid_hole) + y_hole
    ax.plot_surface(x_hole_cyl, y_hole_cyl, z_grid_hole, alpha=0.8, color='black')



ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.set_zlabel('Z (mm)')
ax.set_xlim([-70,70])
ax.set_ylim([-70,70])
ax.set_zlim([0,130])
plt.title('Cylinder with Perforated Disk and Internal Rod (Back Hole)')
plt.tight_layout()
plt.show()