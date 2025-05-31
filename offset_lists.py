import numpy as np
import matplotlib.pyplot as plt

# Update plot parameters (IEEE-style)
plt.rcParams.update({
    "text.usetex": False,  # Set to True if you have LaTeX installed
    "font.size": 10,  # IEEE column text is usually around 8-9 pt
    "font.family": 'serif',
    "axes.labelsize": 13,
    "axes.titlesize": 1,
    "legend.fontsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "lines.linewidth": 2,
    "lines.markersize": 4,
    "figure.dpi": 250,
})

# First distribution: points from -5 to +5 with step size 0.2
x1 = np.arange(-5, 5.2, 0.2)
y1 = np.zeros_like(x1)

# Second distribution: custom list
x2 = [
    -5, -4, -3.5, -3.25, -3, -2.75, -2.5, -2,
    -1.5, -1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1,
    1.5, 2, 2.5, 2.75, 2.875, 3, 3.125, 3.25, 3.375, 3.5,
    3.65, 3.8, 4, 4.5, 5
]
x2 = np.array(x2)
# 1 cm ~ 0.3937 inches on screen, but in data units we'll just offset visually
y2 = np.full_like(x2, -1.0)  # Offset downward

# Plot
plt.figure(figsize=(12, 2))
plt.plot(x1, y1, 'o', label='Regular offset list', color='darkblue')
plt.plot(x2, y2, 'o', label='Optimized offset list', color='royalblue')

# Formatting
plt.yticks([])  # Hide y-axis
plt.xticks([-5,-4,-3,-2,-1,0,1,2,3,4,5])
plt.xlabel("Value")
plt.xlabel('Δω [ppm]')
plt.grid(True, axis='x', linestyle='--', alpha=0.5)
plt.axvline(x=3, color='grey', linestyle='--', linewidth=0.9, alpha=0.9)
plt.ylim(-2, 1.5)
plt.gca().invert_xaxis()
plt.legend()

import os 
plot_name = str("offset_distribution")
my_path = r"c:\asb\ntnu\plotting\master_thesis_pdf"
save_path = os.path.join(my_path, plot_name + ".pdf")
plt.savefig(save_path, format='pdf', bbox_inches='tight')

plt.show()
