import numpy as np
import matplotlib.pyplot as plt

# Set general IEEE-style parameters
plt.rcParams.update({
    "text.usetex": False,  # Set to True if you have LaTeX installed
    "font.family": "serif",
    "font.size": 14,  # IEEE column text is usually around 8-9 pt
    "axes.labelsize": 7,
    "axes.titlesize": 7,
    "legend.fontsize": 6,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "lines.linewidth": 1,
    "lines.markersize": 4,
    "figure.dpi": 250,
})

# Flatten the images
image1_flat = np.loadtxt(r'C:\asb\ntnu\master\GluCEST\flattened_glu1.txt')
image2_flat = np.loadtxt(r'C:\asb\ntnu\master\GluCEST\flattened_glu1_opt.txt')

# Compute mean and difference
mean_vals = (image1_flat + image2_flat) / 2
diff_vals = image1_flat - image2_flat

# Bland–Altman statistics
mean_diff = np.mean(diff_vals)
std_diff = np.std(diff_vals)

# Plot Bland–Altman
plt.figure(figsize=(6, 6))
plt.scatter(mean_vals, diff_vals, alpha=0.3, s=10)
plt.axhline(mean_diff, color='red', linestyle='--', label='Mean difference')
plt.axhline(mean_diff + 1.96 * std_diff, color='gray', linestyle='--', label='±1.96 SD')
plt.axhline(mean_diff - 1.96 * std_diff, color='gray', linestyle='--')
plt.xlabel('Mean of pixels (regular vs. optimized)')
plt.ylabel('Difference of pixels (regular vs. optimized)')
plt.title('Bland–Altman plot for 10Glu + 2Gln')
plt.ylim([-0.025,0.015])
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.3f}'))
plt.legend()
# Make axes box square in screen units
xrange = 0.018         
yrange = 0.04
aspect_ratio = xrange / yrange
plt.gca().set_aspect(aspect_ratio, adjustable='box')
plt.grid(True)
plt.show()
