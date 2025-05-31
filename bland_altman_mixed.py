import numpy as np
import matplotlib.pyplot as plt

# Update plot parameters (IEEE-style)
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.size": 14,
    "axes.labelsize": 10,
    "axes.titlesize": 7,
    "legend.fontsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "lines.linewidth": 1,
    "lines.markersize": 4,
    "figure.dpi": 250,
})

# Define file info: (regular_path, optimized_path, label)
datasets = [
    # 2 mM
    (r'c:\asb\ntnu\semesters\v25\CEST_code\Bland_altman_files\flattened_2Glu_Gln_reg.txt', r'c:\asb\ntnu\semesters\v25\CEST_code\Bland_altman_files\flattened_2Glu_Gln_opt.txt', 'Gln'),
    (r'c:\asb\ntnu\semesters\v25\CEST_code\Bland_altman_files\flattened_2Glu_GABA_reg.txt', r'c:\asb\ntnu\semesters\v25\CEST_code\Bland_altman_files\flattened_2Glu_GABA_opt.txt', 'GABA'),

    # 6 mM
    (r'c:\asb\ntnu\semesters\v25\CEST_code\Bland_altman_files\flattened_6Glu_Gln_reg.txt', r'c:\asb\ntnu\semesters\v25\CEST_code\Bland_altman_files\flattened_6Glu_Gln_opt.txt', 'Gln'),
    (r'c:\asb\ntnu\semesters\v25\CEST_code\Bland_altman_files\flattened_6Glu_GABA_reg.txt', r'c:\asb\ntnu\semesters\v25\CEST_code\Bland_altman_files\flattened_6Glu_GABA_opt.txt', 'GABA'),

    # 10 mM
    (r'c:\asb\ntnu\semesters\v25\CEST_code\Bland_altman_files\flattened_10Glu_Gln_reg.txt', r'c:\asb\ntnu\semesters\v25\CEST_code\Bland_altman_files\flattened_10Glu_Gln_opt.txt', 'Gln'),
    (r'c:\asb\ntnu\semesters\v25\CEST_code\Bland_altman_files\flattened_10Glu_GABA_reg.txt', r'c:\asb\ntnu\semesters\v25\CEST_code\Bland_altman_files\flattened_10Glu_GABA_opt.txt', 'GABA'),
]

# Prepare containers
all_means = []
all_diffs = []
colors = []

# Loop through all dataset pairs
for regular_path, optimized_path, label in datasets:
    reg = np.loadtxt(regular_path)
    opt = np.loadtxt(optimized_path) 
    mean = ((reg + opt) / 2)*100
    diff = (reg - opt)*100

    all_means.append(mean)
    all_diffs.append(diff)

    # Color: red for Gln, blue for GABA
    color = 'red' if label == 'Gln' else 'blue'
    colors.extend([color] * len(mean))

# Concatenate all results
all_means = np.concatenate(all_means) 
all_diffs = np.concatenate(all_diffs) 

# Calculate overall stats
mean_diff = np.mean(all_diffs)
std_diff = np.std(all_diffs)

# Create plot
plt.figure(figsize=(6, 6))

# Plot with correct colors
for mean_val, diff_val, c in zip(all_means, all_diffs, colors):
    plt.scatter(mean_val, diff_val, color=c, alpha=0.5, s=10)

# Add statistical lines
plt.axhline(mean_diff, color='black', linestyle='--', label='Mean difference')
plt.axhline(mean_diff + 1.96 * std_diff, color='gray', linestyle='--', label='±1.96 SD')
plt.axhline(mean_diff - 1.96 * std_diff, color='gray', linestyle='--')

print(mean_diff, 1.96*std_diff)

# Labels and styling
plt.xlabel('Mean gluCEST effect [%]')
plt.ylabel('Difference of gluCEST effect [%]')
#plt.title('Comparison of the regular and the optimized offset list')
plt.ylim([-0.025, 0.015])
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}'))
plt.ylim([-2.5, 2.5])

# Create a dummy legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Glu+Gln', markerfacecolor='red', markersize=6, alpha=0.5),
    Line2D([0], [0], marker='o', color='w', label='Glu+GABA', markerfacecolor='blue', markersize=6, alpha=0.5),
    Line2D([], [], color='black', linestyle='--', label='Mean difference'),
    Line2D([], [], color='gray', linestyle='--', label='±1.96 SD'),
]
plt.legend(handles=legend_elements)

import os

# Aspect ratio and grid
xrange = 13       
yrange = 5
aspect_ratio = xrange / yrange
plt.gca().set_aspect(aspect_ratio, adjustable='box')
plt.grid(True, which='both', linestyle='--', linewidth=0.3, color='lightgrey', alpha=0.7)
plot_name = str("Bland_Altman_mixed")
my_path = r"c:\asb\ntnu\plotting\master_thesis_pdf\stats"
save_path = os.path.join(my_path, plot_name + ".pdf")
plt.savefig(save_path, format='pdf', bbox_inches='tight')
# Show
plt.show()
