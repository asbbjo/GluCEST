import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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

# Define file info: (regular_path, optimized_path, metabolite_label)
datasets = [
    (r'c:\asb\ntnu\semesters\v25\CEST_code\Bland_altman_files\flattened_Glu_reg.txt', r'c:\asb\ntnu\semesters\v25\CEST_code\Bland_altman_files\flattened_Glu_opt.txt', 'Glu'),
    (r'c:\asb\ntnu\semesters\v25\CEST_code\Bland_altman_files\flattened_Gln_reg.txt', r'c:\asb\ntnu\semesters\v25\CEST_code\Bland_altman_files\flattened_Gln_opt.txt', 'Gln'),
    (r'c:\asb\ntnu\semesters\v25\CEST_code\Bland_altman_files\flattened_GABA_reg.txt', r'c:\asb\ntnu\semesters\v25\CEST_code\Bland_altman_files\flattened_GABA_opt.txt', 'GABA'),
    (r'c:\asb\ntnu\semesters\v25\CEST_code\Bland_altman_files\flattened_NAA_reg.txt', r'c:\asb\ntnu\semesters\v25\CEST_code\Bland_altman_files\flattened_NAA_opt.txt', 'NAA'),
    (r'c:\asb\ntnu\semesters\v25\CEST_code\Bland_altman_files\flattened_Cr_reg.txt', r'c:\asb\ntnu\semesters\v25\CEST_code\Bland_altman_files\flattened_Cr_opt.txt', 'Cr'),
    (r'c:\asb\ntnu\semesters\v25\CEST_code\Bland_altman_files\flattened_Taurine_reg.txt', r'c:\asb\ntnu\semesters\v25\CEST_code\Bland_altman_files\flattened_Taurine_opt.txt', 'Taurine'),    
]

# Assign a unique color to each metabolite
metab_colors = {
    'Glu': 'red',
    'Gln': 'blue',
    'GABA': 'orange',
    'NAA': 'green',
    'Cr': 'purple',
    'Taurine': 'brown',
}

# Prepare containers
all_means = []
all_diffs = []
colors = []

# Loop through all dataset pairs
for regular_path, optimized_path, label in datasets:
    reg = np.loadtxt(regular_path)
    opt = np.loadtxt(optimized_path)
    
    mean = ((reg + opt) / 2) * 100
    diff = (reg - opt) * 100
    
    all_means.append(mean)
    all_diffs.append(diff)
    
    color = metab_colors.get(label, 'black')  # fallback to black if not found
    colors.extend([color] * len(mean))

# Concatenate all data
all_means = np.concatenate(all_means)
all_diffs = np.concatenate(all_diffs)

# Statistics
mean_diff = np.mean(all_diffs)
std_diff = np.std(all_diffs)

# Create plot
plt.figure(figsize=(6, 6))

# Scatter plot with colors
for mean_val, diff_val, c in zip(all_means, all_diffs, colors):
    plt.scatter(mean_val, diff_val, color=c, alpha=0.5, s=10)

# Add statistical lines
plt.axhline(mean_diff, color='black', linestyle='--', label='Mean difference')
plt.axhline(mean_diff + 1.96 * std_diff, color='gray', linestyle='--', label='±1.96 SD')
plt.axhline(mean_diff - 1.96 * std_diff, color='gray', linestyle='--')
print(mean_diff, std_diff*1.96)

# Labels and title
plt.xlabel('Mean gluCEST effect [%]')
plt.ylabel('Difference of gluCEST effect [%]')
#plt.title('Comparison of the regular and the optimized offset list')

# Axis formatting
plt.ylim([-2, 3])
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}'))

# Custom legend for metabolites
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label=metab,
           markerfacecolor=color, markersize=6, alpha=0.5)
    for metab, color in metab_colors.items()
]
legend_elements.append(Line2D([], [], color='black', linestyle='--', label='Mean difference'))
legend_elements.append(Line2D([], [], color='gray', linestyle='--', label='±1.96 SD'))

plt.legend(handles=legend_elements, loc='upper right')

xrange = 10         
yrange = 5
aspect_ratio = xrange / yrange
plt.gca().set_aspect(aspect_ratio, adjustable='box')

import os 

# Grid and layout
plt.grid(True, which='both', linestyle='--', linewidth=0.3, color='lightgrey', alpha=0.7)
plot_name = str("Bland_Altman_different")
my_path = r"c:\asb\ntnu\plotting\master_thesis_pdf\stats"
save_path = os.path.join(my_path, plot_name + ".pdf")
plt.savefig(save_path, format='pdf', bbox_inches='tight')
plt.show()
