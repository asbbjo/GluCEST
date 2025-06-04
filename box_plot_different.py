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
    "xtick.labelsize": 9,
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
    'Glu 10mM': 'red',
    'Gln 2mM': 'blue',
    'GABA 2mM': 'orange',
    'NAA 10mM': 'green',
    'Cr 6mM': 'purple',
    'Taurine 2mM': 'brown',
}

# Prepare box plot data
boxplot_data = []
boxplot_labels = []
boxplot_colors = []

import re

label_conc_map = {
    'Glu': '10mM',
    'Gln': '2mM',
    'GABA': '2mM',
    'NAA': '10mM',
    'Cr': '6mM',
    'Taurine': '2mM',
}

for regular_path, optimized_path, label in datasets:
    metab_label = f"{label} {label_conc_map.get(label, 'X')}"
    
    reg = np.loadtxt(regular_path) * 100
    opt = np.loadtxt(optimized_path) * 100

    boxplot_data.append(reg)
    boxplot_labels.append(f'{metab_label} reg')
    boxplot_colors.append(metab_colors.get(metab_label, 'gray'))

    boxplot_data.append(opt)
    boxplot_labels.append(f'{metab_label} opt')
    boxplot_colors.append(metab_colors.get(metab_label, 'gray'))




# Plot
plt.figure(figsize=(6, 6))
box = plt.boxplot(boxplot_data, patch_artist=True)

# Color the boxes based on the metabolite
# Apply colors
for patch, color in zip(box['boxes'], boxplot_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.5)

plt.xticks(ticks=np.arange(1, len(boxplot_labels) + 1), labels=boxplot_labels, rotation=45)
plt.gca().tick_params(axis='x', which='both', pad=5)  # Optional: adjust distance from axis
plt.setp(plt.gca().get_xticklabels(), ha='right', rotation=45, x=-0.05)  # shift left
plt.ylabel('gluCEST effect [%]')
#plt.title('Distribution of gluCEST effect for regular vs optimized offset lists')

# Optional: Add grid
plt.grid(axis='y', linestyle='--', alpha=0.6)
xrange = 12         
yrange = 12
aspect_ratio = xrange / yrange
plt.gca().set_aspect(aspect_ratio, adjustable='box')


import os 

# Grid and layout
plt.grid(True, which='both', linestyle='--', linewidth=0.3, color='lightgrey', alpha=0.7)
plot_name = str("box_plot_different")
my_path = r"c:\asb\ntnu\plotting\master_thesis_pdf\stats"
save_path = os.path.join(my_path, plot_name + ".pdf")
plt.savefig(save_path, format='pdf', bbox_inches='tight')
plt.show()
