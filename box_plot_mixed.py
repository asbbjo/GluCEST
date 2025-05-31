import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Update plot parameters (IEEE-style)
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.size": 14,
    "axes.labelsize": 7,
    "axes.titlesize": 7,
    "legend.fontsize": 6,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "lines.linewidth": 1,
    "lines.markersize": 4,
    "figure.dpi": 250,
})

# Define file info: (regular_path, optimized_path, metabolite_label)
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

# Assign a unique color to each metabolite
metab_colors = {
    '10Glu_Gln': 'red',
    '6Glu_Gln': 'red',
    '2Glu_Gln': 'red',
    '10Glu_GABA': 'blue',
    '6Glu_GABA': 'blue',
    '2Glu_GABA': 'blue',
}

# Prepare data
boxplot_data = []
boxplot_labels = []
boxplot_colors = []

import re

for regular_path, optimized_path, label in datasets:
    filename = regular_path.split('\\')[-1]
    
    conc_raw = filename.split('_')[1]  # e.g., '10Glu', '2Glu'
    conc_match = re.match(r'(\d+)', conc_raw)
    conc = conc_match.group(1) if conc_match else 'X'  # Get '2', '6', '10'


    # Determine color key
    if label == 'Gln':
        color_key = f'{conc}Glu_Gln'
    elif label == 'GABA':
        color_key = f'{conc}Glu_GABA'
    else:
        color_key = 'unknown'

    # Load data
    reg = np.loadtxt(regular_path) * 100
    opt = np.loadtxt(optimized_path) * 100

    # Append data and labels
    boxplot_data.append(reg)
    boxplot_labels.append(f'{color_key}_reg')
    boxplot_colors.append(metab_colors.get(color_key, 'gray'))

    boxplot_data.append(opt)
    boxplot_labels.append(f'{color_key}_opt')
    boxplot_colors.append(metab_colors.get(color_key, 'gray'))

# Plotting
plt.figure(figsize=(6, 6))
box = plt.boxplot(boxplot_data, patch_artist=True)

# Apply colors
for patch, color in zip(box['boxes'], boxplot_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.5)

# Final touches
plt.xticks(ticks=np.arange(1, len(boxplot_labels) + 1), labels=boxplot_labels, rotation=45)
plt.ylabel('gluCEST effect [%]')
#plt.title('Comparison of Regular and Optimized GluCEST Effects')

# Optional: Add grid
plt.grid(axis='y', linestyle='--', alpha=0.6)
xrange = 12         
yrange = 14
aspect_ratio = xrange / yrange
plt.gca().set_aspect(aspect_ratio, adjustable='box')


import os 

# Grid and layout
#plt.grid(True, which='both', linestyle='--', linewidth=0.3, color='lightgrey', alpha=0.7)
plot_name = str("box_plot_mixed")
my_path = r"c:\asb\ntnu\plotting\auto_save_png"
save_path = os.path.join(my_path, plot_name + ".png")
plt.savefig(save_path, format='png', bbox_inches='tight')
plt.show()
