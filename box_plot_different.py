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

# Prepare box plot data
boxplot_data = []
boxplot_labels = []

# Loop through all dataset pairs again for boxplot
for regular_path, optimized_path, label in datasets:
    reg = np.loadtxt(regular_path) * 100
    opt = np.loadtxt(optimized_path) * 100

    boxplot_data.append(reg)
    boxplot_labels.append(f'{label}')

    boxplot_data.append(opt)
    boxplot_labels.append(f'{label}')

# Plot
plt.figure(figsize=(6, 6))
box = plt.boxplot(boxplot_data, patch_artist=True)

# Color the boxes based on the metabolite
for patch, label in zip(box['boxes'], boxplot_labels):
    metab = label.split('_')[0]
    patch.set_facecolor(metab_colors.get(metab, 'gray'))
    patch.set_alpha(0.5)

# Customizing the plot
for i in range(len(boxplot_labels)):
    if i%2:
        boxplot_labels[i] = str(boxplot_labels[i]) + str(' opt')
    else: 
        boxplot_labels[i] = str(boxplot_labels[i]) + str(' reg')
plt.xticks(ticks=np.arange(1, len(boxplot_labels) + 1), labels=boxplot_labels, rotation=45)
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
