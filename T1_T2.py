import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": False,
    "font.size": 10,
    "axes.labelsize": 9,
    "axes.titlesize": 1,
    "legend.fontsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "lines.linewidth": 1,
    "lines.markersize": 1,
    "figure.dpi": 200,
})

# Time axis
t = np.linspace(0, 3000, 500)  # up to 5 seconds
T1 = 800  # seconds
T2 = 400  # seconds
M0 = 100  # equilibrium magnetization

# Magnetization equations for a 90° flip angle (cos(90°) = 0, sin(90°) = 1)
Mz = M0 * (1 - np.exp(-t / T1))           # Longitudinal recovery
Mxy = M0 * np.exp(-t / T2)                # Transverse decay

# Plotting
plt.figure(figsize=(7, 4))
plt.plot(t, Mz, label=r'$M_l$(t)', color='darkblue', lw=1.5)
plt.plot(t, Mxy, label=r'$M_{tr}$(t)', color='royalblue', lw=1.5)

# Reference lines
plt.axvline(T1, color='darkblue', linestyle='--', alpha=0.5)
plt.axvline(T2, color='royalblue', linestyle='--', alpha=0.5)

# Plot annotations
plt.text(T1 + 50, 55, r'63%', color='darkblue', fontsize=7)
plt.text(T2 + 50, 5, r'37%', color='royalblue', fontsize=7)

# Labels and styling
plt.xlabel('Time [ms]')
plt.ylabel('Magnetization [%]')
#plt.title('Relaxation of $T_1$ and $T_2$')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.3)

import os
# Save or show the plot
plot_name = str("T1_T2")
my_path = r"c:\asb\ntnu\plotting\master_thesis_pdf\theory"
save_path = os.path.join(my_path, plot_name + ".pdf")
plt.savefig(save_path, format='pdf', bbox_inches='tight')
plt.show()
