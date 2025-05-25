import os
from pathlib import Path
from scipy.interpolate import interp1d, UnivariateSpline
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import numpy as np
import pydicom
import pypulseq as pp
from csaps import csaps
import scipy as sc

# Store dicoms of each offsets in one folder
# Create a sequence file for the acquisition
# Run with pypulseq==1.4.2 and pydicom==3.0.1
# Make sure to have Grassroots DICOM (pip install gdcm) and pylibjpeg (pip install pylibjpeg pylibjpeg-libjpeg)

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

def ppval(p, x):
    # helper function to evaluate piecewise polinomial
    if callable(p):
        return p(x)
    else:
        n = len(p) - 1
        result = np.zeros_like(x)
        for i in range(n, -1, -1):
            result = result * x + p[i]
        return result

def EVAL_GluCEST(data_path, seq_path):
    import pypulseq as pp # the import over is not found
    seq = pp.Sequence()
    print('--- Reading the sequence protocol ---')
    seq.read(seq_path)
    '''seq.plot(time_range=[0, 0.05])'''  # Plot the sequence protocol. Adjust the time range as needed (in seconds). May need to downgrade to pypulse=1.3.1post1

    offsets = seq.get_definition("offsets_ppm")
    m0_offset = seq.get_definition("M0_offset")
    n_meas = len(offsets)

    dcmpath = data_path
    os.chdir(dcmpath)

    #read data from dicom directory
    collection = [pydicom.dcmread(os.path.join(dcmpath, filename)) for filename in sorted(os.listdir(dcmpath))]
    # extract the volume data
    V = np.stack([dcm.pixel_array for dcm in collection])
    V = np.transpose(V[:,:,:,-1], (1, 2, 0)) # erase the last dimention due to jpeg format ([52,128,128,3] => [52,128,128])
    sz = V.shape
    V = np.reshape(V, [sz[0], sz[1], n_meas, sz[2] // n_meas]).transpose(0, 1, 3, 2)

    # Vectorization
    threshold = 100 #np.max(V)*0.1 # threshold of 10%
    mask = np.squeeze(V[:, :, :, 0]) > threshold
    mask_idx = np.where(mask.ravel())[0]
    V_m_z = V.reshape(-1, n_meas).T
    m_z = V_m_z[:, mask_idx]

    M0_idx = np.where(abs(offsets) >= abs(m0_offset))[0]   
    if len(M0_idx) > 0:
        M0 = np.mean(m_z[M0_idx, :], 0)
        offsets = np.delete(offsets, M0_idx)
        m_z = np.delete(m_z, M0_idx, axis=0)
        Z = m_z / M0  # Normalization
    else:
        print("m0_offset not found in offset")
    
    print('--- B0 correction of data ---')
    Z_corr = np.zeros_like(Z)
    w = offsets
    dB0_stack = np.zeros(Z.shape[1])
    for ii in range(Z.shape[1]):
        if np.all(np.isfinite(Z[:, ii])):
            pp = csaps(w, Z[:, ii], smooth=0.95)
            w_fine = np.arange(-1, 1.005, 0.005)
            z_fine = ppval(pp, w_fine)
    
            min_idx = np.argmin(z_fine)
            dB0_stack[ii] = w_fine[min_idx]
    
            Z_corr[:, ii] = ppval(pp, w + dB0_stack[ii])

    # calculation of MTRasym spectrum
    Z_ref = Z_corr[::-1, :]
    MTRasym = Z_ref - Z_corr

    # Vectorization Backwards
    if Z.shape[1] > 1:
        V_MTRasym = np.zeros((V_m_z.shape[0], V_m_z.shape[1]), dtype=float)
        V_MTRasym[1:, mask_idx] = MTRasym
        V_MTRasym_reshaped = V_MTRasym.reshape(
            V.shape[3], V.shape[0], V.shape[1], V.shape[2]
        ).transpose(1, 2, 3, 0)
    
        V_Z_corr = np.zeros((V_m_z.shape[0], V_m_z.shape[1]), dtype=float)
        V_Z_corr[1:, mask_idx] = Z_corr
        V_Z_corr_reshaped = V_Z_corr.reshape(
            V.shape[3], V.shape[0], V.shape[1], V.shape[2]
        ).transpose(1, 2, 3, 0)

    # Choose pixels for ROI
    pixels_dict = { 
        '10glu 2gln': [76,81,75,80],    # 250410
        '6glu 2gln': [57,62,83,88],     # 250410
        '2glu 2gln': [42,47,70,75],     # 250410
        '10glu 2gaba': [46,51,51,56],   # 250410
        '6glu 2gaba': [64,69,44,49],    # 250410
        '2glu 2gaba': [79,84,56,61],    # 250410
    }

    print('--- Plotting GluCEST images ---')
    slice_of_interest = 0 # pick slice for evaluation (0 if only one slice)
    desired_offset = 3 # pick offset for evaluation (3 for GluCEST at 3 ppm)
    offset_of_interest = np.where(offsets == desired_offset)[0]  
    w_offset_of_interest = offsets[offset_of_interest]

    fig, ax = plt.subplots(figsize=(5, 5)) 
    vmin, vmax = 0.5, 1  # Z-spectra range
    im = ax.imshow(V_Z_corr_reshaped[:, :, slice_of_interest, offset_of_interest],vmin=vmin, vmax=vmax, cmap='rainbow')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = plt.colorbar(im, cax=cax, format="%.2f")
    cb.set_ticks(np.linspace(vmin, vmax, 5)) 
    ax.set_title("Z(Δω) = %.2f ppm" % w_offset_of_interest)
    #plt.show()

    pixels_glu1 = pixels_dict.get('2glu 2gln')
    array_MTR1 = V_MTRasym_reshaped[pixels_glu1[0]:pixels_glu1[1],pixels_glu1[2]:pixels_glu1[3],slice_of_interest,1:] # 1: to remove the M0 scan
    flattened_vectors_MTR_glu1 = array_MTR1.reshape(-1, array_MTR1.shape[-1]) 

    pixels_glu2 = pixels_dict.get('2glu 2gaba')
    array_MTR2 = V_MTRasym_reshaped[pixels_glu2[0]:pixels_glu2[1],pixels_glu2[2]:pixels_glu2[3],slice_of_interest,1:] # 1: to remove the M0 scan
    flattened_vectors_MTR_glu2 = array_MTR2.reshape(-1, array_MTR2.shape[-1]) 

    # For Bland Altman plotting
    '''glu1 = V_MTRasym_reshaped[pixels_glu1[0]:pixels_glu1[1],pixels_glu1[2]:pixels_glu1[3],slice_of_interest,offset_of_interest]
    glu2 = V_MTRasym_reshaped[pixels_glu2[0]:pixels_glu2[1],pixels_glu2[2]:pixels_glu2[3],slice_of_interest,offset_of_interest]

    flatten_glu1 = glu1.flatten()
    flatten_glu2 = glu2.flatten()

    # Save to text file
    np.savetxt(r'C:\asb\ntnu\master\GluCEST\flattened_glu1.txt', flatten_glu1, fmt="%.6f")  # or fmt="%d" for integers
    np.savetxt(r'C:\asb\ntnu\master\GluCEST\flattened_glu2.txt', flatten_glu2, fmt="%.6f")  # or fmt="%d" for integers'''

    main_path = data_path[-22:-18] + str('_') + data_path[-8:]

    MTR_max1 = np.max(flattened_vectors_MTR_glu1)
    MTR_max2 = np.max(flattened_vectors_MTR_glu2)
    MTR_max = max(MTR_max1, MTR_max2)

    fig, ax = plt.subplots(figsize=(5, 5)) 
    vmin, vmax = 0, 0.18 # set GluCEST contrast range
    im = ax.imshow(V_MTRasym_reshaped[:,:,slice_of_interest,offset_of_interest], vmin=vmin, vmax=vmax, cmap='OrRd')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = plt.colorbar(im, cax=cax, format="%.2f")
    cb.set_ticks(np.linspace(vmin, vmax, 5)) 
    #ax.set_title("MTRasym(Δω) = %.2f ppm" % w_offset_of_interest)
    plot_name = main_path + str("_MTR_map")
    my_path = r"c:\asb\ntnu\plotting\auto_save_png\mixed"
    save_path = os.path.join(my_path, plot_name + ".png")
    plt.savefig(save_path, format='png', bbox_inches='tight')
    #plt.show()

    # Choose metabolites
    label_names = ['10Glu 2Gln', '6Glu 2Gln', '2Glu 2Gln', '10Glu 2GABA', '6Glu 2GABA', '2Glu 2GABA']
    m_avg = []
    m_sem = []

    print('--- Plotting GluCEST spectra ---')
    plt.figure(figsize=(5, 5))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(label_names)))

    for i, label in enumerate(label_names):
        key = label.lower()
        pixels_metabolite = pixels_dict.get(key)

        # Spectrum handling phantom
        array_Z = V_Z_corr_reshaped[pixels_metabolite[0]:pixels_metabolite[1],pixels_metabolite[2]:pixels_metabolite[3],slice_of_interest,1:] # 1: to remove the M0 scan
        flattened_vectors_Z = array_Z.reshape(-1, array_Z.shape[-1]) 
        Z_spectrum = flattened_vectors_Z.mean(axis=0)

        # Get statistics
        V_MTRasym_reshaped_pc = V_MTRasym_reshaped*100
        m_roi = V_MTRasym_reshaped_pc[pixels_metabolite[0]:pixels_metabolite[1],pixels_metabolite[2]:pixels_metabolite[3], slice_of_interest, offset_of_interest]
        avg, sem = np.mean(m_roi.reshape(-1)), sc.stats.sem(m_roi.reshape(-1))
        m_avg.append(avg)
        m_sem.append(sem)

        plt.axvline(x=3, color='grey', linestyle='--', linewidth=0.8, alpha=0.7)
        plt.xlim([-5, 5])
        plt.ylim([0.12,1.1])
        plt.plot(w, Z_spectrum, marker='o', markersize=2, label=label_names[i], color=colors[i])
        plt.xlabel('Frequency offset Δω [ppm]')
        plt.ylabel(r'$S_{\mathrm{sat}}/S_{\mathrm{0}}$')
        plt.gca().invert_xaxis()
        plt.grid(True, which='both', linestyle='--', linewidth=0.3, color='lightgrey', alpha=0.7)
        #plt.title("Z-spectra for mixed metabolites")
        # Make axes box square in screen units
        xrange = 10       
        yrange = 1.1 - 0.12
        aspect_ratio = xrange / yrange
        plt.gca().set_aspect(aspect_ratio, adjustable='box')

    plt.legend(loc='lower right')
    plot_name = main_path + str("_Z_spectra")
    my_path = r"c:\asb\ntnu\plotting\auto_save_png\mixed"
    save_path = os.path.join(my_path, plot_name + ".png")
    plt.savefig(save_path, format='png', bbox_inches='tight')
    #plt.show()

    plt.figure(figsize=(5, 5))

    for i, label in enumerate(label_names):
        key = label.lower()
        pixels_metabolite = pixels_dict.get(key)

        # Spectrum handling phantom
        V_MTRasym_reshaped_pc = V_MTRasym_reshaped*100
        array_MTR = V_MTRasym_reshaped_pc[pixels_metabolite[0]:pixels_metabolite[1],pixels_metabolite[2]:pixels_metabolite[3],slice_of_interest,1:] # 1: to remove the M0 scan
        flattened_vectors_MTR = array_MTR.reshape(-1, array_MTR.shape[-1]) 
        MTR_spectrum = flattened_vectors_MTR.mean(axis=0)

        plt.plot(w, MTR_spectrum, marker='o', markersize=2, label=label_names[i], color=colors[i])
        plt.axvline(x=3, color='grey', linestyle='--', linewidth=0.8, alpha=0.7)
        plt.xlim([0, 4])
        plt.ylim([-0.05,18])
        plt.xlabel('Frequency offset Δω [ppm]')
        plt.ylabel('MTRasym [%]')
        plt.gca().invert_xaxis()
        #plt.title("MTRasym-spectra for mixed metabolites")
        plt.grid(True, which='both', linestyle='--', linewidth=0.3, color='lightgrey', alpha=0.7)
        # Make axes box square in screen units
        xrange = 4         
        yrange = 18.05
        aspect_ratio = xrange / yrange
        plt.gca().set_aspect(aspect_ratio, adjustable='box')

    plt.legend(loc='upper right')
    plot_name = main_path + str("_MTR_spectra")
    my_path = r"c:\asb\ntnu\plotting\auto_save_png\mixed"
    save_path = os.path.join(my_path, plot_name + ".png")
    plt.savefig(save_path, format='png', bbox_inches='tight')
    #plt.show()

    # GluCEST effect for each [Glu]
    metabolites = np.arange(len(label_names))
    m_avg = np.array(m_avg)
    m_sem = np.array(m_sem)

    combined = np.concatenate((m_avg, m_sem))
    plot_name = main_path + str("_MTRasym")
    my_path = r"c:\asb\ntnu\plotting\auto_save_png\mixed"
    save_path = os.path.join(my_path, plot_name + ".txt")
    np.savetxt(save_path, combined, fmt='%s')

    # Plot data with error bars
    print('--- Plotting GluCEST effect ---')
    plt.errorbar(metabolites, m_avg, yerr=m_sem, fmt='o', label="Average ± SEM", capsize=6)
    plt.xlabel("Metabolites")
    plt.ylabel("MTRasym contrast [%]")
    #plt.title("GluCEST effect for metabolites")
    plt.xticks(metabolites, label_names)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.3, color='lightgrey', alpha=0.7)
    #plt.show()
    

if __name__ == "__main__":
    globals()["EVAL_GluCEST"] = EVAL_GluCEST 
    EVAL_GluCEST(
        data_path=r'C:\asb\ntnu\MRIscans\250410\dicoms\mixed_E2', 
        seq_path=r'C:\asb\ntnu\MRIscans\250410\seq_files\seq_file_mixed_E2.seq',
    )