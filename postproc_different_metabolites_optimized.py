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
from scipy.interpolate import interp1d

# Store separate dicoms of each frequency offset in one folder
# Create a sequence file for the acquisition
# Run with pypulseq==1.4.2 and pydicom==3.0.1
# Make sure to have Grassroots DICOM (pip install gdcm) and pylibjpeg (pip install pylibjpeg pylibjpeg-libjpeg)

plt.rcParams.update({
    # Parameters for plotting
    "font.size": 10,
    "font.family": 'serif',
    "axes.labelsize": 15,
    "axes.titlesize": 1,
    "legend.fontsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "lines.linewidth": 1,
    "lines.markersize": 4,
    "figure.dpi": 250,
})

def ppval(p, x):
    # Helper function to evaluate piecewise polinomial
    if callable(p):
        return p(x)
    else:
        n = len(p) - 1
        result = np.zeros_like(x)
        for i in range(n, -1, -1):
            result = result * x + p[i]
        return result

def EVAL_GluCEST(data_path, seq_path, date):
    # Pipeline for glutamate-weighted CEST
    import pypulseq as pp # the import over is not found
    seq = pp.Sequence()
    print('--- Reading the sequence protocol ---')
    seq.read(seq_path)
    offsets = seq.get_definition("offsets_ppm")
    m0_offset = seq.get_definition("M0_offset")
    n_meas = len(offsets)

    dcmpath = data_path
    os.chdir(dcmpath)

    # Read data from dicom directory
    collection = [pydicom.dcmread(os.path.join(dcmpath, filename)) for filename in sorted(os.listdir(dcmpath))]
    # Extract the volume data
    V = np.stack([dcm.pixel_array for dcm in collection])
    V = np.transpose(V[:,:,:,-1], (1, 2, 0)) # Erase the last dimention due to jpeg format ([f,x,y,rgb] => [f,x,y])
    sz = V.shape
    V = np.reshape(V, [sz[0], sz[1], n_meas, sz[2] // n_meas]).transpose(0, 1, 3, 2)

    # Vectorization
    threshold = 100 # Change wrt. the noise in the image
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

    # Create symmetric offsets: negative and positive pairs
    w_pos_unique = np.unique(np.abs(w[w != 0]))
    w_symm = np.sort(np.concatenate([-w_pos_unique, [0], w_pos_unique]))
    Z_corr_symm = np.zeros((len(w_symm), Z_corr.shape[1]))

    for i in range(Z_corr.shape[1]):
        interp_func = interp1d(w, Z_corr[:, i], kind='linear', fill_value="extrapolate")
        Z_corr_symm[:, i] = interp_func(w_symm)

    # Calculation of MTRasym spectrum
    Z_ref = Z_corr_symm[::-1, :]
    MTRasym = Z_ref - Z_corr_symm

    # Interpolating back to old w
    interp_back_func = interp1d(w_symm, Z_corr_symm, axis=0, kind='linear', fill_value='extrapolate')
    Z_corr_symm_resampled = interp_back_func(w)  

    interp_back_func_mtr = interp1d(w_symm, MTRasym, axis=0, kind='linear', fill_value='extrapolate')
    MTRasym_resampled = interp_back_func_mtr(w) 


    # Vectorization Backwards
    if Z.shape[1] > 1:
        V_MTRasym = np.zeros((V_m_z.shape[0], V_m_z.shape[1]), dtype=float)
        V_MTRasym[1:, mask_idx] = MTRasym_resampled        
        V_MTRasym_reshaped = V_MTRasym.reshape(
            V.shape[3], V.shape[0], V.shape[1], V.shape[2]
        ).transpose(1, 2, 3, 0)
    
        V_Z_corr = np.zeros((V_m_z.shape[0], V_m_z.shape[1]), dtype=float)
        V_Z_corr[1:, mask_idx] = Z_corr_symm_resampled
        V_Z_corr_reshaped = V_Z_corr.reshape(
            V.shape[3], V.shape[0], V.shape[1], V.shape[2]
        ).transpose(1, 2, 3, 0)

    # Choose pixels for ROI
    pixels_dict = {  
        'glu 10 mm': [44,49,54,59],    
        'gln 2 mm': [42,47,73,78],    
        'gaba 2 mm': [59,64,86,91],    
        'naa 10 mm': [76,81,77,82],     
        'cr 6 mm': [78,83,58,63],      
        'taurine 2 mm': [63,68,47,52], 
    }

    print('--- Plotting GluCEST images ---')
    slice_of_interest = 0 # Pick slice for evaluation (0 if only one slice)
    desired_offset = 3 # Pick offset for evaluation (3 for GluCEST at 3 ppm)
    offset_of_interest = np.where(offsets == desired_offset)[0]  
    w_offset_of_interest = offsets[offset_of_interest]

    fig, ax = plt.subplots(figsize=(5, 5)) 
    vmin, vmax = 0.5, 1 
    im = ax.imshow(V_Z_corr_reshaped[:, :, slice_of_interest, offset_of_interest],vmin=vmin, vmax=vmax, cmap='rainbow')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = plt.colorbar(im, cax=cax, format="%.2f")
    cb.set_ticks(np.linspace(vmin, vmax, 5)) 
    ax.set_title("Z(Δω) = %.2f ppm" % w_offset_of_interest)
    plt.show()

    pixels_glu = pixels_dict.get('glu 10 mm') # Evaluate for glu
    array_MTR = V_MTRasym_reshaped[pixels_glu[0]:pixels_glu[1],pixels_glu[2]:pixels_glu[3],slice_of_interest,1:] # 1: to remove the M0 scan
    flattened_vectors_MTR_glu = array_MTR.reshape(-1, array_MTR.shape[-1]) 

    MTR_max = np.max(flattened_vectors_MTR_glu)
    fig, ax = plt.subplots(figsize=(5, 5)) 
    vmin, vmax = 0, MTR_max # set GluCEST contrast range
    im = ax.imshow(V_MTRasym_reshaped[:,:,slice_of_interest,offset_of_interest], vmin=vmin, vmax=vmax, cmap='OrRd')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = plt.colorbar(im, cax=cax, format="%.2f")
    cb.set_ticks(np.linspace(vmin, vmax, 5)) 
    ax.set_title("MTRasym(Δω) = %.2f ppm" % w_offset_of_interest)
    plt.show()

    # Choose metabolites
    label_names = ['Glu 10 mM', 'Gln 2 mM', 'GABA 2 mM', 'NAA 10 mM', 'Cr 6 mM', 'Taurine 2 mM']
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
        plt.title("Z-spectra for different metabolites")
        xrange = 10       
        yrange = 1.1 - 0.12
        aspect_ratio = xrange / yrange
        plt.gca().set_aspect(aspect_ratio, adjustable='box')

    plt.legend(loc='lower right')
    plt.show()

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
        plt.ylim([-0.1,12])
        plt.xlabel('Δω [ppm]')
        plt.ylabel('MTRasym [%]')
        plt.gca().invert_xaxis()
        plt.title("MTRasym-spectra for different metabolites")
        plt.grid(True, which='both', linestyle='--', linewidth=0.3, color='lightgrey', alpha=0.7)
        xrange = 4         
        yrange = 12.1
        aspect_ratio = xrange / yrange
        plt.gca().set_aspect(aspect_ratio, adjustable='box')

    plt.legend(loc='upper right')
    plt.show()

    # GluCEST effect for each [conc]
    metabolites = np.arange(len(label_names))
    m_avg = np.array(m_avg)
    m_sem = np.array(m_sem)

    # Plot data with error bars
    plt.figure(figsize=(5, 5))
    print('--- Plotting GluCEST effect ---')
    plt.errorbar(metabolites, m_avg, yerr=m_sem, fmt='o', label="Average ± SEM", capsize=6)
    plt.xlabel("Metabolites")
    plt.ylabel("MTRasym contrast [%]")
    plt.title("GluCEST effect for metabolites")
    plt.xticks(metabolites, label_names)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.3, color='lightgrey', alpha=0.7)
    plt.show()
    

if __name__ == "__main__":
    globals()["EVAL_GluCEST"] = EVAL_GluCEST 
    EVAL_GluCEST(
        data_path=r'your_dicom_path', 
        seq_path=r'your_seq_file_path',
    )