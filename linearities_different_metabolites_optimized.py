import os
from pathlib import Path
from scipy.interpolate import interp1d, UnivariateSpline
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
    "font.size": 15,  # IEEE column text is usually around 8-9 pt
    "axes.labelsize": 9,
    "axes.titlesize": 12,
    "legend.fontsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "lines.linewidth": 1,
    "lines.markersize": 4,
    "figure.dpi": 200,
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

    # 1. Find unique positive values (ignore 0)
    w_pos_unique = np.unique(np.abs(w[w != 0]))

    # 2. Create symmetric offsets: negative and positive pairs
    w_symm = np.sort(np.concatenate([-w_pos_unique, [0], w_pos_unique]))

    # 3. Interpolate Z_corr onto this new symmetric grid
    from scipy.interpolate import interp1d

    Z_corr_symm = np.zeros((len(w_symm), Z_corr.shape[1]))

    for i in range(Z_corr.shape[1]):
        interp_func = interp1d(w, Z_corr[:, i], kind='linear', fill_value="extrapolate")
        Z_corr_symm[:, i] = interp_func(w_symm)

    # calculation of MTRasym spectrum
    Z_ref = Z_corr_symm[::-1, :]
    MTRasym = Z_ref - Z_corr_symm

    # Interpolating back to old w
    interp_back_func = interp1d(w_symm, Z_corr_symm, axis=0, kind='linear', fill_value='extrapolate')
    Z_corr_symm_resampled = interp_back_func(w)  # shape [36, 16382]

    interp_back_func_mtr = interp1d(w_symm, MTRasym, axis=0, kind='linear', fill_value='extrapolate')
    MTRasym_resampled = interp_back_func_mtr(w)  # shape [36, 16382]


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

    slice_of_interest = 0 # pick slice for evaluation (0 if only one slice)
    desired_offset = 3 # pick offset for evaluation (3 for GluCEST at 3 ppm)
    offset_of_interest = np.where(offsets == desired_offset)[0]  
    w_offset_of_interest = offsets[offset_of_interest]


    # Choose pixels for ROI
    pixels_dict = {            # 250409 & 250410
        'glu': [44,49,54,59],    
        'gln': [42,47,73,78],    
        'gaba': [59,64,86,91],    
        'naa': [76,81,77,82],     
        'cr': [78,83,58,63],      
        'taurine': [63,68,47,52], 
    }

    # Choose metabolites
    label_names = ['Glu', 'Gln', 'GABA', 'NAA', 'Cr', 'Taurine']

    V_MTRasym_reshaped_pc = V_MTRasym_reshaped*100
    mm_avg = []
    mm_sem = []
    print('--- Statistical measurements ---')
    for i, label in enumerate(label_names):
        key = label.lower()
        pixels_metabolite = pixels_dict.get(key)        
        mm = V_MTRasym_reshaped_pc[pixels_metabolite[0]:pixels_metabolite[1],pixels_metabolite[2]:pixels_metabolite[3], slice_of_interest, offset_of_interest]
        avg, sem = np.mean(mm.reshape(-1)), sc.stats.sem(mm.reshape(-1))
        mm_avg.append(avg)
        mm_sem.append(sem)

    return np.array(mm_avg), np.array(mm_sem)

if __name__ == "__main__":

    metabolites = ['Glu', 'Gln', 'GABA', 'NAA', 'Cr', 'Taurine']

    # 250312
    dcm_names = np.array(['9','14','18','7','22','26'])
    label_names = ['10e-6s', '1s', '2s', '3s', '5s', '10s']
    title = str("Linear trends with recovery times")

    MTR_contrasts_avg = np.empty((6,6),dtype=float)
    MTR_contrasts_sem = np.empty((6,6),dtype=float)

    for i in range(len(dcm_names)):
        print(f'Loop: {i+1}')
        data_path = str(r'C:\asb\ntnu\MRIscans\250410\dicoms\E') + dcm_names[i]
        seq_path = str(r'C:\asb\ntnu\MRIscans\250410\seq_files\seq_file_E') + dcm_names[i] + str('.seq')
        mm_avg, mm_sem = EVAL_GluCEST(data_path, seq_path)

        for j in range(len(mm_avg)):
            MTR_contrasts_avg[j][i] = mm_avg[j]
            MTR_contrasts_sem[j][i] = mm_sem[j]


    mm = np.array([0,1,2,3,5,10])
    plt.figure(figsize=(12, 4))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(label_names)))

    for i in range(len(MTR_contrasts_avg[0])):
        # Fit a linear trend line
        slope, intercept = np.polyfit(mm, MTR_contrasts_avg[i], 1)  # Linear fit (degree=1)
        trend_line = slope * mm + intercept  # Calculate trend line values

        # MSE
        mse = np.mean((MTR_contrasts_avg[i] - trend_line) ** 2)
        label_uT = str(metabolites[i]) + ":   MSE = " + str(round(mse,4))
        
        # Plot data with error bars
        plt.errorbar(mm, MTR_contrasts_avg[i], yerr=MTR_contrasts_sem[i], fmt='o', label=label_uT, capsize=6, color=colors[i])
        
        # Plot the trend line
        plt.plot(mm, trend_line,'--', color=colors[i])

        # Labels and legend
        plt.xlabel("Recovery times")
        plt.ylabel("MTRasym contrast [%]")
        plt.title(title)
        plt.xticks(mm, label_names)
        plt.grid(True, which='both', linestyle='--', linewidth=0.3, color='lightgrey', alpha=0.7)
        plt.legend()

    plt.show()