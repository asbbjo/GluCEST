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

def EVAL_GluCEST(data_path, seq_path, date):
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
    dB0_stack = np.zeros(Z.shape[1]) # could be changes for wasabi
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

    slice_of_interest = 0 # pick slice for evaluation (0 if only one slice)
    desired_offset = 3 # pick offset for evaluation (3 for GluCEST at 3 ppm)
    offset_of_interest = np.where(offsets == desired_offset)[0]  
    w_offset_of_interest = offsets[offset_of_interest]

    # Choose pixels for ROI
    if date == '250312':
        pixels_0mm = [66,71,80,85] # 250312
        pixels_2mm = [81,86,67,72] # 250312
        pixels_4mm = [76,81,47,52] # 250312
        pixels_6mm = [57,62,41,46] # 250312
        pixels_8mm = [43,48,54,59] # 250312
        pixels_10mm = [47,52,74,79] # 250312
    elif date == '250313':
        pixels_0mm = [66,71,80,85] # 250313
        pixels_2mm = [81,86,67,72] # 250313
        pixels_4mm = [76,81,47,52] # 250313
        pixels_6mm = [57,62,41,46] # 250313
        pixels_8mm = [43,48,54,59] # 250313
        pixels_10mm = [47,52,74,79] # 250313
    elif date == '250317':
        pixels_0mm = [62,67,81,86] # 250317
        pixels_2mm = [77,82,68,73] # 250317
        pixels_4mm = [72,77,49,54] # 250317
        pixels_6mm = [54,59,42,47] # 250317
        pixels_8mm = [40,45,56,61] # 250317
        pixels_10mm = [43,48,75,80] # 250317
    elif date == '250324':
        pixels_0mm = [40,45,56,61] # 250324
        pixels_2mm = [44,49,75,80] # 250324
        pixels_4mm = [62,67,81,86] # 250324
        pixels_6mm = [77,82,67,72] # 250324
        pixels_8mm = [72,77,48,53] # 250324
        pixels_10mm = [54,59,42,47] # 250324

    print('--- Statistical measurements ---')
    V_MTRasym_reshaped_pc = V_MTRasym_reshaped*100
    mm0 = V_MTRasym_reshaped_pc[pixels_0mm[0]:pixels_0mm[1],pixels_0mm[2]:pixels_0mm[3], slice_of_interest, offset_of_interest]
    mm0_avg, mm0_sem = np.mean(mm0.reshape(-1)), sc.stats.sem(mm0.reshape(-1))

    mm2 = V_MTRasym_reshaped_pc[pixels_2mm[0]:pixels_2mm[1],pixels_2mm[2]:pixels_2mm[3], slice_of_interest, offset_of_interest]
    mm2_avg, mm2_sem = np.mean(mm2.reshape(-1)), sc.stats.sem(mm2.reshape(-1))

    mm4 = V_MTRasym_reshaped_pc[pixels_4mm[0]:pixels_4mm[1],pixels_4mm[2]:pixels_4mm[3], slice_of_interest, offset_of_interest]
    mm4_avg, mm4_sem = np.mean(mm4.reshape(-1)), sc.stats.sem(mm4.reshape(-1))

    mm6 = V_MTRasym_reshaped_pc[pixels_6mm[0]:pixels_6mm[1],pixels_6mm[2]:pixels_6mm[3], slice_of_interest, offset_of_interest]
    mm6_avg, mm6_sem = np.mean(mm6.reshape(-1)), sc.stats.sem(mm6.reshape(-1))

    mm8 = V_MTRasym_reshaped_pc[pixels_8mm[0]:pixels_8mm[1],pixels_8mm[2]:pixels_8mm[3], slice_of_interest, offset_of_interest]
    mm8_avg, mm8_sem = np.mean(mm8.reshape(-1)), sc.stats.sem(mm8.reshape(-1))

    mm10 = V_MTRasym_reshaped_pc[pixels_10mm[0]:pixels_10mm[1],pixels_10mm[2]:pixels_10mm[3], slice_of_interest, offset_of_interest]
    mm10_avg, mm10_sem = np.mean(mm10.reshape(-1)), sc.stats.sem(mm10.reshape(-1))

    mm = np.array([0,2,4,6,8,10])
    mm_avg = np.array([mm0_avg, mm2_avg, mm4_avg, mm6_avg, mm8_avg, mm10_avg])
    mm_sem = np.array([mm0_sem, mm2_sem, mm4_sem, mm6_sem, mm8_sem, mm10_sem])


    return mm, mm_avg, mm_sem
    

if __name__ == "__main__":

    # 250312
    '''dcm_names = np.array(['23','28','29','30','32','33','34'])
    label_names = ['10e-5s', '1s', '2s', '3s', '4s', '6s', '10s']
    title = str("Linear trends with recovery times")'''

    # 250313
    dcm_names = np.array(['12','13','14','15','16'])
    label_names = ['1uT', '2uT', '3uT', '4uT', '5uT']
    title = str("Linear trends with pulse powers")

    # 250317
    '''dcm_names = np.array(['22','24','14','23','25'])
    label_names = ['15ms', '30ms', '50ms', '100ms', '300ms']
    title = str("Linear trends with pulse lengths")'''

    # 250324
    '''dcm_names = np.array(['10','11','12','13','14','15'])
    label_names = ['10e-5s', '1s', '2s', '3s', '5s', '10s']
    title = str("Linear trends with recovery times")

    #dcm_names = np.array(['16','17','18','19','20']) 
    #label_names = ['1uT', '2uT', '3uT', '4uT', '5uT'] 
    #title = str("Linear trends with pulse powers")

    #dcm_names = np.array(['21','22','23','24','25'])
    #label_names = ['15ms', '30ms', '50ms', '100ms', '300ms']
    #title = str("Linear trends with pulse lengths")

    #dcm_names = np.array(['13','18','23','35'])
    #label_names = ['baseline 1', 'baseline 2', 'baseline 3', 'baseline 4']
    #title = str("Linear trends with baseline measurements")'''

    plt.figure(figsize=(10, 4))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(dcm_names)))

    input('Correct path for you acquisitions?\n')
    for i in range(len(dcm_names)):
        print(f'Loop: {i+1}')
        data_path = str(r'C:\asb\ntnu\MRIscans\250313\dicoms\E') + dcm_names[i]
        seq_path = str(r'C:\asb\ntnu\MRIscans\250313\seq_files\seq_file_E') + dcm_names[i] + str('.seq')
        date = '250313'
        mm, mm_avg, mm_sem = EVAL_GluCEST(data_path, seq_path, date)

        # Fit a linear trend line
        slope, intercept = np.polyfit(mm, mm_avg, 1)  # Linear fit (degree=1)
        trend_line = slope * mm + intercept  # Calculate trend line values

        # MSE
        mse = np.mean((mm_avg - trend_line) ** 2)
        label_uT = str(label_names[i]) + ":   MSE = " + str(round(mse,5))
        
        # Plot data with error bars
        plt.errorbar(mm, mm_avg, yerr=mm_sem, fmt='o', label=label_uT, capsize=6, color=colors[i])
        
        # Plot the trend line
        plt.plot(mm, trend_line,'--', color=colors[i], label=f"Trend {label_names[i]}: y={slope:.2f}x + {intercept:.2f}")

        # Labels and legend
        plt.xlabel("Concentration of Glu [mM]")
        plt.ylabel("MTRasym contrast [%]")
        plt.title(title)
        '''plt.grid(True)'''
        plt.legend()
    plt.show()