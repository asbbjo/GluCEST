#Author: Asbjørn Bjørkkjær, NTNU
#Use scipy 1.15 or newer
#Based on the theory from Johannes Windschuh et al.(2015) and the work from Simon Köppel @ FAU


#%% Importing Packages

import numpy as np
from matplotlib import pyplot as plt
import os
import nibabel as nib
from natsort import natsorted
import pydicom
import pypulseq as pp

from csaps import csaps

from scipy import interpolate


# helper function to evaluate piecewise polinomial
def ppval(p, x):
     if callable(p):
         return p(x)
     else:
         n = len(p) - 1
         result = np.zeros_like(x)
         for i in range(n, -1, -1):
             result = result * x + p[i]
         return result

#%% Load dcm
data_dir=r'C:\asb\ntnu\v25\CEST_dicoms\phantomB1corr' 
seq_dir=r'C:\asb\ntnu\v25\CEST_dicoms\phantomB1corr\sequence_files'

#File Names of your nifits
B1_folders=["E11","E12","E13"]
B1_sequences=["E11.seq","E12.seq","E13.seq"]
B1_values=[2.0,2.5,3.0] # B1 values of your dicoms, the one in the middle is the one to which all values are corrected to

def load_dicoms(CEST_acq, n_meas):
    # check data path
    dcmpath = CEST_acq
    os.chdir(dcmpath)
    # read data from dicom directory
    collection = [pydicom.dcmread(os.path.join(dcmpath, filename)) for filename in sorted(os.listdir(dcmpath))]
    # extract the volume data
    V = np.stack([dcm.pixel_array for dcm in collection])
    V = np.transpose(V, (1, 2, 0))
    sz = V.shape
    V = np.reshape(V, [sz[0], sz[1], n_meas, sz[2] // n_meas]).transpose(0, 1, 3, 2) # V.shape=[x,y,z,freq_offset]
    return V

print('')
print('--- Reading the sequence protocol. This may take a while ---')
CEST_data=np.empty((3,128,128,1,51))
for i in range(len(B1_folders)):
    print(f"Managing B1 acquisition {i+1}")
    # find the folder for each acquisition
    CEST_acq = os.path.join(data_dir, B1_folders[i])
    CEST_seq = os.path.join(seq_dir, B1_sequences[i])

    # read the sequence file
    seq = pp.Sequence()
    seq.read(CEST_seq)
    offsets = seq.get_definition("offsets_ppm")
    m0_offset = seq.get_definition("M0_offset")
    n_meas = len(offsets) # these parameters will be set by the B1 folder. Assume that all acq are equal

    # extract volume data
    volume_data = load_dicoms(CEST_acq, n_meas)
    CEST_data[i] = volume_data

#insert a 0muT B1_value "measurement" for stability puorposeses, also allows for spline fitting since it requires at least 4 data points
B1_values=np.insert(B1_values,0,0)

CEST_data=np.insert(CEST_data,0,np.ones_like(CEST_data[1]),axis=0) # "Data" for the 0muT B1 measurement
CEST_data=np.transpose(CEST_data,(1,2,3,4,0)) # CEST_data.shape=[128,128,1,51,4]

#%% Load B0 + B1 Data
WASABI_CORR=False # change between WASSR and WASABI B0 correction

WASABI_B0_image=nib.load(str(Path+"B0map.nii")) # Load B0Map
WASABI_B1_image=nib.load(str(Path+"B1map.nii")) # Load B1 Map

WASABI_B1_real=WASABI_B1_image.get_fdata()
WASABI_B1_real=WASABI_B1_real[:,::-1,:] #changes (reverse) the orientation to correct in MATLAB generated images
WASABI_B1=np.reshape(WASABI_B1_real,-1) #flattens the matrix

WASABI_B0_real=WASABI_B0_image.get_fdata()[:,::-1,:]
WASABI_B0=np.reshape(WASABI_B0_real,-1)


#%% make sure the orientation of the data and the B0/B1 Map aligns
plt.close("all")
fig=plt.figure(13)
ax=fig.add_subplot(121)
ax.imshow(WASABI_B1_real[:,:,1])

ax2=fig.add_subplot(122)
ax2.imshow(CEST_data[:,:,1,40,2])


#%% Sequence Stuff
# only one seq file for the offsets
seq=pp.Sequence()
print("Loading in Sequence -- This may take some time")
seq.read('Path to your seq file') #change this
m0_offset = seq.get_definition("M0_offset")
offsets = seq.get_definition("offsets_ppm")

offsets=np.asarray(natsorted(offsets))
n_meas = len(offsets)
M0_idx = np.where(abs(offsets) >= abs(m0_offset[0]))[0]  #M0_idx is longer than 1 since we measure our M0 offset multiple times as dummy scans, to train our ADC, should also work with just a list with one entry
offsets = np.delete(offsets, M0_idx)
# n_meas=85
#%%
mask = np.squeeze(B1_Data[:, :, :, 0,2]) > 100
mask_idx = np.where(mask.ravel())[0]
def make_B0_correction(D4Data,mask_idx):
     # Vectorization
     global offsets
     D4Data=D4Data[:,:,:,:]
     V_m_z = D4Data.reshape(-1, n_meas).T
     m_z = V_m_z[:, mask_idx]

     M0 = (m_z[M0_idx[-1], :])

     m_z = np.delete(m_z, M0_idx, axis=0)
     Z = m_z / M0  # Normalization


     Z_corr = np.zeros_like(Z)
     w = offsets
     dB0_stack = np.zeros(Z.shape[1])
     for ii in range(Z.shape[1]):
         if ii%1000==0:
             print("Calculating Spline: ", np.round(ii/Z.shape[1]*100,1),"%")
         if np.all(np.isfinite(Z[:, ii])):
             pp = csaps(w, Z[:, ii],smooth=0.999)
             w_fine = np.arange(-1, 1.005, 0.005)
             z_fine = ppval(pp, w_fine)

             min_idx = np.argmin(z_fine)
             dB0_stack[ii] = w_fine[min_idx]

             if WASABI_CORR==False:
                 Z_corr[:, ii] = ppval(pp, w + dB0_stack[ii])
             else:
                 # Z_corr[:, ii] = ppval(pp, (w + WASABI_B0[ii]))*WASABI_B1[ii] # not sure about the B1 correction here
                 Z_corr[:, ii] = ppval(pp, (w + WASABI_B0[ii]))
     # Vectorization Backwards
     if Z.shape[1] > 1:
         V_Z_corr = np.zeros((V_m_z.shape[0], V_m_z.shape[1]), dtype=float)
         V_Z_corr[len(m0_offset):, mask_idx] = Z_corr
         V_Z_corr_reshaped = V_Z_corr.reshape(D4Data.shape[3], D4Data.shape[0], D4Data.shape[1], D4Data.shape[2]).transpose(1, 2, 3, 0)

         return V_Z_corr_reshaped
     else:
         raise Exception("Something went wrong in the backwards reshaping process")


B1_Data_B0=[]
for i in range(len(B1_values)):
     B1_Data_B0.append(make_B0_correction(B1_Data[:,:,:,:,i], mask_idx))
#%%
B1_Data_B0=np.asarray(B1_Data_B0)
B1_Data_B0=np.transpose(B1_Data_B0,(1,2,3,4,0))

plt.close(13)
fig=plt.figure(13)
ax=fig.add_subplot(231)
ax.imshow(WASABI_B1_real[:,:,8]*0.6)

ax2=fig.add_subplot(232)
ax2.imshow(B1_Data[:,:,8,40,2])

ax4=fig.add_subplot(234)
ax4.imshow(B1_Data_B0[:,:,8,40,1])
ax4.set_title("Lower B1 Level")

ax5=fig.add_subplot(235)
ax5.imshow(B1_Data_B0[:,:,8,40,2])
ax5.set_title("Target B1 Level")

ax6=fig.add_subplot(236)
ax6.imshow(B1_Data_B0[:,:,8,40,3])
ax6.set_title("Higher B1 Level")

#%% Lets do the B1 correction

Absolute_B1_Map=B1_values[2]*WASABI_B1_real

def interpolate_single_spec(data,B1):
     #calculates the Interpolation for all offsets from one voxel
     Interpolation_Result=np.empty(data.shape[:1], dtype=object)
     # print(np.shape(Interpolation_Result))
     for i in range(len(Interpolation_Result)):
         # Interpolation_Result[i]=interpolate.make_interp_spline(B1,data[i,:],k=1)
         Interpolation_Result[i]=interpolate.make_splrep(B1,data[i,:],k=1,s=0.0) # k=1 => linear interpolation | k=3 => cubic spline interpolatiion, s=0 => no smoothing
         # print("Interpolated voxel:", [i])
     return Interpolation_Result


def calc_B1_Spec_correction(B1,Interpolation):
     # calculates the B1 correction with the Interpolation and the actual B1 level from the absolute B1Map
     Spec_corr=[]
     B1=B1_values[2]+(B1_values[2]-B1) # calculates the shift necessary to get the real B1 level to the intended one, stored in the first position of the B1_values array (remeber the inserted 0 muT "measurement")
     for i in range(len(Interpolation)):
         Spec_corr.append(Interpolation[i](B1))
     # print(Spec_corr)
     return np.asarray(Spec_corr)


def calc_B1_all_correction(Data,absB1Map,B1_values):
     #calculates the B1 correction for the whole 3D array given the input of the B0_corrected Data, absolute B1Map and the B1 values
     absB1Map=np.nan_to_num(absB1Map)
     Data_B0_B1_corr=np.zeros_like(Data[:,:,:,:,1])
     shape=np.shape(Data[:,:,:,:,1])
     # DatamalB1=relB1Map[:,:,:,np.newaxis,:]*Data[:,:,:,:,:]
     for i in range((shape[0])):
         for j in range((shape[1])):
             for k in range((shape[2])):
                 print("Voxel: ",[i,j,k])
                 Interpolation=interpolate_single_spec(Data[i,j,k,:,:],B1_values)
                 # print(Interpolation)
                 Data_B0_B1_corr[i,j,k,:]=calc_B1_Spec_correction(absB1Map[i,j,k],Interpolation[:])
     
     return Data_B0_B1_corr

B1_corrected_all=calc_B1_all_correction(B1_Data_B0,
Absolute_B1_Map[:,:,:,0], B1_values) #remove the fourth dimension of your absolute B1Map if your array is 3D
B1_corrected_all=np.nan_to_num(B1_corrected_all)
#%%
plt.close(11)
fig=plt.figure(11)
ax=fig.add_subplot()
ax.imshow(B1_corrected_all[:,:,2,30],vmin=0.5,vmax=1.1)
ax.set_title("B1 corrected Image")

#%%
affine=np.eye(4)

nib_file=nib.Nifti1Image(B1_corrected_all,affine)
nib.save(nib_file,Path+"B1_0p6muT_corrected_0p6_lin.nii")