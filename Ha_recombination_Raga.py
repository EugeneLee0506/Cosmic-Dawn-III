# Calculates the cell-wise H-alpha luminosity density due to the recombination, based on the Raga's recombinational coefficient. 

import numpy as np
import astropy
import math
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

H0 = 0.677699966430664E+02
G = 6.67408E-8
c = 2.99792458e10
omega_b=0.450000017881393E-01
omega_m=0.307
omega_l=0.693
hv_alpha = 3.028113999E-12  
m_p = 1.6726E-24 
k_b = 1.3807E-16
e = 2.71828182845904
H_alpha = 6.5646e-5
Mpc = 3.086e24
num_files = 64
output_num = '000023'
redshift_num = 22  # output_num - 1 without zero

base_path = '/scratch/08389/tg876886/2048_data/output_' + output_num + '/'
unit_d_path = '/home1/10000/elee0506/unit_d.npy'
redshift_path = '/home1/10000/elee0506/redshift.npy'
scale_path = '/home1/10000/elee0506/scale_factor.npy'

unit_d = np.load(unit_d_path).astype(np.float32)
redshift = np.load(redshift_path).astype(np.float32)
scale = np.load(scale_path).astype(np.float32)

for i in range(num_files):
    # Construct the file names
    rho_file = os.path.join(base_path, f'rho_{i:02}.npy')
    xion_file = os.path.join(base_path, f'xion_{i:02}.npy')
    temp_file = os.path.join(base_path, f'temp_{i:02}.npy')
    
    # Load temp file
    temp = np.load(temp_file).astype(np.float32)  # Load temp file (32 GB)

    temp1 = temp ** 1.5                           # Compute T**1.5 (in-place)
    temp **= 0.568                                # Compute T**0.568 (in-place)
    temp += 3.85E-5 * temp1                       # Combine in-place: 3.85E-5 * T**1.5 + T**0.568
    del temp1
    Temp_rec = 1 / temp 
    del temp
    
    # Load rho and xion files
    rho = np.load(rho_file).astype(np.float32)  # Load rho (32 GB)
    xion = np.load(xion_file).astype(np.float32)  # Load xion (32 GB)
    
    # Avoid large intermediate array (rho * xion)**2
    rho_xion = rho * xion     # Direct element-wise multiplication in-place
    del rho
    del xion
    rho_xion **= 2             # Square in-place to avoid large intermediate array
    
    Const = (unit_d[redshift_num] / m_p)**2 * 4.85E-23 * 0.76**2
    rec = Const * Temp_rec * rho_xion
    
    output_file = f'/home1/10000/elee0506/scratch/output_{output_num}/rec/raga_rec_{i:02}.npy'
    np.save(output_file, rec, allow_pickle=True, fix_imports=True)
    del rec   