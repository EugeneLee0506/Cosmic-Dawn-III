# Calculates the cell-wise H-alpha luminosity density due to the recombination, based on the Draine's effective recombinational coefficient. 

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

#fianl rec equation = hv_alpha * 1.17E-13*(T/10E3)**(-0.942-0.031*np.log(T/10E3))*((rho*unit_d[117]/m_p)*x_ion)**2 *0.76**2

for i in range(num_files):
    # Construct the file names
    rho_file = os.path.join(base_path, f'rho_{i:02}.npy')
    xion_file = os.path.join(base_path, f'xion_{i:02}.npy')
    temp_file = os.path.join(base_path, f'temp_{i:02}.npy')
    

    # Load the rho, x_ion, and temp files
    temp = np.load(temp_file)/10E3
    Temp_rec = (temp)**(-0.942-0.031*np.log(temp))
    del(temp)

    rho = np.load(rho_file)
    xion = np.load(xion_file)
    rho_xion = (rho*xion)**2
    del(rho)
    del(xion)
  
    Const = (unit_d[redshift_num]/m_p)**2 * 1.17E-13 * 0.76**2
    rec = hv_alpha * Const * Temp_rec * rho_xion 

    output_file = '/home1/10000/elee0506/scratch/output_' + output_num + '/rec/' + f'rec_{i:02}.npy'
    np.save(output_file, rec, allow_pickle=True, fix_imports=True)
    del(rec)