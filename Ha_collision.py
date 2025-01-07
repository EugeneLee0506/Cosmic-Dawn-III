# Calculates the cell-wise H-alpha luminosity density due to the collision, based on the Raga's coefficient. 

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
H_alpha = 6.5646e-5
Mpc = 3.086e24
num_files = 64
output_num = '000042'
redshift_num = 41 # output_num - 1 without zero

base_path = '/scratch/08389/tg876886/2048_data/output_' + output_num + '/'
unit_d_path = '/home1/10000/elee0506/unit_d.npy'
redshift_path = '/home1/10000/elee0506/redshift.npy'
scale_path = '/home1/10000/elee0506/scale_factor.npy'

unit_d = np.load(unit_d_path).astype(np.float32)
redshift = np.load(redshift_path).astype(np.float32)
scale = np.load(scale_path).astype(np.float32)

#col = (3.57E-17/T**0.5)*e**(-140360/T)*(1+7.8/(1+5E5/T))*(rho*unit_d[117]/m_p)**2*xion*(1-xion) *0.76**2

for i in [64]:
    # Construct the file names
    rho_file = os.path.join(base_path, f'rho_{i:02}.npy')
    xion_file = os.path.join(base_path, f'xion_{i:02}.npy')
    temp_file = os.path.join(base_path, f'temp_{i:02}.npy')
    
    T = np.load(temp_file)
    del(temp_file)
    Temp_col_1 = (1/T**0.5)*(1+7.8/(1+5E5/T)) * np.exp(-140360/T, dtype=np.float32)
    del(T)

    rho = np.load(rho_file)
    del(rho_file)
    Col_rho = rho**2 
    del(rho)

    xion = np.load(xion_file)
    del(xion_file)
    Col_x_ion = xion * (1-xion)
    del xion

    const = 3.57E-17 * ((unit_d[redshift_num])/m_p)**2 * 0.76**2 
    Col = const * Temp_col_1  * Col_rho * Col_x_ion
    del Temp_col_1, Col_rho, Col_x_ion

    #Col = const * (3.57E-17/T**0.5)*e**(-140360/T)*(1+7.8/(1+5E5/T))*(rho*unit_d[117]/m_p)**2*x_ion*(1-x_ion) *0.76**2
    output_file = '/home1/10000/elee0506/scratch/output_' + output_num + '/col/' + f'col_{i:02}.npy'
    np.save(output_file, Col, allow_pickle=True, fix_imports=True)
    del(Col)