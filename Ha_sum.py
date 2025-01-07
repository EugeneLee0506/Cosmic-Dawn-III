# Sums luminoisty densities from 64 npy files (1 npy file: luminosity density of 2048^3 cells) produced from either Ha_recombination or Ha_collision codes.

import numpy as np
import astropy
import math
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

#output_num is 3 digit numbers for the corresponding output (max: 118), and type can be either 'rec' or 'col'

output_num = '023'
type = 'rec'

rec_path = '/home1/10000/elee0506/scratch/output_000' + output_num + '/rec/'
col_path = '/home1/10000/elee0506/scratch/output_000' + output_num + '/col/'

if type == 'col':
    file_path = col_path
if type == 'rec':
    file_path = rec_path

Lsum = 0
results = []
for N in range(64):
    file = np.load(os.path.join(file_path, type + f'_{N:02}.npy'))

    t = np.add.reduce(file).sum() 
    Lsum += t
        
    results.append([N, Lsum])
    del file

    print(N, Lsum, t)
    
np.save('/home1/10000/elee0506/scratch/output_000'+output_num+'/Lsum/alpha_'+ type + '_full.npy', results)