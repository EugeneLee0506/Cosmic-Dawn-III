# Applies dark matter overdensity (50/100/200) to 64 luminoisty density data (either rec_00 - 63 / col_00 - 63), and sums the luminosity densities from the post-filtered cells.

import numpy as np
import astropy
import math
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

#output_num is 3 digit numbers. filter_density is the value of overdensity filter, which can be 50/100/200. type is either 'rec' or 'col'.

output_num = '014'
filter_density = '50'
type = 'rec'

rec_path = '/home1/10000/elee0506/scratch/output_000' + output_num + '/rec/'
col_path = '/home1/10000/elee0506/scratch/output_000' + output_num + '/col/'
filter_path = '/scratch/08389/tg876886/filter_DM/output_000' +output_num + '/'

if type == 'col':
    file_path = col_path
if type == 'rec':
    file_path = rec_path


Lsum = 0
results = []
for N in range(64):

    cubic_file = np.load(os.path.join(file_path, 'raga_' + type +  f'_{N:02}.npy'))
    filter = np.load(os.path.join(filter_path, 'DM_' + filter_density + f'_{N:02}.npy'))  # Load filter

    # Ensure filter is boolean
    if filter.dtype != bool:
        filter = filter.astype(bool)

    t = (cubic_file[filter]).sum()  # Apply the boolean filter to the rec_file
    Lsum += t
        
    results.append([N, Lsum])
    del cubic_file, filter

    print(N, Lsum, t)

np.save('/home1/10000/elee0506/scratch/output_000'+output_num+'/Lsum/alpha_' + type + '_DM_' + filter_density+ '_raga.npy', results)