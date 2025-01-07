# Testing of dust attenuation filter (incomplete)

import numpy as np
import astropy
import math
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

unit_d_path = '/home1/10000/elee0506/unit_d.npy'
redshift_path = '/home1/10000/elee0506/redshift.npy'
scale_path = '/home1/10000/elee0506/scale_factor.npy'
unit_d = np.load(unit_d_path).astype(np.float32)
redshift = np.load(redshift_path).astype(np.float32)
scale = np.load(scale_path).astype(np.float32)

output_num = '118'
type = 'rec'

kappa = 3.251e3 #[cm^2/g] : Kappa_abs = absorption cross section per mass of dust of H_alpha
const = kappa * 20 * 0.01152709961 * 3.086e+24 * scale[int(output_num)-1] * unit_d[int(output_num)-1]
#kappa * 20cells * 1 cell in physical cm * unit_d

rec_path = '/home1/10000/elee0506/scratch/output_000' + output_num + '/rec/'
col_path = '/home1/10000/elee0506/scratch/output_000' + output_num + '/col/'
attenuation_path = '/scratch/08389/tg876886/dust_sum/output_000' +output_num + '/'

if type == 'col':
    file_path = col_path
if type == 'rec':
    file_path = rec_path


Lsum = 0
results = []
for N in range(64):

    cubic_file = np.load(os.path.join(file_path, type + f'_{N:02}.npy'))
    attenuation = np.load(os.path.join(attenuation_path, f'dust_sum_20_{N:02}.npy')) * const  # tau 
    post_att = cubic_file * np.exp(-attenuation)
    np.save('/home1/10000/elee0506/scratch/output_000'+output_num+'/post_attenuation/alpha_'+type+'_attenuation_'+ f'{N:02}.npy', post_att)         
    t = np.add.reduce(post_att).sum()
    Lsum += t
    results.append([N, Lsum, t])
    del cubic_file
    del attenuation
    del post_att
    print(N, Lsum, t)

np.save('/home1/10000/elee0506/scratch/output_000'+output_num+'/att_list/alpha_'+ type + '_attenuation.npy', results)