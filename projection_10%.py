# Projects (sums) 10% of the quantity (cell-wise luminosity density) of the cube. Projection depth could be manually modified. 

import numpy as np
output_num = '118'
type = 'rec'

rec_path = '/home1/10000/elee0506/scratch/output_000' + output_num + '/rec/'
col_path = '/home1/10000/elee0506/scratch/output_000' + output_num + '/col/'

if type == 'col':
    file_path = col_path
if type == 'rec':
    file_path = rec_path

def get_indices_from_N(N):
    """Map integer N (from 0 to 63) to 3D indices (i, j, k)."""
    k = N // 16  # Divide by 16 to get the i index (row)
    j = (N % 16) // 4  # Mod 16 and divide by 4 to get the j index
    i = N % 4  # Mod 4 to get the k index
    return i, j, k

# Initialize an empty array for the 2D projection of shape (8192, 8192)
array = np.zeros((8192, 8192), dtype='float32')

sub_i = 2048  # Size of each cube along each axis

# Iterate over the first 16 cubes (N=0 to 15)
for N in range(16):
    i, j, k = get_indices_from_N(N)  # Get the (i, j, k) position of the cube
    
    # Load the corresponding 2048x2048x2048 cube
    file = np.load(file_path + f"{type}_{N:02}.npy")
    
    # Project the first 819 slices along the Z-axis, sum over the Z-dimension (axis=2)
    array_2d = np.sum(file[:, :, :819], axis=2)  # Sum over the first 819 slices / 10% size of the cube (8192 cells in total). Ranges can be changed depending on desired projection depth.
    
    del file

    # Add the summed result to the appropriate place in the larger 2D array
    # We now place each cube based on (i, j, k) indices in the final 2D array
    # Each sub-cube size is 2048x2048, so place them in the corresponding area of the 8192x8192 array
    array[(sub_i*i):(sub_i*(i+1)), (sub_i*j):(sub_i*(j+1))] += array_2d
    del array_2d

# Save the final 2D projection array of shape (8192, 8192)
np.save('/home1/10000/elee0506/scratch/output_000' + output_num + '/projection/10%projection_' + type + '.npy', array)

