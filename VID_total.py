# Adds the cell-wise luminosity density due to both collision and recombination. Based on the summed total luminosity densities, this code creates 1D histogram (i.e. VID)

import time
import astropy
import math
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import numpy as np
import gc
import os 
from numba import njit

outputnum = '034'

@njit
def generic_1d_histogram(data, bin_edges):
    hist_shape = len(bin_edges) - 1
    hist = np.zeros(hist_shape, dtype=np.uint64)
    # Iterate over each point in the dataset
    for point in data:
        i0 = np.searchsorted(bin_edges, point, side='right') - 1
        if (0 <= i0 < hist_shape):
            hist[i0] += 1
    return hist

start_time = time.time()
nbin = 1000
bin_edges = np.logspace(-40, -21, nbin + 1)
hist_cumulative = np.zeros(nbin, dtype=np.uint64)
directory_rec = '/home1/10000/elee0506/scratch/output_000' + outputnum + '/rec/'
directory_col = '/home1/10000/elee0506/scratch/output_000' + outputnum + '/col/'




# Assuming generic_1d_histogram and bin_edges are defined elsewhere
for i in range(64):
    # Construct file paths
    filename_rec = f'rec_{i:02d}.npy'
    filename_col = f'col_{i:02d}.npy'
    file_path_rec = os.path.join(directory_rec, filename_rec)
    file_path_col = os.path.join(directory_col, filename_col)

    # Load the data
    lumin_rec = np.load(file_path_rec)
    lumin_col = np.load(file_path_col)

    # Combine the arrays in-place (avoids creating extra copies)
    lumin_rec += lumin_col
    del lumin_col  # Delete the 'col' array after combining

    # Flatten and filter in-place to reduce memory usage
    lumin_filtered = lumin_rec.ravel()
    lumin_filtered = lumin_filtered[(lumin_filtered >= 10**-40) & (lumin_filtered <= 10**-21)]
    
    # Delete the original 'rec' array and trigger garbage collection to free memory
    del lumin_rec
    gc.collect()

    # Compute the histogram
    hist_current = generic_1d_histogram(lumin_filtered, bin_edges)

    # Add to cumulative histogram (avoid storing large intermediate results if possible)
    hist_cumulative += hist_current

    # Save the current histogram to disk
    np.save('/home1/10000/elee0506/scratch/VID/' + outputnum + '_VID/' + f'combined_{i:02d}.npy', hist_current)
    
    # Delete the current histogram to free memory
    del hist_current
    gc.collect()  # Force garbage collection to ensure memory is freed




# Calculate bin midpoints for plotting
bin_midpoints = 0.5 * (bin_edges[:-1] + bin_edges[1:])
np.save('/home1/10000/elee0506/scratch/VID/' + outputnum + '_VID/' + 'combined_hist_cumulative.npy', hist_cumulative)
np.save('/home1/10000/elee0506/scratch/VID/' + outputnum + '_VID/' + 'combined_bin_midpoints.npy', bin_midpoints)

# Track execution time
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Total execution time: {elapsed_time:.2f} seconds")