#Calculates the min/max value (cell-wise luminosity density) from either rec_00 to 63 or col_00 to 63 (For the VID min/max value analysis) 

import numpy as np
import time
import os

start_time = time.time()

# List of output numbers to process
output_nums = ["000014", "000034", "000082", "000118"]

for output_num in output_nums:
    directory_rec = f'/home1/10000/elee0506/scratch/output_{output_num}/rec/'

    # Initialize global min and max values for each output_num
    global_min_rec = float('inf')
    global_max_rec = float('-inf')

    for i in range(64):
        filename_rec = f'rec_{i:02d}.npy'
        file_path_rec = os.path.join(directory_rec, filename_rec)

        # Load the data from the file
        lumin_rec = np.load(file_path_rec)

        # Find the local min and max values
        local_min_rec = np.min(lumin_rec)
        local_max_rec = np.max(lumin_rec)

        # Update global min and max
        if local_min_rec < global_min_rec:
            global_min_rec = local_min_rec
        if local_max_rec > global_max_rec:
            global_max_rec = local_max_rec

        # Free memory
        del lumin_rec

    # Save the global min and max values for this output_num
    print(f"Global minimum value of recombination for {output_num}: {global_min_rec}")
    print(f"Global maximum value of recombination for {output_num}: {global_max_rec}")
    np.save(f'/home1/10000/elee0506/scratch/VID/{output_num}_rec_min.npy', global_min_rec)
    np.save(f'/home1/10000/elee0506/scratch/VID/{output_num}_rec_max.npy', global_max_rec)

end_time = time.time()
# Calculate and print the elapsed time
elapsed_time = end_time - start_time
print(f"Total execution time: {elapsed_time:.2f} seconds")