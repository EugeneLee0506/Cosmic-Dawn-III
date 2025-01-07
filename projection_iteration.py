#Same as projection_10%.py, but iterates procedure for all the specified snaps (output_nums), 
#and also perform projection for recombinational, collisional, total, and saves each proejction to corresponding path. 

import numpy as np
import time  # To measure runtime

start_time = time.time()

# List of output numbers. Can be manually adjusted
output_nums = ['090', '082', '052', '034', '023', '014']

def get_indices_from_N(N):
    """Map integer N (from 0 to 63) to 3D indices (i, j, k)."""
    k = N // 16  # Divide by 16 to get the i index (row)
    j = (N % 16) // 4  # Mod 16 and divide by 4 to get the j index
    i = N % 4  # Mod 4 to get the k index
    return i, j, k

def calculate_projection(file_path, type_name, output_num):
    """Calculate 2D projection for a given file path and type."""
    # Initialize an empty array for the 2D projection of shape (8192, 8192)
    array = np.zeros((8192, 8192), dtype='float32')

    sub_i = 2048  # Size of each cube along each axis

    # Iterate over the first 16 cubes (N=0 to 15)
    for N in range(16):
        i, j, k = get_indices_from_N(N)  # Get the (i, j, k) position of the cube
        
        # Load the corresponding 2048x2048x2048 cube
        file = np.load(file_path + f"{type_name}_{N:02}.npy")
        
        # Project the first 819 slices along the Z-axis, sum over the Z-dimension (axis=2)
        array_2d = np.sum(file[:, :, :819], axis=2)  # Sum over the first 819 slices
        
        del file

        # Add the summed result to the appropriate place in the larger 2D array
        array[(sub_i*i):(sub_i*(i+1)), (sub_i*j):(sub_i*(j+1))] += array_2d
        del array_2d

    # Save the final 2D projection array of shape (8192, 8192)
    save_path = f'/home1/10000/elee0506/scratch/output_000{output_num}/projection/10%projection_{type_name}.npy'
    np.save(save_path, array)
    print(f"Projection for type '{type_name}' saved to {save_path}")
    return array  # Return the array for later use

# Process each output number
for output_num in output_nums:
    rec_path = f'/home1/10000/elee0506/scratch/output_000{output_num}/rec/'
    col_path = f'/home1/10000/elee0506/scratch/output_000{output_num}/col/'
    
    print(f"Processing output_num {output_num}...")

    # Calculate for type = 'rec'
    array_rec = calculate_projection(rec_path, 'rec', output_num)

    # Calculate for type = 'col'
    array_col = calculate_projection(col_path, 'col', output_num)

    # Combine the two arrays
    array_combined = array_rec + array_col

    # Save the combined projection
    combined_save_path = f'/home1/10000/elee0506/scratch/output_000{output_num}/projection/10%projection_combined.npy'
    np.save(combined_save_path, array_combined)
    print(f"Combined projection for output_num {output_num} saved to {combined_save_path}")

# End the timer
end_time = time.time()
total_time = end_time - start_time
print(f"Total runtime: {total_time:.2f} seconds")