import numpy as np
import matplotlib.pyplot as plt

# Create a 100x100x4 numpy array with all zeros (transparent background)
scale=10
image = np.ones((100*scale, 100*scale, 3))

def add_triangle(image, triangle, color):
    # Fill the triangle with red color (RGBA = [1, 0, 0, 1])
    for x in range(100*scale):
        for y in range(100*scale):
            # Check if the point (x, y) is inside the triangle using the barycentric technique
            denominator = ((triangle[1][1] - triangle[2][1]) * (triangle[0][0] - triangle[2][0]) +
                        (triangle[2][0] - triangle[1][0]) * (triangle[0][1] - triangle[2][1]))
            a = ((triangle[1][1] - triangle[2][1]) * (x - triangle[2][0]) +
                (triangle[2][0] - triangle[1][0]) * (y - triangle[2][1])) / denominator
            b = ((triangle[2][1] - triangle[0][1]) * (x - triangle[2][0]) +
                (triangle[0][0] - triangle[2][0]) * (y - triangle[2][1])) / denominator
            c = 1 - a - b

            if 0 <= a <= 1 and 0 <= b <= 1 and 0 <= c <= 1:
                image[y, x] = color  # Red color with full opacity

# Coordinates of the red triangle (vertices)
triangle1 = np.array([[25*scale, 75*scale], [75*scale, 75*scale], [50*scale, 25*scale]])
triangle2 = np.array([[25*scale, 75*scale], [50*scale, 55*scale], [45*scale, 25*scale]])
triangle3 = np.array([[35*scale, 85*scale], [40*scale, 65*scale], [25*scale, 35*scale]])
color1 = [1, 0, 0]
color2 = [0, 1, 0]
color3 = [1, 0, 1]
add_triangle(image,triangle1,color1)
add_triangle(image,triangle2,color2)
add_triangle(image,triangle3,color3)
import numpy as np

def apply_nxn_conv(image, filters,n):
    """
    Applies a set of 3x3xC filters to a given image.
    
    Parameters:
    - image: A 3D numpy array representing the image with shape (H, W, C)
             where H is the height, W is the width, and C is the number of channels.
    - filters: A numpy array of shape (num_filters, 3, 3, C) representing the filters.
    
    Returns:
    - output: A numpy array of shape (num_filters, H-2, W-2), the result of applying each filter.
    """
    H, W, C = image.shape
    num_filters, filter_height, filter_width, filter_channels = filters.shape
    
    assert filter_height == n and filter_width == n, "Filters must be 3x3 in size."
    assert filter_channels == C, "Number of channels in the filter must match the input image."

    # Calculate output dimensions
    output_height = H - n+1
    output_width = W - n+1

    # Initialize output
    output = np.zeros((output_height, output_width, num_filters))
    
    # Apply each filter to the image
    for i in range(output_height):
        for j in range(output_width):
            for f in range(num_filters):
                # Extract the 3x3 region for each position
                region = image[i:i+n, j:j+n, :]
                # Perform element-wise multiplication and sum the results
                scalar=np.sum(region * filters[f])
                output[i, j, f] = scalar

    return output

from numba import cuda, float32

@cuda.jit
def conv_kernel(image, filters, output, H, W, C, num_filters, n):
    """
    CUDA Kernel to perform convolution.

    Parameters:
    - image: 3D array of shape (H, W, C)
    - filters: 4D array of shape (num_filters, n, n, C)
    - output: 3D array of shape (num_filters, H - n + 1, W - n + 1)
    - H, W, C: Dimensions of the image
    - num_filters: Number of filters
    - n: Size of the filters (assumed square)
    """
    # Calculate the global thread indices
    f = cuda.blockIdx.z  # Filter index
    i = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y  # Height index
    j = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x  # Width index

    output_height = H - n + 1
    output_width = W - n + 1

    # Boundary check
    if f < num_filters and i < output_height and j < output_width:
        tmp = 0.0
        for c in range(C):
            for di in range(n):
                for dj in range(n):
                    tmp += image[i + di, j + dj, c] * filters[f, di, dj, c]
        output[i, j, f] = tmp

def apply_nxn_conv_cuda(image,filters,n,chunksize):
    
    #def cuda kernel (image)
    #   num_filters=filters.shape[0]
    #   i,j=blockidxx*blocksizex,blockidxy*blocksizey
    #   for ii in range(i,i+blocksizex)
    #       for jj in range(j,j+blocksizey)
    #           for f in range(num_filters)
    #                scalar=np.sum(image[ii:ii+n,jj:jj+n,:]*filters[f])
    #
    
    #put the image onto the device

    #rows,cols,channels=image.shape
    #blockx,blocky=chunksize,chunksize
    #gridx,gridy=rows//chunksize,cols//chunksize
    
    #invoke kernel


    #transfer data back to host
    #return it
    # Ensure the input data is in float32
    image = image.astype(np.float32)
    filters = filters.astype(np.float32)

    H, W, C = image.shape
    num_filters, filter_height, filter_width, filter_channels = filters.shape

    assert filter_height == n and filter_width == n, f"Filters must be {n}x{n} in size."
    assert filter_channels == C, "Number of channels in the filter must match the input image."

    output_height = H - n + 1
    output_width = W - n + 1

    # Initialize output array
    output = np.zeros((output_height, output_width, num_filters), dtype=np.float32)

    # Allocate device memory
    d_image = cuda.to_device(image)
    d_filters = cuda.to_device(filters)
    d_output = cuda.to_device(output)

    # Define block and grid dimensions
    threads_per_block = (chunksize, chunksize, 1)  # (THREADS_X, THREADS_Y, THREADS_Z)

    # Calculate grid dimensions
    grid_x = math.ceil(output_width / threads_per_block[0])
    grid_y = math.ceil(output_height / threads_per_block[1])
    grid_z = num_filters  # One block per filter

    blocks_per_grid = (grid_x, grid_y, grid_z)

    # Launch the kernel
    conv_kernel[blocks_per_grid, threads_per_block](d_image, d_filters, d_output, H, W, C, num_filters, n)

    # Copy the result back to host
    d_output.copy_to_host(output)

    return output

def normalize_output(output):
    # Normalize the output to the range 0 to 1
    output_min = np.min(output)
    output_max = np.max(output)
    normalized_output = (output - output_min) / (output_max - output_min)
    return normalized_output

import math
# And we have 3 filters, each 3x3 with 3 channels
color_filter = np.random.rand(3, 10, 10, 3)
for i in range(3):
    color_filter[i, i, i, i] += 50  # Identity mapping for each channel

import time

# Apply the convolution kernel to stylize the image
start=time.time()
stylized_image = normalize_output(apply_nxn_conv(image, color_filter,10))
end=time.time()
print(end-start)

start=time.time()
stylized_image_II = normalize_output(apply_nxn_conv_cuda(image, color_filter,10,10))
end=time.time()
print(end-start)

# Plot the original and processed images side by side
fig, axs = plt.subplots(1, 3, figsize=(10, 5))

axs[0].imshow(image)
axs[0].set_title('Original Image')
axs[0].axis('off')

axs[1].imshow(stylized_image)
axs[1].set_title('Processed Image')
axs[1].axis('off')

axs[2].imshow(stylized_image_II)
axs[2].set_title('Processe Image II')
axs[2].axis('off')

plt.show()