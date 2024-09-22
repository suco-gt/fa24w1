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

# Plot the original and processed images side by side
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

axs[0].imshow(image)
axs[0].set_title('Original Image')
axs[0].axis('off')

axs[1].imshow(stylized_image)
axs[1].set_title('Processed Image')
axs[1].axis('off')

plt.show()