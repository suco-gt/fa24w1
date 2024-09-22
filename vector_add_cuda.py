import numpy as np
import time
from numba import cuda, float32

# Define vector size
N = 250000000

# CPU implementation of vector addition
def cpu_vector_add(a, b):
    return a + b

# CUDA kernel for vector addition
@cuda.jit
def gpu_vector_add(a, b, c):
    idx = cuda.grid(1)
    if idx < a.size:
        c[idx] = a[idx] + b[idx]

# Create input vectors
a = np.random.rand(N).astype(np.float32)
b = np.random.rand(N).astype(np.float32)
c = np.zeros(N, dtype=np.float32)

# Time the CPU vector addition
start = time.time()
c_cpu = cpu_vector_add(a, b)
cpu_time = time.time() - start
print(f"CPU time: {cpu_time:.5f} seconds")

# Allocate memory on the GPU
a_gpu = cuda.to_device(a)
b_gpu = cuda.to_device(b)
c_gpu = cuda.device_array(N)

# Time the GPU vector addition
threads_per_block = 1024
blocks_per_grid = (N + (threads_per_block - 1)) // threads_per_block

start = time.time()
gpu_vector_add[blocks_per_grid, threads_per_block](a_gpu, b_gpu, c_gpu)
cuda.synchronize()
gpu_time = time.time() - start

# Copy result back to the host
c_result = c_gpu.copy_to_host()

print(f"GPU time: {gpu_time:.5f} seconds")