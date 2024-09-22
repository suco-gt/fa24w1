import numpy as np
import time

# Define vector size
N = 250000000

# CPU implementation of vector addition
def cpu_vector_add(a, b):
    return a + b

# Create input vectors
a = np.random.rand(N).astype(np.float32)
b = np.random.rand(N).astype(np.float32)
c = np.zeros(N, dtype=np.float32)

# Time the CPU vector addition
start = time.time()
c_cpu = cpu_vector_add(a, b)
cpu_time = time.time() - start
print(f"CPU time: {cpu_time:.5f} seconds")