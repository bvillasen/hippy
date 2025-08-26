import sys
sys.path.append('HIP')
import numpy as np
import hip_tools as hip
from gpuarray import gpu_array

# Initialize HIP
hip.initialize( 'HIP/libHIPcode.so' )
hip.set_device(0)

# Initialize CPU data
A = 10.0
n = 1024 * 1024 * 256
host_x = np.random.rand(n)
host_y = np.random.rand(n)

# Initialize GPU data
device_x = gpu_array( np_arr=host_x )
device_y = gpu_array( np_arr=host_y )

n_iter = 100

for i in range(n_iter):
  
  if i == 90: hip.roctracer_start()

  # Execute GPU daxpy
  hip.gpu_daxpy( device_x, device_y, A )

hip.synchronize_device()
hip.roctracer_stop()  





 








