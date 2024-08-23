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

# Execute CPU daxpy
result_cpu = A * host_x + host_y

# Execute GPU daxpy
hip.gpu_daxpy( device_x, device_y, A )

result_gpu = np.zeros_like( result_cpu )
device_y.copy_to_host( result_gpu )
hip.synchronize_device()

#Validate
diff = np.abs( result_gpu - result_cpu )
print( f'Max difference: {diff.max()}' )



 








