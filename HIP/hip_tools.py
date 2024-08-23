from ctypes import cdll, c_int, c_void_p, c_size_t, c_double, c_char_p  

encode = lambda s : s.encode('utf-8')

hip_set_device, synchronize_device = None, None
hip_allocate_device, free_device = None, None
copy_host_to_device, copy_device_to_host = None, None
_roctx_push, roctx_pop = None, None
hip_daxpy = None


def initialize(hip_lib_path):
  global hip_set_device, synchronize_device
  global hip_allocate_device, free_device
  global copy_host_to_device, copy_device_to_host 
  global _roctx_push, roctx_pop
  global hip_daxpy 

  print(f'Loading HIP library: {hip_lib_path}')
  libhip = cdll.LoadLibrary( hip_lib_path )

  #Load functions from the HIP code library
  hip_set_device = libhip.hip_set_device
  hip_set_device.argtypes = [ c_int ]
  hip_set_device.restype = c_int

  synchronize_device = libhip.hip_synchronize_device

  hip_allocate_device = libhip.hip_allocate_device
  hip_allocate_device.argtypes = [ c_size_t ]
  hip_allocate_device.restype = c_void_p

  free_device = libhip.hip_free_device
  free_device.argtypes = [ c_void_p ] 

  copy_host_to_device = libhip.hip_copy_host_to_device
  copy_host_to_device.argtypes = [ c_void_p, c_void_p, c_size_t ]

  copy_device_to_host = libhip.hip_copy_device_to_host
  copy_device_to_host.argtypes = [ c_void_p, c_void_p, c_size_t ]    

  _roctx_push = libhip.roctxr_push
  _roctx_push.argtype = [c_char_p]

  roctx_pop = libhip.roctxr_pop

  hip_daxpy = libhip.hip_daxpy
  hip_daxpy.argtypes = [ c_void_p, c_void_p, c_double, c_int ] 

def set_device( device_indx ):
  dev_id = hip_set_device( device_indx )
  return dev_id
  
def allocate_device( array_size ):
  device_ptr = hip_allocate_device( array_size )
  return device_ptr

def roctx_push( tag_name ): 
  _roctx_push( encode(tag_name) )

def gpu_daxpy( device_x, device_y, a):
  N = device_x.size
  hip_daxpy( device_x.dev_ptr, device_y.dev_ptr, a, N )