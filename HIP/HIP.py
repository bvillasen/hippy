import os 
from ctypes import cdll, c_int, c_void_p, c_size_t  

class hip_tools:    
  def __init__(self, hip_lib_path):
    print(f'Loading HIP library: {hip_lib_path}')
    self.libhip = cdll.LoadLibrary( hip_lib_path )

    #Load functions from the HIP code library
    self.set_device = self.libhip.hip_set_device
    self.set_device.argtypes = [ c_int ]
    self.set_device.restype = c_int

    self.allocate_device = self.libhip.hip_allocate_device
    self.allocate_device.argtypes = [ c_size_t ]
    self.allocate_device.restype = c_void_p

    self.free_device = self.libhip.hip_free_device
    self.free_device.argtypes = [ c_void_p ] 

    self.copy_host_to_device = self.libhip.hip_copy_host_to_device
    self.copy_host_to_device.argtypes = [ c_void_p, c_void_p, c_size_t ]

    self.copy_device_to_host = self.libhip.hip_copy_device_to_host
    self.copy_device_to_host.argtypes = [ c_void_p, c_void_p, c_size_t ]    


  def set_device( self, device_indx ):
    dev_id = self.set_device( device_indx )
    return dev_id
    
  def allocate_device( self, array_size ):
    device_ptr = self.allocate_device( array_size )
    return device_ptr
  
  def free_device( self, array_ptr ):
    self.free_device( array_ptr )
  
  def copy_host_to_device( self, dev_ptr, host_ptr, array_size ):
    self.copy_host_to_device( dev_ptr, host_ptr, array_size )
