import numpy
import ctypes
import hip_tools as hip

class gpu_array:
  def __init__(self, np_arr=None, copy_to_device=True):
    self.dtype   = None
    self.shape   = None
    self.size    = 0
    self.nbytes  = 0
    self.dev_ptr = None 
    if np_arr is not None:
      np_ptr       = np_arr.ctypes.data_as( ctypes.c_void_p )
      self.dtype   = np_arr.dtype 
      self.size    = np_arr.size
      self.nbytes  = np_arr.itemsize * self.size
      self.shape   = np_arr.shape
      self.dev_ptr = ctypes.c_void_p( hip.allocate_device( self.nbytes ) )
      if copy_to_device: hip.copy_host_to_device( self.dev_ptr, np_ptr, self.nbytes )
      return 
  
  def copy_to_host( self, np_arr ):
    np_size     = np_arr.size
    np_itemsize = np_arr.itemsize
    np_ptr      = np_arr.ctypes.data_as( ctypes.c_void_p )
    if self.dev_ptr is None:
      return
    if self.size != np_size:
      print('ERROR: GPUArray copy_to_host. np_size != size ')
      return 
    if self.nbytes != np_itemsize*np_size:
      print('ERROR: GPUArray copy_to_host. np_nbytes != nbytes ')
      return
    hip.copy_device_to_host( np_ptr, self.dev_ptr, self.nbytes )

    

