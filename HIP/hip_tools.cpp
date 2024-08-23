#include <stdio.h>
#include <iostream>
#include <hip/hip_runtime.h>
#include "hip_global.h"


extern "C" {


int hip_set_device( int device_id ) {
  int n_devices; 
	CHECK( hipGetDeviceCount(&n_devices) );
	if (n_devices == 0){
		std::cout << "ERROR: No hip devices found" << std::endl;
		return -1;
	}
	int device = device_id % n_devices;
	CHECK( hipSetDevice(device) );

  hipDeviceProp_t prop;
	CHECK( hipGetDeviceProperties( &prop, device_id ) );
  std::cout << "Using device: " << device_id << " " << prop.name  << std::endl;
	return device;
}

void hip_synchronize_device(){
	CHECK( hipDeviceSynchronize());
}

void* hip_allocate_device( size_t nbytes ){	
	void* dev_array;
	CHECK(hipMalloc( &dev_array, nbytes ));
	return dev_array;
}

void hip_free_device( void* dev_array ){
	CHECK(hipFree( dev_array));
}

void hip_copy_host_to_device( void *dev_array, void *host_array, size_t nbytes ){
	CHECK(hipMemcpy( dev_array, host_array, nbytes, hipMemcpyHostToDevice));
}

void hip_copy_device_to_host( void *host_array, void *dev_array, size_t nbytes ){
	CHECK(hipMemcpy( host_array, dev_array, nbytes, hipMemcpyDeviceToHost));
}

#define TPB 256
__global__ void __launch_bounds__(TPB, 1)
daxpy_kernel( double *x, double *y, double a, int N ){
  int tid = blockIdx.x *blockDim.x + threadIdx.x;
  if ( tid >= N ) return;
  y[tid] += a*x[tid];
}  

void hip_daxpy( double *x, double *y, double a, int N  ){
  int n_blocks = (N-1)/TPB + 1;
  daxpy_kernel<<<n_blocks,TPB>>>( x, y, a, N);
}

}