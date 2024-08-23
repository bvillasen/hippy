# hippy

Calling GPU functionality through HIP in you Python code

## How to use

Write your hip code and build it using `hipcc`:

```
cd HIP
make 
```

this will crate the shared library `libHIPcode.so`. 

Load it into python using `cdll` and call the HIP functions.


## Examples

### Daxpy

Simple example showing usage of basic GPU functionality and running a DAXPY kernel on the CPU and on the GPU.

```
python gpu_daxpy.py
```


### ROCTx Markers

A version of the DAXPY example with some ROCTx markers, to run use:

```
rocprof --roctx-trace python gpu_daxpy_roctx.py
```






 