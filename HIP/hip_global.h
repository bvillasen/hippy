#ifndef GLOBAL_H
#define GLOBAL_H

#define CHECK(command) {   \
  hipError_t status = command; \
  if (status!=hipSuccess) {    \
    std::cout << "Error: HIP reports " << hipGetErrorString(status) << std::endl; \
    std::abort(); }}



#endif