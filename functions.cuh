// functions.cuh
#ifndef FUNCTIONS_CUH
#define FUNCTIONS_CUH
#endif

#include <cuda_runtime.h>
#include <cuda/atomic>

__device__ uint permute_id(uint id, uint factor, uint mask);
__device__ uint stripe_workgroup(uint workgroup_id, uint local_id, uint testing_workgroups);
__device__ void spin(cuda::atomic<uint>* barrier, uint limit);
__device__ void do_stress(uint* scratchpad, uint* scratch_locations, uint iterations, uint pattern);
