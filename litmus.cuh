// litmus.cuh
#ifndef LITMUS_CUH
#define LITMUS_CUH
#endif

#include <cuda_runtime.h>
#include <cuda/atomic>

#ifdef SCOPE_DEVICE
typedef cuda::atomic<uint, cuda::thread_scope_device> d_atomic_uint;
#elif defined(SCOPE_BLOCK)
typedef cuda::atomic<uint, cuda::thread_scope_block> d_atomic_uint;
#elif defined(SCOPE_SYSTEM)
typedef cuda::atomic<uint, cuda::thread_scope_system> d_atomic_uint;
#else
typedef cuda::atomic<uint> d_atomic_uint; // default, which is system too
#endif

#ifdef FENCE_SCOPE_BLOCK
    cuda::thread_scope fence_scope = cuda::thread_scope_block;
#elif defined(FENCE_SCOPE_DEVICE)
    cuda::thread_scope fence_scope = cuda::thread_scope_device;
#elif defined(FENCE_SCOPE_SYSTEM)
    cuda::thread_scope fence_scope = cuda::thread_scope_system;
#else
    cuda::thread_scope fence_scope = cuda::thread_scope_system; // default to system scope
#endif


#define THREE_THREAD_TWO_MEM_LOCATIONS() \
  uint x_0 = (wg_offset + id_0) * kernel_params->mem_stride * 2; \
  uint x_1 = (wg_offset + id_1) * kernel_params->mem_stride * 2; \
  uint y_1 = (wg_offset + permute_id(id_1, kernel_params->permute_location, total_ids)) * kernel_params->mem_stride * 2 + kernel_params->mem_offset; \
  uint x_2 = (wg_offset + id_2) * kernel_params->mem_stride * 2; \
  uint y_2 = (wg_offset + permute_id(id_2, kernel_params->permute_location, total_ids)) * kernel_params->mem_stride * 2 + kernel_params->mem_offset;



#define PRE_STRESS() \
  if (kernel_params->pre_stress) { \
    do_stress(scratchpad, scratch_locations, kernel_params->pre_stress_iterations, kernel_params->pre_stress_pattern); \
  } \
  if (kernel_params->barrier) { \
    spin(barrier, blockDim.x * kernel_params->testing_workgroups); \
  }

#define MEM_STRESS() \
  else if (kernel_params->mem_stress) { \
    do_stress(scratchpad, scratch_locations, kernel_params->mem_stress_iterations, kernel_params->pre_stress_iterations); \
  }

typedef struct {
  uint r0;
  uint r1;
  uint r2;
  uint r3;
} ReadResults;

typedef struct {
  cuda::atomic<uint, cuda::thread_scope_device> res0; // up to 16 combinations of valid test results in the implemented tests
  cuda::atomic<uint, cuda::thread_scope_device> res1;
  cuda::atomic<uint, cuda::thread_scope_device> res2;
  cuda::atomic<uint, cuda::thread_scope_device> res3;
  cuda::atomic<uint, cuda::thread_scope_device> res4;
  cuda::atomic<uint, cuda::thread_scope_device> res5;
  cuda::atomic<uint, cuda::thread_scope_device> res6;
  cuda::atomic<uint, cuda::thread_scope_device> res7;
  cuda::atomic<uint, cuda::thread_scope_device> res8;
  cuda::atomic<uint, cuda::thread_scope_device> res9;
  cuda::atomic<uint, cuda::thread_scope_device> res10;
  cuda::atomic<uint, cuda::thread_scope_device> res11;
  cuda::atomic<uint, cuda::thread_scope_device> res12;
  cuda::atomic<uint, cuda::thread_scope_device> res13;
  cuda::atomic<uint, cuda::thread_scope_device> res14;
  cuda::atomic<uint, cuda::thread_scope_device> res15;
  cuda::atomic<uint, cuda::thread_scope_device> weak; // this is the weak behavior we are looking for
  cuda::atomic<uint, cuda::thread_scope_device> na; // some threads don't execute the test if thread ids clash
  cuda::atomic<uint, cuda::thread_scope_device> other; // this should always be 0
} TestResults;


typedef struct {
  bool barrier;
  bool mem_stress;
  int mem_stress_iterations;
  int mem_stress_pattern;
  bool pre_stress;
  int pre_stress_iterations;
  int pre_stress_pattern;
  int permute_thread;
  int permute_location;
  int testing_workgroups;
  int mem_stride;
  int mem_offset;
} KernelParams;

__global__ void litmus_test(
  d_atomic_uint* test_locations,
  ReadResults* read_results,
  uint* shuffled_workgroups,
  cuda::atomic<uint, cuda::thread_scope_device>* barrier,
  uint* scratchpad,
  uint* scratch_locations,
  KernelParams* kernel_params);

__global__ void check_results(
  d_atomic_uint* test_locations,
  ReadResults* read_results,
  TestResults* test_results,
  KernelParams* kernel_params);

int host_check_results(TestResults* results, bool print);

