#include <iostream>
#include "litmus.cuh"
#include "functions.cu"

__global__ void litmus_test(
  d_atomic_uint* test_locations,
  ReadResults* read_results,
  uint* shuffled_workgroups,
  cuda::atomic<uint, cuda::thread_scope_device>* barrier,
  uint* scratchpad,
  uint* scratch_locations,
  KernelParams* kernel_params,
  TestInstance* test_instances) {

  uint shuffled_workgroup = shuffled_workgroups[blockIdx.x];
  if (shuffled_workgroup < kernel_params->testing_workgroups) {

#ifdef RELEASE
    cuda::memory_order store_order = cuda::memory_order_release;
    #define FENCE()
#elif defined(RELAXED)
    cuda::memory_order store_order = cuda::memory_order_relaxed;
    #define FENCE()
#elif defined(BOTH_FENCE)
    cuda::memory_order store_order = cuda::memory_order_relaxed;
    #define FENCE() cuda::atomic_thread_fence(cuda::memory_order_acq_rel, FENCE_SCOPE);
#else
    cuda::memory_order store_order = cuda::memory_order_relaxed;
    #define FENCE()
#endif

    // defined for different distributions of threads across threadblocks
    DEFINE_IDS();

    uint x_0 = (wg_offset + id_0) * kernel_params->mem_stride * 2;
    uint y_0 = (wg_offset + permute_id(id_0, kernel_params->permute_location, total_ids)) * kernel_params->mem_stride * 2 + kernel_params->mem_offset;
    uint x_1 = (wg_offset + id_1) * kernel_params->mem_stride * 2;
    uint y_1 = (wg_offset + permute_id(id_1, kernel_params->permute_location, total_ids)) * kernel_params->mem_stride * 2 + kernel_params->mem_offset;

    PRE_STRESS();

    if (id_0 != id_1) {

      // Thread 0
      test_locations[x_0].store(1, cuda::memory_order_relaxed);
      FENCE()
      test_locations[y_0].store(2, store_order);

      // Thread 1
      test_locations[y_1].store(1, cuda::memory_order_relaxed);
      FENCE()
      test_locations[x_1].store(2, store_order);
    }
  }

  MEM_STRESS();
}

__global__ void check_results(
  d_atomic_uint* test_locations,
  ReadResults* read_results,
  TestResults* test_results,
  KernelParams* kernel_params,
  bool* weak) {
  RESULT_IDS();
  uint id_0 = blockIdx.x * blockDim.x + threadIdx.x;
  uint x = test_locations[id_0 * kernel_params->mem_stride * 2];
  uint y_loc = (wg_offset + permute_id(id_0, kernel_params->permute_location, total_ids)) * kernel_params->mem_stride * 2 + kernel_params->mem_offset;
  uint y = test_locations[y_loc];


  if (x == 1 && y == 1) {
    test_results->weak.fetch_add(1);
  }
  else if (x == 0) { // thread skipped
    test_results->na.fetch_add(1);
  }
  else {
    test_results->other.fetch_add(1);
  }
}

int host_check_results(TestResults* results, bool print) {
  if (print) {
    std::cout << "x=1, y=1 (weak): " << results->weak << "\n";
    std::cout << "thread skipped: " << results->na << "\n";
    std::cout << "other: " << results->other << "\n";
  }
  return results->weak;
}
