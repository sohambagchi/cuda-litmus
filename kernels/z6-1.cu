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

#ifdef ACQ_REL 
    cuda::memory_order store_order = cuda::memory_order_release;
    cuda::memory_order load_order = cuda::memory_order_acquire;
#elif defined(RELAXED)
    cuda::memory_order store_order = cuda::memory_order_relaxed;
    cuda::memory_order load_order = cuda::memory_order_relaxed;
#else
    cuda::memory_order store_order = cuda::memory_order_relaxed;
    cuda::memory_order load_order = cuda::memory_order_relaxed;
#endif

    // defined for different distributions of threads across threadblocks
    DEFINE_IDS();

    // defined for all three thread two memory locations tests
    THREE_THREAD_TWO_MEM_LOCATIONS();

    PRE_STRESS();

    if (id_0 != id_1 && id_1 != id_2 && id_0 != id_2) {

      // Thread 0
      test_locations[x_0].store(2, store_order);
      test_locations[y_0].store(1, store_order);

      // Thread 1
      test_locations[y_1].store(2, store_order);
      test_locations[z_1].store(1, store_order);

      // Thread 2
      uint r0 = test_locations[z_2].load(load_order);
      test_locations[x_2].store(1, store_order);

      cuda::atomic_thread_fence(cuda::memory_order_seq_cst);
      read_results[wg_offset + id_2].r0 = r0;
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
  uint id_0 = blockIdx.x * blockDim.x + threadIdx.x;
  uint r0 = read_results[id_0].r0;
  uint x = test_locations[id_0 * kernel_params->mem_stride * 2];
  uint y_loc = (wg_offset + permute_id(id_0, kernel_params->permute_location, total_ids)) * kernel_params->mem_stride * 2 + kernel_params->mem_offset;
  uint y = test_locations[y_loc];

  if (x == 0) {
    test_results->na.fetch_add(1); // thread skipped
  }
  else if (x == 2 && y == 2 && r0 == 1) {
    test_results->weak.fetch_add(1);
    weak[id_0] = true;
  }
  else {
    test_results->other.fetch_add(1);
  }
}

int host_check_results(TestResults* results, bool print) {
  if (print) {
    std::cout << "r0=1, x=2, y=2 (weak): " << results->weak << "\n";
    std::cout << "thread skipped: " << results->na << "\n";
    std::cout << "other: " << results->other << "\n";
  }
  return results->weak;
}

