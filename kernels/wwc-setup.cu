#include <iostream>
#include "litmus.cuh"
#include "functions.cuh"

#ifdef TB_0_1_2
#include "tb-0-1-2.h"
#elif defined(TB_01_2)
#include "tb-01-2.h"
#elif defined(TB_0_12)
#include "tb-0-12.h"
#elif defined(TB_012)
#include "tb-012.h"
#else
#include "tb-0-1-2.h" // default to all different threadblocks
#endif

__global__ void litmus_test(
  d_atomic_uint* test_locations,
  ReadResults* read_results,
  uint* shuffled_workgroups,
  cuda::atomic<uint, cuda::thread_scope_device>* barrier,
  uint* scratchpad,
  uint* scratch_locations,
  KernelParams* kernel_params) {

  uint shuffled_workgroup = shuffled_workgroups[blockIdx.x];
  if (shuffled_workgroup < kernel_params->testing_workgroups) {

#ifdef ACQ_REL
    cuda::memory_order thread_1_load = cuda::memory_order_acquire;
    cuda::memory_order thread_1_store = cuda::memory_order_relaxed;
    cuda::memory_order thread_2_load = cuda::memory_order_relaxed;
    cuda::memory_order thread_2_store = cuda::memory_order_release;
#elif defined(ACQ_ACQ)
    cuda::memory_order thread_1_load = cuda::memory_order_acquire;
    cuda::memory_order thread_1_store = cuda::memory_order_relaxed;
    cuda::memory_order thread_2_load = cuda::memory_order_acquire;
    cuda::memory_order thread_2_store = cuda::memory_order_relaxed;
#elif defined(REL_ACQ)
    cuda::memory_order thread_1_load = cuda::memory_order_relaxed;
    cuda::memory_order thread_1_store = cuda::memory_order_release;
    cuda::memory_order thread_2_load = cuda::memory_order_acquire;
    cuda::memory_order thread_2_store = cuda::memory_order_relaxed;
#elif defined(REL_REL)
    cuda::memory_order thread_1_load = cuda::memory_order_relaxed;
    cuda::memory_order thread_1_store = cuda::memory_order_release;
    cuda::memory_order thread_2_load = cuda::memory_order_relaxed;
    cuda::memory_order thread_2_store = cuda::memory_order_release;
#elif defined(RELAXED)
    cuda::memory_order thread_1_load = cuda::memory_order_relaxed;
    cuda::memory_order thread_1_store = cuda::memory_order_relaxed;
    cuda::memory_order thread_2_load = cuda::memory_order_relaxed;
    cuda::memory_order thread_2_store = cuda::memory_order_relaxed;
#else
    cuda::memory_order thread_1_load = cuda::memory_order_relaxed; // default to all relaxed
    cuda::memory_order thread_1_store = cuda::memory_order_relaxed;
    cuda::memory_order thread_2_load = cuda::memory_order_relaxed;
    cuda::memory_order thread_2_store = cuda::memory_order_relaxed;
#endif

    // defined for different distributions of threads across threadblocks
    DEFINE_IDS();

    // defined for all three thread two memory locations tests
    THREE_THREAD_TWO_MEM_LOCATIONS();

    PRE_STRESS();

    if (id_0 != id_1 && id_1 != id_2) {

      // Thread 0
      test_locations[x_0].store(2, cuda::memory_order_relaxed);

      // Thread 1
      uint r0 = test_locations[x_1].load(thread_1_load);
      test_locations[y_1].store(1, thread_1_store);

      // Thread 2
      uint r1 = test_locations[y_2].load(thread_2_load);
      test_locations[x_2].store(1, thread_2_store);

      cuda::atomic_thread_fence(cuda::memory_order_seq_cst);
      read_results[wg_offset + id_1].r0 = r0;
      read_results[wg_offset + id_2].r1 = r1;
    }
  }

  MEM_STRESS();
}

__global__ void check_results(
  d_atomic_uint* test_locations,
  ReadResults* read_results,
  TestResults* test_results,
  KernelParams* kernel_params) {
  uint id_0 = blockIdx.x * blockDim.x + threadIdx.x;
  uint r0 = read_results[id_0].r0;
  uint r1 = read_results[id_0].r1;
  uint x = test_locations[id_0 * kernel_params->mem_stride * 2];

  if (r0 == 1 && r1 == 1 && x == 1) {
    test_results->seq_inter1.fetch_add(1); // this is actually a load buffer weak behavior
  }
  else if (r0 == 1 && r1 == 1 && x == 2) { // also load buffer weak behavior
    test_results->seq_inter1.fetch_add(1);
  }
  else if (r0 == 2 && r1 == 1 && x == 2) { // this is the non-mca weak behavior
    test_results->weak.fetch_add(1);
  } 
  else if (r0 <= 2 && r1 <= 1 && (x == 1 || x == 2)) { // catch all for other sequential/interleaved behaviors
    test_results->seq0.fetch_add(1);
  }
  else {
    test_results->other.fetch_add(1);
  }
}

int host_check_results(TestResults* results, bool print) {
  if (print) {
    std::cout << "r0 <= 2, r1 <= 1, x = (1 || 2) (seq/interleaved): " << results->seq0 << "\n";
    std::cout << "r0=1, r1=1, x = (1 || 2) (lb weak): " << results->seq_inter1 << "\n";
    std::cout << "r0=2, r1=1, x=2 (weak): " << results->weak << "\n";
    std::cout << "other: " << results->other << "\n";
  }
  return results->weak;
}
