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

#ifdef STORE_SC 
    cuda::memory_order thread_1_load = cuda::memory_order_acquire;
    cuda::memory_order thread_2_store = cuda::memory_order_seq_cst;
    cuda::memory_order thread_2_load = cuda::memory_order_relaxed;
#elif defined(LOAD_SC)
    cuda::memory_order thread_1_load = cuda::memory_order_acquire;
    cuda::memory_order thread_2_store = cuda::memory_order_relaxed;
    cuda::memory_order thread_2_load = cuda::memory_order_seq_cst;
#elif defined(RELAXED)
    cuda::memory_order thread_1_load = cuda::memory_order_relaxed;
    cuda::memory_order thread_2_store = cuda::memory_order_relaxed;
    cuda::memory_order thread_2_load = cuda::memory_order_relaxed;
#else
    cuda::memory_order thread_1_load = cuda::memory_order_relaxed; // default to all relaxed
    cuda::memory_order thread_2_store = cuda::memory_order_relaxed;
    cuda::memory_order thread_2_load = cuda::memory_order_relaxed;
#endif

    // defined for different distributions of threads across threadblocks
    DEFINE_IDS();

    // defined for all three thread two memory locations tests
    THREE_THREAD_TWO_MEM_LOCATIONS();

    PRE_STRESS();

    if (id_0 != id_1 && id_1 != id_2) {

      // Thread 0
      test_locations[x_0].store(1, cuda::memory_order_relaxed);

      // Thread 1
      uint r0 = test_locations[x_1].load(thread_1_load);
      uint r1 = test_locations[y_1].load(cuda::memory_order_relaxed);

      // Thread 2
      test_locations[y_2].store(1, thread_2_store);
      uint r2 = test_locations[x_2].load(thread_2_load);

      cuda::atomic_thread_fence(cuda::memory_order_seq_cst);
      read_results[wg_offset + id_1].r0 = r0;
      read_results[wg_offset + id_1].r1 = r1;
      read_results[wg_offset + id_2].r2 = r2;
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
  uint r2 = read_results[id_0].r2;

  if (r0 == 1 && r1 == 1 && r2 == 0) {
    test_results->res0.fetch_add(1);
  }
  else if (r0 == 0 && r1 == 0 && r2 == 1) {
    test_results->res1.fetch_add(1);
  }
  else if (r0 == 0 && r1 == 0 && r2 == 0) {
    test_results->res2.fetch_add(1);
  }
  else if (r0 == 1 && r1 == 1 && r2 == 1) {
    test_results->res3.fetch_add(1);
  }
  else if (r0 == 0 && r1 == 1 && r2 == 0) {
    test_results->res4.fetch_add(1);
  }
  else if (r0 == 1 && r1 == 0 && r2 == 0) {
    test_results->weak.fetch_add(1);
  }
  else {
    test_results->other.fetch_add(1);
  }
}

int host_check_results(TestResults* results, bool print) {
  if (print) {
    std::cout << "r0=1, r1=1, r2=0 (seq): " << results->res0 << "\n";
    std::cout << "r0=0, r1=0, r2=1 (seq): " << results->res1 << "\n";
    std::cout << "r0=0, r1=0, r2=0 (seq): " << results->res2 << "\n";
    std::cout << "r0=1, r1=1, r2=1 (interleaved): " << results->res3 << "\n";
    std::cout << "r0=0, r1=1, r2=0 (interleaved): " << results->res4 << "\n";
    std::cout << "r0=1, r1=0, r2=0 (weak): " << results->weak << "\n";
    std::cout << "other: " << results->other << "\n";
  }
  return results->weak;
}
