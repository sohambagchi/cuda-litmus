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

#ifdef ACQUIRE
    cuda::memory_order thread_1_load = cuda::memory_order_acquire;
    cuda::memory_order thread_1_store = cuda::memory_order_relaxed;
    cuda::memory_order thread_2_store = cuda::memory_order_release;
    #define FENCE_1()
    #define FENCE_2()
#elif defined(RELEASE)
    cuda::memory_order thread_1_load = cuda::memory_order_relaxed;
    cuda::memory_order thread_1_store = cuda::memory_order_release;
    cuda::memory_order thread_2_store = cuda::memory_order_release;
    #define FENCE_1()
    #define FENCE_2()
#elif defined(RELAXED)
    cuda::memory_order thread_1_load = cuda::memory_order_relaxed;
    cuda::memory_order thread_1_store = cuda::memory_order_relaxed;
    cuda::memory_order thread_2_store = cuda::memory_order_relaxed;
    #define FENCE_1()
    #define FENCE_2()
#elif defined(BOTH_FENCE)
    cuda::memory_order thread_1_load = cuda::memory_order_relaxed;
    cuda::memory_order thread_1_store = cuda::memory_order_relaxed;
    cuda::memory_order thread_2_store = cuda::memory_order_relaxed;
    #define FENCE_1() cuda::atomic_thread_fence(cuda::memory_order_acq_rel, FENCE_SCOPE);
    #define FENCE_2() cuda::atomic_thread_fence(cuda::memory_order_acq_rel, FENCE_SCOPE);
#elif defined(THREAD_1_FENCE)
    cuda::memory_order thread_1_load = cuda::memory_order_relaxed;
    cuda::memory_order thread_1_store = cuda::memory_order_relaxed;
    cuda::memory_order thread_2_store = cuda::memory_order_release;
    #define FENCE_1() cuda::atomic_thread_fence(cuda::memory_order_acq_rel, FENCE_SCOPE);
    #define FENCE_2()
#elif defined(THREAD_2_FENCE_ACQ)
    cuda::memory_order thread_1_load = cuda::memory_order_acquire;
    cuda::memory_order thread_1_store = cuda::memory_order_relaxed;
    cuda::memory_order thread_2_store = cuda::memory_order_relaxed;
    #define FENCE_1()
    #define FENCE_2() cuda::atomic_thread_fence(cuda::memory_order_acq_rel, FENCE_SCOPE);
#elif defined(THREAD_2_FENCE_REL)
    cuda::memory_order thread_1_load = cuda::memory_order_relaxed;
    cuda::memory_order thread_1_store = cuda::memory_order_release;
    cuda::memory_order thread_2_store = cuda::memory_order_relaxed;
    #define FENCE_1()
    #define FENCE_2() cuda::atomic_thread_fence(cuda::memory_order_acq_rel, FENCE_SCOPE);

#else
    cuda::memory_order thread_1_load = cuda::memory_order_relaxed; // default to all relaxed
    cuda::memory_order thread_1_store = cuda::memory_order_relaxed;
    cuda::memory_order thread_2_store = cuda::memory_order_relaxed;
    #define FENCE_1()
    #define FENCE_2()
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
      FENCE_1()
      test_locations[y_1].store(1, thread_1_store);

      // Thread 2
      test_locations[y_2].store(2, cuda::memory_order_relaxed);
      FENCE_2()
      test_locations[x_2].store(1, thread_2_store);

      cuda::atomic_thread_fence(cuda::memory_order_seq_cst);
      read_results[wg_offset + id_1].r0 = r0;
    }
  }

  MEM_STRESS();
}

__global__ void check_results(
  d_atomic_uint* test_locations,
  ReadResults* read_results,
  TestResults* test_results,
  KernelParams* kernel_params) {
  RESULT_IDS();
  uint id_0 = blockIdx.x * blockDim.x + threadIdx.x;
  uint r0 = read_results[id_0].r0;
  uint x = test_locations[id_0 * kernel_params->mem_stride * 2];
  uint y_loc = (wg_offset + permute_id(id_0, kernel_params->permute_location, total_ids)) * kernel_params->mem_stride * 2 + kernel_params->mem_offset;
  uint y = test_locations[y_loc];

  if (r0 == 0 && x == 1 && y == 1) { 
    test_results->res0.fetch_add(1);
  }
  else if (r0 == 0 && x == 1 && y == 2) { 
    test_results->res1.fetch_add(1);
  }
  else if (r0 == 0 && x == 2 && y == 1) { 
    test_results->res2.fetch_add(1);
  }
  else if (r0 == 0 && x == 2 && y == 2) { 
    test_results->res3.fetch_add(1);
  }

  else if (r0 == 1 && x == 1 && y == 1) { 
    test_results->res4.fetch_add(1);
  }
  else if (r0 == 1 && x == 1 && y == 2) { 
    test_results->res5.fetch_add(1);
  }
  else if (r0 == 1 && x == 2 && y == 1) { 
    test_results->res6.fetch_add(1);
  }
  else if (r0 == 1 && x == 2 && y == 2) { 
    test_results->res7.fetch_add(1);
  }

  else if (r0 == 2 && x == 1 && y == 1) { 
    test_results->res8.fetch_add(1);
  }
  else if (r0 == 2 && x == 1 && y == 2) { 
    test_results->res9.fetch_add(1);
  }
  else if (r0 == 2 && x == 2 && y == 1) { 
    test_results->res10.fetch_add(1);
  }
  else if (r0 == 2 && x == 2 && y == 2) { // this is the non-mca weak behavior
    test_results->weak.fetch_add(1);
  }

  else if (x == 0 && y == 0) {
    test_results->na.fetch_add(1);
  }
  else {
    test_results->other.fetch_add(1);
  }
}

int host_check_results(TestResults* results, bool print) {
  if (print) {
    std::cout << "r0=0, x=1, y=1: " << results->res0 << "\n";
    std::cout << "r0=0, x=1, y=2: " << results->res1 << "\n";
    std::cout << "r0=0, x=2, y=1: " << results->res2 << "\n";
    std::cout << "r0=0, x=2, y=2: " << results->res3 << "\n";
    std::cout << "r0=1, x=1, y=1: " << results->res4 << "\n";
    std::cout << "r0=1, x=1, y=2: " << results->res5 << "\n";
    std::cout << "r0=1, x=2, y=1: " << results->res6 << "\n";
    std::cout << "r0=0, x=2, y=2: " << results->res7 << "\n";
    std::cout << "r0=2, x=1, y=1: " << results->res8 << "\n";
    std::cout << "r0=2, x=1, y=2: " << results->res9 << "\n";
    std::cout << "r0=2, x=2, y=1: " << results->res10 << "\n";
    std::cout << "r0=2, x=2, y=2 (weak): " << results->weak << "\n";
    std::cout << "thread skipped: " << results->na << "\n";
    std::cout << "other: " << results->other << "\n";
  }
  return results->weak;
}
