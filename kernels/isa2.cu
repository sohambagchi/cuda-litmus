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
    cuda::memory_order thread_0_store = cuda::memory_order_release;
    cuda::memory_order thread_1_load = cuda::memory_order_acquire;
    cuda::memory_order thread_1_store = cuda::memory_order_relaxed;
    cuda::memory_order thread_2_load = cuda::memory_order_acquire;
    #define THREAD_0_FENCE()
    #define THREAD_1_FENCE()
    #define THREAD_2_FENCE()
#elif defined(RELEASE)
    cuda::memory_order thread_0_store = cuda::memory_order_release;
    cuda::memory_order thread_1_load = cuda::memory_order_relaxed;
    cuda::memory_order thread_1_store = cuda::memory_order_release;
    cuda::memory_order thread_2_load = cuda::memory_order_acquire;
    #define THREAD_0_FENCE()
    #define THREAD_1_FENCE()
    #define THREAD_2_FENCE()
#elif defined(RELAXED)
    cuda::memory_order thread_0_store = cuda::memory_order_relaxed;
    cuda::memory_order thread_1_load = cuda::memory_order_relaxed;
    cuda::memory_order thread_1_store = cuda::memory_order_relaxed;
    cuda::memory_order thread_2_load = cuda::memory_order_relaxed;
    #define THREAD_0_FENCE()
    #define THREAD_1_FENCE()
    #define THREAD_2_FENCE()
#elif defined(ALL_FENCE)
    cuda::memory_order thread_0_store = cuda::memory_order_relaxed;
    cuda::memory_order thread_1_load = cuda::memory_order_relaxed;
    cuda::memory_order thread_1_store = cuda::memory_order_relaxed;
    cuda::memory_order thread_2_load = cuda::memory_order_relaxed;
    #define THREAD_0_FENCE() cuda::atomic_thread_fence(cuda::memory_order_acq_rel, fence_scope);
    #define THREAD_1_FENCE() cuda::atomic_thread_fence(cuda::memory_order_acq_rel, fence_scope);
    #define THREAD_2_FENCE() cuda::atomic_thread_fence(cuda::memory_order_acq_rel, fence_scope);
#elif defined(THREAD_0_FENCE_ACQ)
    cuda::memory_order thread_0_store = cuda::memory_order_relaxed;
    cuda::memory_order thread_1_load = cuda::memory_order_acquire;
    cuda::memory_order thread_1_store = cuda::memory_order_relaxed;
    cuda::memory_order thread_2_load = cuda::memory_order_acquire;
    #define THREAD_0_FENCE() cuda::atomic_thread_fence(cuda::memory_order_acq_rel, fence_scope);
    #define THREAD_1_FENCE()
    #define THREAD_2_FENCE()
#elif defined(THREAD_0_FENCE_REL)
    cuda::memory_order thread_0_store = cuda::memory_order_relaxed;
    cuda::memory_order thread_1_load = cuda::memory_order_relaxed;
    cuda::memory_order thread_1_store = cuda::memory_order_release;
    cuda::memory_order thread_2_load = cuda::memory_order_acquire;
    #define THREAD_0_FENCE() cuda::atomic_thread_fence(cuda::memory_order_acq_rel, fence_scope);
    #define THREAD_1_FENCE()
    #define THREAD_2_FENCE()
#elif defined(THREAD_1_FENCE)
    cuda::memory_order thread_0_store = cuda::memory_order_release;
    cuda::memory_order thread_1_load = cuda::memory_order_relaxed;
    cuda::memory_order thread_1_store = cuda::memory_order_relaxed;
    cuda::memory_order thread_2_load = cuda::memory_order_acquire;
    #define THREAD_0_FENCE() 
    #define THREAD_1_FENCE() cuda::atomic_thread_fence(cuda::memory_order_acq_rel, fence_scope);
    #define THREAD_2_FENCE()
#elif defined(THREAD_2_FENCE_ACQ)
    cuda::memory_order thread_0_store = cuda::memory_order_release;
    cuda::memory_order thread_1_load = cuda::memory_order_acquire;
    cuda::memory_order thread_1_store = cuda::memory_order_relaxed;
    cuda::memory_order thread_2_load = cuda::memory_order_relaxed;
    #define THREAD_0_FENCE() 
    #define THREAD_1_FENCE()
    #define THREAD_2_FENCE() cuda::atomic_thread_fence(cuda::memory_order_acq_rel, fence_scope);
#elif defined(THREAD_2_FENCE_REL)
    cuda::memory_order thread_0_store = cuda::memory_order_release;
    cuda::memory_order thread_1_load = cuda::memory_order_relaxed;
    cuda::memory_order thread_1_store = cuda::memory_order_release;
    cuda::memory_order thread_2_load = cuda::memory_order_relaxed;
    #define THREAD_0_FENCE() 
    #define THREAD_1_FENCE()
    #define THREAD_2_FENCE() cuda::atomic_thread_fence(cuda::memory_order_acq_rel, fence_scope);
#elif defined(THREAD_0_1_FENCE)
    cuda::memory_order thread_0_store = cuda::memory_order_relaxed;
    cuda::memory_order thread_1_load = cuda::memory_order_relaxed;
    cuda::memory_order thread_1_store = cuda::memory_order_relaxed;
    cuda::memory_order thread_2_load = cuda::memory_order_acquire;
    #define THREAD_0_FENCE() cuda::atomic_thread_fence(cuda::memory_order_acq_rel, fence_scope);
    #define THREAD_1_FENCE() cuda::atomic_thread_fence(cuda::memory_order_acq_rel, fence_scope);
    #define THREAD_2_FENCE() 
#elif defined(THREAD_0_2_FENCE_ACQ)
    cuda::memory_order thread_0_store = cuda::memory_order_relaxed;
    cuda::memory_order thread_1_load = cuda::memory_order_acquire;
    cuda::memory_order thread_1_store = cuda::memory_order_relaxed;
    cuda::memory_order thread_2_load = cuda::memory_order_relaxed;
    #define THREAD_0_FENCE() cuda::atomic_thread_fence(cuda::memory_order_acq_rel, fence_scope);
    #define THREAD_1_FENCE() 
    #define THREAD_2_FENCE() cuda::atomic_thread_fence(cuda::memory_order_acq_rel, fence_scope);
#elif defined(THREAD_0_2_FENCE_REL)
    cuda::memory_order thread_0_store = cuda::memory_order_relaxed;
    cuda::memory_order thread_1_load = cuda::memory_order_relaxed;
    cuda::memory_order thread_1_store = cuda::memory_order_release;
    cuda::memory_order thread_2_load = cuda::memory_order_relaxed;
    #define THREAD_0_FENCE() cuda::atomic_thread_fence(cuda::memory_order_acq_rel, fence_scope);
    #define THREAD_1_FENCE() 
    #define THREAD_2_FENCE() cuda::atomic_thread_fence(cuda::memory_order_acq_rel, fence_scope);
#elif defined(THREAD_1_2_FENCE)
    cuda::memory_order thread_0_store = cuda::memory_order_release;
    cuda::memory_order thread_1_load = cuda::memory_order_relaxed;
    cuda::memory_order thread_1_store = cuda::memory_order_relaxed;
    cuda::memory_order thread_2_load = cuda::memory_order_relaxed;
    #define THREAD_0_FENCE() 
    #define THREAD_1_FENCE() cuda::atomic_thread_fence(cuda::memory_order_acq_rel, fence_scope);
    #define THREAD_2_FENCE() cuda::atomic_thread_fence(cuda::memory_order_acq_rel, fence_scope);
#else
    cuda::memory_order thread_0_store = cuda::memory_order_relaxed; // default to all relaxed
    cuda::memory_order thread_1_load = cuda::memory_order_relaxed;
    cuda::memory_order thread_1_store = cuda::memory_order_relaxed;
    cuda::memory_order thread_2_load = cuda::memory_order_relaxed;
    #define THREAD_0_FENCE()
    #define THREAD_1_FENCE()
    #define THREAD_2_FENCE()
#endif

    // defined for different distributions of threads across threadblocks
    DEFINE_IDS();

    uint x_0 = (wg_offset + id_0) * kernel_params->mem_stride * 3;
    uint y_0 = (wg_offset + permute_id(id_0, kernel_params->permute_location, total_ids)) * kernel_params->mem_stride * 3 + kernel_params->mem_offset;
    uint permute_id_1 = permute_id(id_1, kernel_params->permute_location, total_ids);
    uint y_1 = (wg_offset + permute_id_1) * kernel_params->mem_stride * 3 + kernel_params->mem_offset;
    uint z_1 = (wg_offset + permute_id(permute_id_1, kernel_params->permute_location, total_ids)) * kernel_params->mem_stride * 3 + 2 * kernel_params->mem_offset;
    uint x_2 = (wg_offset + id_2) * kernel_params->mem_stride * 3;
    uint permute_id_2 = permute_id(id_2, kernel_params->permute_location, total_ids);
    uint z_2 = (wg_offset + permute_id(permute_id_2, kernel_params->permute_location, total_ids)) * kernel_params->mem_stride * 3 + 2 * kernel_params->mem_offset;

    if (kernel_params->pre_stress) {
      do_stress(scratchpad, scratch_locations, kernel_params->pre_stress_iterations, kernel_params->pre_stress_pattern);
    }
    if (kernel_params->barrier) {
      spin(barrier, blockDim.x * kernel_params->testing_workgroups);
    }

    if (id_0 != id_1 && id_1 != id_2) {

      // Thread 0
      test_locations[x_0].store(1, cuda::memory_order_relaxed);
      THREAD_0_FENCE()
      test_locations[y_0].store(1, thread_0_store);

      // Thread 1
      uint r0 = test_locations[y_1].load(thread_1_load);
      THREAD_1_FENCE()
      test_locations[z_1].store(1, thread_1_store);

      // Thread 2
      uint r1 = test_locations[z_2].load(thread_2_load);
      THREAD_2_FENCE()
      uint r2 = test_locations[x_2].load(cuda::memory_order_relaxed);

      cuda::atomic_thread_fence(cuda::memory_order_seq_cst);
      read_results[wg_offset + id_1].r0 = r0;
      read_results[wg_offset + id_2].r1 = r1;
      read_results[wg_offset + id_2].r2 = r2;
    }
  }
  else if (kernel_params->mem_stress) {
    do_stress(scratchpad, scratch_locations, kernel_params->mem_stress_iterations, kernel_params->pre_stress_iterations);
  }
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
  uint x = test_locations[id_0 * kernel_params->mem_stride * 3];

  if (x == 0) {
    test_results->na.fetch_add(1); // thread skipped
  }
  else if (r0 == 1 && r1 == 1 && r2 == 1) {
    test_results->res0.fetch_add(1);
  }
  else if (r0 == 0 && r1 == 0 && r2 == 0) {
    test_results->res1.fetch_add(1);
  }
  else if (r0 == 0 && r1 == 0 && r2 == 1) {
    test_results->res2.fetch_add(1);
  }
  else if (r0 == 0 && r1 == 1 && r2 == 0) {
    test_results->res3.fetch_add(1);
  }
  else if (r0 == 0 && r1 == 1 && r2 == 1) {
    test_results->res4.fetch_add(1);
  }

  else if (r0 == 1 && r1 == 0 && r2 == 0) {
    test_results->res5.fetch_add(1);
  }
  else if (r0 == 1 && r1 == 0 && r2 == 1) {
    test_results->res6.fetch_add(1);
  }
  else if (r0 == 1 && r1 == 1 && r2 == 0) {
    test_results->weak.fetch_add(1);
  }
  else {
    test_results->other.fetch_add(1);
  }
}

int host_check_results(TestResults* results, bool print) {
  if (print) {
    std::cout << "r0=0, r1=1, r2=1 (seq): " << results->res0 << "\n";
    std::cout << "r0=0, r1=0, r2=0 (seq): " << results->res1 << "\n";
    std::cout << "r0=0, r1=0, r2=1 (seq): " << results->res2 << "\n";
    std::cout << "r0=0, r1=1, r2=0: " << results->res3 << "\n";
    std::cout << "r0=0, r1=1, r2=1: " << results->res4 << "\n";
    std::cout << "r0=1, r1=0, r2=0 (seq): " << results->res5 << "\n";
    std::cout << "r0=1, r1=0, r2=1 (interleaved): " << results->res6 << "\n";
    std::cout << "r0=1, r1=1, r2=0 (weak): " << results->weak << "\n";
    std::cout << "thread skipped: " << results->na << "\n";
    std::cout << "other: " << results->other << "\n";
  }
  return results->weak;
}

