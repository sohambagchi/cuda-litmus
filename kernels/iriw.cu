#include <iostream>
#include "litmus.cuh"
#include "functions.cuh"

#ifdef TB_0_1_2_3
#include "tb-0-1-2-3.h"
#elif defined(TB_01_23)
#include "tb-01-23.h"
#elif defined(TB_0123)
#include "tb-0123.h"
#else
#include "tb-0-1-2-3.h" // default to all different threadblocks
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
    cuda::memory_order first_mem_order = cuda::memory_order_acquire;
    #define FENCE()
#elif defined(ACQ_FENCE_BLOCK)
    cuda::memory_order first_mem_order = cuda::memory_order_relaxed;
    #define FENCE() cuda::atomic_thread_fence(cuda::memory_order_acq_rel, cuda::thread_scope_block);
#elif defined(ACQ_FENCE_DEVICE)
    cuda::memory_order first_mem_order = cuda::memory_order_relaxed;
    #define FENCE() cuda::atomic_thread_fence(cuda::memory_order_acq_rel, cuda::thread_scope_device);
#elif defined(ACQ_FENCE_SYSTEM)
    cuda::memory_order first_mem_order = cuda::memory_order_relaxed;
    #define FENCE() cuda::atomic_thread_fence(cuda::memory_order_acq_rel, cuda::thread_scope_system);
#elif defined(RELAXED)
    cuda::memory_order first_mem_order = cuda::memory_order_relaxed;
    #define FENCE()
#else
    cuda::memory_order first_mem_order = cuda::memory_order_relaxed; // default to relaxed
    #define FENCE()
#endif

    // defined for different distributions of threads across threadblocks
    DEFINE_IDS();

    uint mem_0;
    if (id_0_first_half) {
      mem_0 = (wg_offset + id_0_final) * kernel_params->mem_stride * 2;
    }
    else {
      mem_0 = (wg_offset + permute_id(id_0_final, kernel_params->permute_location, total_ids)) * kernel_params->mem_stride * 2 + kernel_params->mem_offset;
    }
    uint x_1 = (wg_offset + id_1_final) * kernel_params->mem_stride * 2;
    uint y_1 = (wg_offset + permute_id(id_1_final, kernel_params->permute_location, total_ids)) * kernel_params->mem_stride * 2 + kernel_params->mem_offset;

    PRE_STRESS();

    if (id_0_final != id_1_final) {

      test_locations[mem_0].store(1, cuda::memory_order_relaxed); // write to either x or y depending on thread

      if (id_1_first_half) { // one observer thread reads x then y
        uint r0 = test_locations[x_1].load(first_mem_order);
        FENCE()
        uint r1 = test_locations[y_1].load(cuda::memory_order_relaxed);
        cuda::atomic_thread_fence(cuda::memory_order_seq_cst);
        read_results[wg_offset + id_1_final].r0 = r0;
        read_results[wg_offset + id_1_final].r1 = r1;
      }
      else { // other observer thread reads y then x
        uint r2 = test_locations[y_1].load(first_mem_order);
        FENCE()
        uint r3 = test_locations[x_1].load(cuda::memory_order_relaxed);
        cuda::atomic_thread_fence(cuda::memory_order_seq_cst);
        read_results[wg_offset + id_1_final].r2 = r2;
        read_results[wg_offset + id_1_final].r3 = r3;
      }
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
  if (id_0 < (blockDim.x * kernel_params->testing_workgroups) / 2) {
    uint x = test_locations[id_0 * kernel_params->mem_stride * 2];
    uint r0 = read_results[id_0].r0;
    uint r1 = read_results[id_0].r1;
    uint r2 = read_results[id_0].r2;
    uint r3 = read_results[id_0].r3;

    if (x == 0) {
      test_results->na.fetch_add(1); // thread skipped
    }
    else if (r0 == 0 && r1 == 0 && r2 == 0 && r3 == 0) { // both observers run first
      test_results->res0.fetch_add(1);
    }
    else if (r0 == 1 && r1 == 1 && r2 == 1 && r3 == 1) { // both observers run last
      test_results->res1.fetch_add(1);
    }
    else if (r0 == 0 && r1 == 0 && r2 == 1 && r3 == 1) { // first observer runs first
      test_results->res2.fetch_add(1);
    }
    else if (r0 == 1 && r1 == 1 && r2 == 0 && r3 == 0) { // second observer runs first
      test_results->res3.fetch_add(1);
    }
    else if (r0 == r1 && r2 != r3) { // second observer interleaved
      test_results->res4.fetch_add(1);
    }
    else if (r0 != r1 && r2 == r3) { // first observer interleaved
      test_results->res5.fetch_add(1);
    }
    else if (r0 == 0 && r1 == 1 && r2 == 0 && r3 == 1) { // both interleaved
      test_results->res6.fetch_add(1);
    }
    else if (r0 == 0 && r1 == 1 && r2 == 1 && r3 == 0) { // both interleaved
      test_results->res7.fetch_add(1);
    }
    else if (r0 == 1 && r1 == 0 && r2 == 0 && r3 == 1) { // both interleaved
      test_results->res8.fetch_add(1);
    }
    else if (r0 == 1 && r1 == 0 && r2 == 1 && r3 == 0) { // observer threads see x/y in different orders
      test_results->weak.fetch_add(1);
    }
    else {
      test_results->other.fetch_add(1);
    }
  }
}

int host_check_results(TestResults* results, bool print) {
  if (print) {
    std::cout << "r0=0, r1=0, r2=0, r3=0 (seq): " << results->res0 << "\n";
    std::cout << "r0=1, r1=1, r2=1, r3=1 (seq): " << results->res1 << "\n";
    std::cout << "r0=0, r1=0, r2=1, r3=1 (seq): " << results->res2 << "\n";
    std::cout << "r0=1, r1=1, r2=0, r3=0 (seq): " << results->res3 << "\n";
    std::cout << "r0 == r1, r2 != r3 (seq/interleaved): " << results->res4 << "\n";
    std::cout << "r0 != r1, r2 == r3 (interleaved/seq): " << results->res5 << "\n";
    std::cout << "r0=0, r1=1, r2=0, r3=1 (interleaved): " << results->res6 << "\n";
    std::cout << "r0=0, r1=1, r2=1, r3=0 (interleaved): " << results->res7 << "\n";
    std::cout << "r0=1, r1=0, r2=0, r3=1 (interleaved): " << results->res8 << "\n";
    std::cout << "r0=1, r1=0, r2=1, r3=0 (weak): " << results->weak << "\n";
    std::cout << "thread skipped: " << results->na << "\n";
    std::cout << "other: " << results->other << "\n";
  }
  return results->weak;
}

