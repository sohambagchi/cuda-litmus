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
    cuda::memory_order store_order0 = cuda::memory_order_release;
    cuda::memory_order store_order1 = cuda::memory_order_release;
    cuda::memory_order load_order = cuda::memory_order_acquire;
    #define FENCE0()
    #define FENCE1()
    #define FENCE2()
#elif defined(RELAXED)
    cuda::memory_order store_order0 = cuda::memory_order_relaxed;
    cuda::memory_order store_order1 = cuda::memory_order_relaxed;
    cuda::memory_order load_order = cuda::memory_order_relaxed;
    #define FENCE0()
    #define FENCE1()
    #define FENCE2()
#elif defined(ALL_FENCE)
    cuda::memory_order store_order0 = cuda::memory_order_relaxed;
    cuda::memory_order store_order1 = cuda::memory_order_relaxed;
    cuda::memory_order load_order = cuda::memory_order_relaxed;
    #define FENCE0() cuda::atomic_thread_fence(cuda::memory_order_acq_rel, FENCE_SCOPE);
    #define FENCE1() cuda::atomic_thread_fence(cuda::memory_order_acq_rel, FENCE_SCOPE);
    #define FENCE2() cuda::atomic_thread_fence(cuda::memory_order_acq_rel, FENCE_SCOPE);
#elif defined(FENCE_0)
    cuda::memory_order store_order0 = cuda::memory_order_relaxed;
    cuda::memory_order store_order1 = cuda::memory_order_release;
    cuda::memory_order load_order = cuda::memory_order_acquire;
    #define FENCE0() cuda::atomic_thread_fence(cuda::memory_order_acq_rel, FENCE_SCOPE);
    #define FENCE1()
    #define FENCE2()
#elif defined(FENCE_1)
    cuda::memory_order store_order0 = cuda::memory_order_release;
    cuda::memory_order store_order1 = cuda::memory_order_relaxed;
    cuda::memory_order load_order = cuda::memory_order_acquire;
    #define FENCE0()
    #define FENCE1() cuda::atomic_thread_fence(cuda::memory_order_acq_rel, FENCE_SCOPE);
    #define FENCE2()
#elif defined(FENCE_2)
    cuda::memory_order store_order0 = cuda::memory_order_release;
    cuda::memory_order store_order1 = cuda::memory_order_release;
    cuda::memory_order load_order = cuda::memory_order_relaxed;
    #define FENCE0()
    #define FENCE1()
    #define FENCE2() cuda::atomic_thread_fence(cuda::memory_order_acq_rel, FENCE_SCOPE);
#elif defined(FENCE_01)
    cuda::memory_order store_order0 = cuda::memory_order_relaxed;
    cuda::memory_order store_order1 = cuda::memory_order_relaxed;
    cuda::memory_order load_order = cuda::memory_order_acquire;
    #define FENCE0() cuda::atomic_thread_fence(cuda::memory_order_acq_rel, FENCE_SCOPE);
    #define FENCE1() cuda::atomic_thread_fence(cuda::memory_order_acq_rel, FENCE_SCOPE);
    #define FENCE2()
#elif defined(FENCE_02)
    cuda::memory_order store_order0 = cuda::memory_order_relaxed;
    cuda::memory_order store_order1 = cuda::memory_order_release;
    cuda::memory_order load_order = cuda::memory_order_relaxed;
    #define FENCE0() cuda::atomic_thread_fence(cuda::memory_order_acq_rel, FENCE_SCOPE);
    #define FENCE1()
    #define FENCE2() cuda::atomic_thread_fence(cuda::memory_order_acq_rel, FENCE_SCOPE);
#elif defined(FENCE_12)
    cuda::memory_order store_order0 = cuda::memory_order_release;
    cuda::memory_order store_order1 = cuda::memory_order_relaxed;
    cuda::memory_order load_order = cuda::memory_order_relaxed;
    #define FENCE0()
    #define FENCE1() cuda::atomic_thread_fence(cuda::memory_order_acq_rel, FENCE_SCOPE);
    #define FENCE2() cuda::atomic_thread_fence(cuda::memory_order_acq_rel, FENCE_SCOPE);
#else
    cuda::memory_order store_order0 = cuda::memory_order_relaxed;
    cuda::memory_order store_order1 = cuda::memory_order_relaxed;
    cuda::memory_order load_order = cuda::memory_order_relaxed;
    #define FENCE0()
    #define FENCE1()
    #define FENCE2()
#endif

    // defined for different distributions of threads across threadblocks
    DEFINE_IDS();

    // defined for all three thread three memory locations tests
    THREE_THREAD_THREE_MEM_LOCATIONS();

    PRE_STRESS();

    if (id_0 != id_1 && id_1 != id_2 && id_0 != id_2) {

      // Thread 0
      test_locations[x_0].store(1, cuda::memory_order_relaxed);
      FENCE0()
      test_locations[y_0].store(1, store_order0);

      // Thread 1
      test_locations[y_1].store(2, cuda::memory_order_relaxed);
      FENCE1()
      test_locations[z_1].store(1, store_order1);

      // Thread 2
      uint r0 = test_locations[z_2].load(load_order);
      FENCE2()
      uint r1 = test_locations[x_2].load(cuda::memory_order_relaxed);

      cuda::atomic_thread_fence(cuda::memory_order_seq_cst);
      read_results[wg_offset + id_2].r0 = r0;
      read_results[wg_offset + id_2].r1 = r1;
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
  uint r0 = read_results[id_0].r0;
  uint r1 = read_results[id_0].r1;
  uint x = test_locations[id_0 * kernel_params->mem_stride * 3];
  uint y_loc = (wg_offset + permute_id(id_0, kernel_params->permute_location, total_ids)) * kernel_params->mem_stride * 3 + kernel_params->mem_offset;
  uint y = test_locations[y_loc];

  if (x == 0) {
    test_results->na.fetch_add(1); // thread skipped
  }
  else if (y == 2 && r0 == 1 && r1 == 0) {
    test_results->weak.fetch_add(1);
    weak[id_0] = true;
  }
  else {
    test_results->other.fetch_add(1);
  }
}

int host_check_results(TestResults* results, bool print) {
  if (print) {
    std::cout << "r0=1, r1=0, y=2 (weak): " << results->weak << "\n";
    std::cout << "thread skipped: " << results->na << "\n";
    std::cout << "other: " << results->other << "\n";
  }
  return results->weak;
}

