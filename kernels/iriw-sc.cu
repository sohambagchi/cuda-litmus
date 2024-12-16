#include <iostream>
#include "litmus.cuh"
#include "functions.cu"

// iriw thread mappings
// thread 0: write x
// thread 1: read x, read y
// thread 2: write y
// thread 3: read y, read x

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

#ifdef STORES_SC
    cuda::memory_order thread_0 = cuda::memory_order_seq_cst;
    cuda::memory_order thread_1_a = cuda::memory_order_acquire;
    cuda::memory_order thread_1_b = cuda::memory_order_relaxed;
    cuda::memory_order thread_2 = cuda::memory_order_seq_cst;
    cuda::memory_order thread_3_a = cuda::memory_order_acquire;
    cuda::memory_order thread_3_b = cuda::memory_order_relaxed;
#elif defined(FIRST_LOAD_SC)
    cuda::memory_order thread_0 = cuda::memory_order_relaxed;
    cuda::memory_order thread_1_a = cuda::memory_order_seq_cst;
    cuda::memory_order thread_1_b = cuda::memory_order_relaxed;
    cuda::memory_order thread_2 = cuda::memory_order_relaxed;
    cuda::memory_order thread_3_a = cuda::memory_order_seq_cst;
    cuda::memory_order thread_3_b = cuda::memory_order_relaxed;
#elif defined(SECOND_LOAD_SC)
    cuda::memory_order thread_0 = cuda::memory_order_relaxed;
    cuda::memory_order thread_1_a = cuda::memory_order_relaxed;
    cuda::memory_order thread_1_b = cuda::memory_order_seq_cst;
    cuda::memory_order thread_2 = cuda::memory_order_relaxed;
    cuda::memory_order thread_3_a = cuda::memory_order_relaxed;
    cuda::memory_order thread_3_b = cuda::memory_order_seq_cst;
#elif defined(ALL_SC)
    cuda::memory_order thread_0 = cuda::memory_order_seq_cst;
    cuda::memory_order thread_1_a = cuda::memory_order_seq_cst;
    cuda::memory_order thread_1_b = cuda::memory_order_seq_cst;
    cuda::memory_order thread_2 = cuda::memory_order_seq_cst;
    cuda::memory_order thread_3_a = cuda::memory_order_seq_cst;
    cuda::memory_order thread_3_b = cuda::memory_order_seq_cst;
#elif defined(RELEASE_WRITES)
    cuda::memory_order thread_0 = cuda::memory_order_release;
    cuda::memory_order thread_1_a = cuda::memory_order_acquire;
    cuda::memory_order thread_1_b = cuda::memory_order_relaxed;
    cuda::memory_order thread_2 = cuda::memory_order_release;
    cuda::memory_order thread_3_a = cuda::memory_order_acquire;
    cuda::memory_order thread_3_b = cuda::memory_order_relaxed;
#elif defined(ALL_ACQUIRE)
    cuda::memory_order thread_0 = cuda::memory_order_relaxed;
    cuda::memory_order thread_1_a = cuda::memory_order_acquire;
    cuda::memory_order thread_1_b = cuda::memory_order_acquire;
    cuda::memory_order thread_2 = cuda::memory_order_relaxed;
    cuda::memory_order thread_3_a = cuda::memory_order_acquire;
    cuda::memory_order thread_3_b = cuda::memory_order_acquire;
#else
    cuda::memory_order thread_0 = cuda::memory_order_relaxed;
    cuda::memory_order thread_1_a = cuda::memory_order_acquire;
    cuda::memory_order thread_1_b = cuda::memory_order_relaxed;
    cuda::memory_order thread_2 = cuda::memory_order_relaxed;
    cuda::memory_order thread_3_a = cuda::memory_order_acquire;
    cuda::memory_order thread_3_b = cuda::memory_order_relaxed;
#endif


    // defined for different distributions of threads across threadblocks
    DEFINE_IDS();

    uint x_0 = (wg_offset + id_0) * kernel_params->mem_stride * 2;
    uint x_1 = (wg_offset + id_1) * kernel_params->mem_stride * 2;
    uint y_1 = (wg_offset + permute_id(id_1, kernel_params->permute_location, total_ids)) * kernel_params->mem_stride * 2 + kernel_params->mem_offset;
    uint y_2 = (wg_offset + permute_id(id_2, kernel_params->permute_location, total_ids)) * kernel_params->mem_stride * 2 + kernel_params->mem_offset;
    uint x_3 = (wg_offset + id_3) * kernel_params->mem_stride * 2;
    uint y_3 = (wg_offset + permute_id(id_3, kernel_params->permute_location, total_ids)) * kernel_params->mem_stride * 2 + kernel_params->mem_offset;

    PRE_STRESS();

    if (id_0 != id_1 && id_0 != id_2 && id_0 != id_3 && id_1 != id_2 && id_1 != id_3 && id_2 != id_3) {
      
      test_locations[x_0].store(1, thread_0); // write x

      uint r0 = test_locations[x_1].load(thread_1_a); // read x
      uint r1 = test_locations[y_1].load(thread_1_b); // read y

      test_locations[y_2].store(1, thread_2); // write y

      uint r2 = test_locations[y_3].load(thread_3_a); // read y
      uint r3 = test_locations[x_3].load(thread_3_b); // read x

      cuda::atomic_thread_fence(cuda::memory_order_seq_cst);
      read_results[wg_offset + id_1].r0 = r0;
      read_results[wg_offset + id_1].r1 = r1;
      read_results[wg_offset + id_3].r2 = r2;
      read_results[wg_offset + id_3].r3 = r3;
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

