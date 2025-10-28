#include <iostream>
#include "litmus.cuh"
#include "functions.cu"

// thread mappings
// thread 0: write x, write z
// thread 1: read z, write y, read y
// thread 2: write y, write a
// thread 3: read a, read x

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

#ifdef DISALLOWED
    cuda::memory_order load_order = cuda::memory_order_acquire;
    cuda::memory_order store_order = cuda::memory_order_release;
#else
    cuda::memory_order load_order = cuda::memory_order_relaxed;
    cuda::memory_order store_order = cuda::memory_order_relaxed;
#endif

    DEFINE_IDS();

    uint permute_id_0 = permute_id(id_0, kernel_params->permute_location, total_ids);
    uint z_0 = (wg_offset + permute_id(permute_id_0, kernel_params->permute_location, total_ids)) * kernel_params->mem_stride * 3 + 2 * kernel_params->mem_offset;

    uint permute_id_1 = permute_id(id_1, kernel_params->permute_location, total_ids);
    uint z_1 = (wg_offset + permute_id(permute_id_1, kernel_params->permute_location, total_ids)) * kernel_params->mem_stride * 3 + 2 * kernel_params->mem_offset;
    uint x_1 = (wg_offset + id_1) * kernel_params->mem_stride * 3;

    uint permute_id_2 = permute_id(id_2, kernel_params->permute_location, total_ids);
    uint x_2 = (wg_offset + id_2) * kernel_params->mem_stride * 3;
    uint y_2 = (wg_offset + permute_id_2) * kernel_params->mem_stride * 3 + kernel_params->mem_offset;

    uint permute_id_3 = permute_id(id_3, kernel_params->permute_location, total_ids);
    uint y_3 = (wg_offset + permute_id_3) * kernel_params->mem_stride * 3 + kernel_params->mem_offset;
    uint z_3 = (wg_offset + permute_id(permute_id_3, kernel_params->permute_location, total_ids)) * kernel_params->mem_stride * 3 + 2 * kernel_params->mem_offset;

    cuda::atomic<uint, cuda::thread_scope_block>* z_1_ptr = (cuda::atomic<uint, cuda::thread_scope_block>*) &test_locations[z_1];
    cuda::atomic<uint, cuda::thread_scope_block>* x_1_ptr = (cuda::atomic<uint, cuda::thread_scope_block>*) &test_locations[x_1];

    PRE_STRESS();

    if (id_0 != id_1 && id_0 != id_2 && id_0 != id_3 && id_1 != id_2 && id_1 != id_3 && id_2 != id_3) {

      test_locations[z_0].store(1, cuda::memory_order_seq_cst); // write z, sys/dev scope

      uint r0 = z_1_ptr->load(cuda::memory_order_seq_cst); // read z, cta scope
      x_1_ptr->store(1, cuda::memory_order_seq_cst); // write x, cta scope

      test_locations[x_2].store(2, cuda::memory_order_seq_cst); // write x, sys/dev scope
      test_locations[y_2].store(1, cuda::memory_order_release); // write y, sys/dev scope

      uint r1 = test_locations[y_3].load(cuda::memory_order_acquire); // read y, sys/dev scope
      uint r2 = test_locations[z_3].load(cuda::memory_order_seq_cst); // read z, sys/dev scope

      cuda::atomic_thread_fence(cuda::memory_order_seq_cst);
      read_results[id_1].r0 = r0;
      read_results[id_3].r1 = r1;
      read_results[id_3].r2 = r2;
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
  uint total_ids = blockDim.x * kernel_params->testing_workgroups;
  uint id_0 = blockIdx.x * blockDim.x + threadIdx.x;
  uint x = test_locations[id_0 * kernel_params->mem_stride * 3];
  uint r0 = read_results[id_0].r0;
  uint r1 = read_results[id_0].r1;
  uint r2 = read_results[id_0].r2;
  uint x_loc = id_0 * kernel_params->mem_stride * 3;
  uint x = test_locations[x_loc];

  if (x == 0) {
    test_results->na.fetch_add(1); // thread skipped
  }
  else if (r0 == 1 && x == 2 && r1 == 1 && r2 == 0) { // weak behavior
    test_results->weak.fetch_add(1);
  }
  else {
    test_results->other.fetch_add(1);
  }
}

int host_check_results(TestResults* results, bool print) {
  if (print) {
    std::cout << "r0=1, x=2, r2=1, r3=0 (weak): " << results->weak << "\n";
    std::cout << "other: " << results->other << "\n";
    std::cout << "thread skipped: " << results->na << "\n";
  }
  return results->weak;
}

