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

    uint x_0 = (wg_offset + id_0) * kernel_params->mem_stride * 4;
    uint y_0 = (wg_offset + permute_id(id_0, kernel_params->permute_location, total_ids)) * kernel_params->mem_stride * 4 + kernel_params->mem_offset;

    uint permute_id_1 = permute_id(id_1, kernel_params->permute_location, total_ids);
    uint y_1 = (wg_offset + permute_id_1) * kernel_params->mem_stride * 4 + kernel_params->mem_offset;
    uint z_1 = (wg_offset + permute_id(permute_id_1, kernel_params->permute_location, total_ids)) * kernel_params->mem_stride * 4 + 2 * kernel_params->mem_offset;

    uint permute_id_2 = permute_id(id_2, kernel_params->permute_location, total_ids);
    uint permute_permute_id_2 = permute_id(permute_id_2, kernel_params->permute_location, total_ids);
    uint z_2 = (wg_offset + permute_permute_id_2) * kernel_params->mem_stride * 4 + 2 * kernel_params->mem_offset;
    uint a_2 = (wg_offset + permute_id(permute_permute_id_2, kernel_params->permute_location, total_ids)) * kernel_params->mem_stride * 4 + 3 * kernel_params->mem_offset;

    uint permute_id_3 = permute_id(id_3, kernel_params->permute_location, total_ids);
    uint permute_permute_id_3 = permute_id(permute_id_3, kernel_params->permute_location, total_ids);
    uint a_3 = (wg_offset + permute_id(permute_permute_id_3, kernel_params->permute_location, total_ids)) * kernel_params->mem_stride * 4 + 3 * kernel_params->mem_offset;
    uint x_3 = (wg_offset + id_3) * kernel_params->mem_stride * 4;

    PRE_STRESS();

    if (id_0 != id_1 && id_0 != id_2 && id_0 != id_3 && id_1 != id_2 && id_1 != id_3 && id_2 != id_3) {

      test_locations[x_0].store(1, store_order); // write x
      test_locations[y_0].store(1, store_order); // write y

      uint r0 = test_locations[y_1].load(load_order); // read y
      test_locations[z_1].store(1, store_order); // write z

      test_locations[z_2].store(2, store_order); // write z
      test_locations[a_2].store(1, store_order); // write a

      uint r1 = test_locations[a_3].load(load_order); // read a
      uint r2 = test_locations[x_3].load(load_order); // read x

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
  uint id_0 = blockIdx.x * blockDim.x + threadIdx.x;
  uint x = test_locations[id_0 * kernel_params->mem_stride * 4];
  uint r0 = read_results[id_0].r0;
  uint r1 = read_results[id_0].r1;
  uint r2 = read_results[id_0].r2;
  uint z = test_locations[id_0 * kernel_params->mem_stride * 4 + 2 * kernel_params->mem_offset];

  if (x == 0) {
    test_results->na.fetch_add(1); // thread skipped
  }
  else if (r0 == 1 && z == 2 && r1 == 1 && r2 == 0) { // weak behavior
    test_results->weak.fetch_add(1);
  }
  else {
    test_results->other.fetch_add(1);
  }
}

int host_check_results(TestResults* results, bool print) {
  if (print) {
    std::cout << "r0=1, z=2, r2=1, r3=0 (weak): " << results->weak << "\n";
    std::cout << "other: " << results->other << "\n";
    std::cout << "thread skipped: " << results->na << "\n";
  }
  return results->weak;
}

