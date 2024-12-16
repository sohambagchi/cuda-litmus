#include <iostream>
#include "litmus.cuh"
#include "functions.cu"

// iriw thread mappings
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
  KernelParams* kernel_params) {
  uint shuffled_workgroup = shuffled_workgroups[blockIdx.x];
  if (shuffled_workgroup < kernel_params->testing_workgroups) {

    uint total_ids = blockDim.x * kernel_params->testing_workgroups;
    uint id_0 = shuffled_workgroup * blockDim.x + threadIdx.x;
    uint workgroup_1 = stripe_workgroup(shuffled_workgroup, threadIdx.x, kernel_params->testing_workgroups);
    uint id_1 = workgroup_1 * blockDim.x + threadIdx.x;
    uint workgroup_2 = stripe_workgroup(workgroup_1, threadIdx.x, kernel_params->testing_workgroups);
    uint id_2 = workgroup_2 * blockDim.x + threadIdx.x;
    uint workgroup_3 = stripe_workgroup(workgroup_2, threadIdx.x, kernel_params->testing_workgroups);
    uint id_3 = workgroup_3 * blockDim.x + threadIdx.x;

    uint x_0 = id_0 * kernel_params->mem_stride * 4;
    uint permute_id_0 = permute_id(id_0, kernel_params->permute_location, total_ids);
    uint z_0 = permute_id(permute_id_0, kernel_params->permute_location, total_ids) * kernel_params->mem_stride * 4 + 2 * kernel_params->mem_offset;

    uint permute_id_1 = permute_id(id_1, kernel_params->permute_location, total_ids);
    uint y_1 = permute_id_1 * kernel_params->mem_stride * 4 + kernel_params->mem_offset;
    uint z_1 = permute_id(permute_id_1, kernel_params->permute_location, total_ids) * kernel_params->mem_stride * 4 + 2 * kernel_params->mem_offset;

    uint permute_id_2 = permute_id(id_2, kernel_params->permute_location, total_ids);
    uint y_2 = permute_id_2 * kernel_params->mem_stride * 4 + kernel_params->mem_offset;
    uint permute_permute_id_2 = permute_id(permute_id_2, kernel_params->permute_location, total_ids);
    uint a_2 = permute_id(permute_permute_id_2, kernel_params->permute_location, total_ids) * kernel_params->mem_stride * 4 + 3 * kernel_params->mem_offset;

    uint x_3 = id_3 * kernel_params->mem_stride * 4;
    uint permute_id_3 = permute_id(id_3, kernel_params->permute_location, total_ids);
    uint permute_permute_id_3 = permute_id(permute_id_3, kernel_params->permute_location, total_ids);
    uint a_3 = permute_id(permute_permute_id_3, kernel_params->permute_location, total_ids) * kernel_params->mem_stride * 4 + 3 * kernel_params->mem_offset;

    PRE_STRESS();

    if (id_0 != id_1 && id_0 != id_2 && id_0 != id_3 && id_1 != id_2 && id_1 != id_3 && id_2 != id_3) {

      cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_device);
      test_locations[x_0].store(1, cuda::memory_order_release); // write x
      test_locations[z_0].store(1, cuda::memory_order_release); // write z

      uint r0 = test_locations[z_1].load(cuda::memory_order_acquire); // read z
      test_locations[y_1].store(1, cuda::memory_order_release); // write y
      cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_device);
      uint r1 = test_locations[y_1].load(cuda::memory_order_relaxed); // read y

      cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_device);
      test_locations[y_2].store(2, cuda::memory_order_release); // write y
      test_locations[a_2].store(1, cuda::memory_order_release); // write a

      uint r2 = test_locations[a_3].load(cuda::memory_order_acquire); // read a
      uint r3 = test_locations[x_3].load(cuda::memory_order_acquire); // read x
      cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_device);

      cuda::atomic_thread_fence(cuda::memory_order_seq_cst);
      read_results[id_1].r0 = r0;
      read_results[id_1].r1 = r1;
      read_results[id_3].r2 = r2;
      read_results[id_3].r3 = r3;
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
  uint x = test_locations[id_0 * kernel_params->mem_stride * 4];
  uint r0 = read_results[id_0].r0;
  uint r1 = read_results[id_0].r1;
  uint r2 = read_results[id_0].r2;
  uint r3 = read_results[id_0].r3;

  if (x == 0) {
    test_results->na.fetch_add(1); // thread skipped
  }
  else if (r0 == 1 && r1 == 2 && r2 == 1 && r3 == 0) { // weak behavior
    test_results->weak.fetch_add(1);
  }
  else {
    test_results->other.fetch_add(1);
  }
}

int host_check_results(TestResults* results, bool print) {
  if (print) {
    std::cout << "r0=1, r1=2, r2=1, r3=0 (weak): " << results->weak << "\n";
    std::cout << "other: " << results->other << "\n";
    std::cout << "thread skipped: " << results->na << "\n";
  }
  return results->weak;
}

