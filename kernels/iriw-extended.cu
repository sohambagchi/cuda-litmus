#include <iostream>
#include "litmus.cuh"
#include "functions.cu"

// iriw-extended thread mappings
// thread 0: write x, fence, read a, read b
// thread 1: read x, read y, fence, write b
// thread 2: write y, fence, read b, read a
// thread 3: read y, read x, fence, write a

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

#ifdef ACQUIRE
    cuda::memory_order load_order = cuda::memory_order_acquire;
#else
    cuda::memory_order load_order = cuda::memory_order_relaxed;
#endif

    // defined for different distributions of threads across threadblocks
    DEFINE_IDS();

    uint x_0 = (wg_offset + id_0) * kernel_params->mem_stride * 4;
    uint permute_id_0 = permute_id(id_0, kernel_params->permute_location, total_ids);
    //uint y_0 = (wg_offset + permute_id_0) * kernel_params->mem_stride * 4 + kernel_params->mem_offset;
    uint permute_2_id_0 = permute_id(permute_id_0, kernel_params->permute_location, total_ids);
    uint a_0 = (wg_offset + permute_2_id_0) * kernel_params->mem_stride * 4 + 2 * kernel_params->mem_offset;
    uint b_0 = (wg_offset + permute_id(permute_2_id_0, kernel_params->permute_location, total_ids)) * kernel_params->mem_stride * 4 + 3 * kernel_params->mem_offset;

    uint x_1 = (wg_offset + id_1) * kernel_params->mem_stride * 4;
    uint permute_id_1 = permute_id(id_1, kernel_params->permute_location, total_ids);
    uint y_1 = (wg_offset + permute_id_1) * kernel_params->mem_stride * 4 + kernel_params->mem_offset;
    uint permute_2_id_1 = permute_id(permute_id_1, kernel_params->permute_location, total_ids);
    uint b_1 = (wg_offset + permute_id(permute_2_id_1, kernel_params->permute_location, total_ids)) * kernel_params->mem_stride * 4 + 3 * kernel_params->mem_offset;

    uint permute_id_2 = permute_id(id_2, kernel_params->permute_location, total_ids);
    uint y_2 = (wg_offset + permute_id_2) * kernel_params->mem_stride * 4 + kernel_params->mem_offset;
    uint permute_2_id_2 = permute_id(permute_id_2, kernel_params->permute_location, total_ids);
    uint a_2 = (wg_offset + permute_2_id_2) * kernel_params->mem_stride * 4 + 2 * kernel_params->mem_offset;
    uint b_2 = (wg_offset + permute_id(permute_2_id_2, kernel_params->permute_location, total_ids)) * kernel_params->mem_stride * 4 + 3 * kernel_params->mem_offset;

    uint x_3 = (wg_offset + id_3) * kernel_params->mem_stride * 4;
    uint permute_id_3 = permute_id(id_3, kernel_params->permute_location, total_ids);
    uint y_3 = (wg_offset + permute_id_3) * kernel_params->mem_stride * 4 + kernel_params->mem_offset;
    uint a_3 = (wg_offset + permute_id(permute_id_3, kernel_params->permute_location, total_ids)) * kernel_params->mem_stride * 4 + 2 * kernel_params->mem_offset;

    // Save threads and memory locations involved in a test instance
    //uint t_id = blockIdx.x * blockDim.x + threadIdx.x;
    //test_instances[id_0].t0 = t_id;
    //test_instances[id_1].t1 = t_id;
    //test_instances[id_2].t2 = t_id;
    //test_instances[id_3].t3 = t_id;
    //test_instances[id_0].x = x_0;
    //test_instances[id_0].y = y_0;

    PRE_STRESS();

    if (id_0 != id_1 && id_0 != id_2 && id_0 != id_3 && id_1 != id_2 && id_1 != id_3 && id_2 != id_3) {

      test_locations[x_0].store(1, cuda::memory_order_relaxed); // write x
//      cuda::atomic_thread_fence(cuda::memory_order_seq_cst);
      uint r4 = test_locations[a_0].load(load_order); // read a
      uint r5 = test_locations[b_0].load(cuda::memory_order_relaxed); // read b

      uint r0 = test_locations[x_1].load(load_order); // read x
      uint r1 = test_locations[y_1].load(cuda::memory_order_relaxed); // read y
 //     cuda::atomic_thread_fence(cuda::memory_order_seq_cst);
      test_locations[b_1].store(1, cuda::memory_order_relaxed); // write b

      test_locations[y_2].store(1, cuda::memory_order_relaxed); // write y
  //    cuda::atomic_thread_fence(cuda::memory_order_seq_cst);
      uint r6 = test_locations[b_2].load(load_order); // read b
      uint r7 = test_locations[a_2].load(cuda::memory_order_relaxed); // read a

      uint r2 = test_locations[y_3].load(load_order); // read y
      uint r3 = test_locations[x_3].load(cuda::memory_order_relaxed); // read x
   //   cuda::atomic_thread_fence(cuda::memory_order_seq_cst);
      test_locations[a_3].store(1, cuda::memory_order_relaxed); // write a

      cuda::atomic_thread_fence(cuda::memory_order_seq_cst);
      read_results[wg_offset + id_0].r4 = r4;
      read_results[wg_offset + id_0].r5 = r5;
      read_results[wg_offset + id_1].r0 = r0;
      read_results[wg_offset + id_1].r1 = r1;
      read_results[wg_offset + id_2].r6 = r6;
      read_results[wg_offset + id_2].r7 = r7;
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
  KernelParams* kernel_params,
  bool* weak) {
  uint id_0 = blockIdx.x * blockDim.x + threadIdx.x;
  uint x = test_locations[id_0 * kernel_params->mem_stride * 4];
  uint r0 = read_results[id_0].r0;
  uint r1 = read_results[id_0].r1;
  uint r2 = read_results[id_0].r2;
  uint r3 = read_results[id_0].r3;
  uint r4 = read_results[id_0].r4;
  uint r5 = read_results[id_0].r5;
  uint r6 = read_results[id_0].r6;
  uint r7 = read_results[id_0].r7;

  if (x == 0) {
    test_results->na.fetch_add(1); // thread skipped
    return;
  }
  bool checked = false;
  if (r0 == 1 && r1 == 0 && r2 == 1 && r3 == 0) { // observer threads see x/y in different orders
    test_results->res14.fetch_add(1);
    checked = true;
  }
  if (r4 == 1 && r5 == 0 && r6 == 1 && r7 == 0) { // observer threads see a/b in different orders
    test_results->res15.fetch_add(1);
    checked = true;
  }
  if (r0 == 1 && r1 == 0 && r2 == 1 && r3 == 0 && r4 == 1 && r5 == 0 && r6 == 1 && r7 == 0) { // both tests see weak behavior
    test_results->weak.fetch_add(1);
    weak[id_0] = true;
    checked = true;
  }
  if (!checked) {
    test_results->other.fetch_add(1);
  }
}

int host_check_results(TestResults* results, bool print) {
  if (print) {
    std::cout << "r0=1, r1=0, r2=1, r3=0, r4=1, r5=0, r6=1, r7=0 (both weak): " << results->weak << "\n";
    std::cout << "r0=1, r1=0, r2=1, r3=0 (x/y weak): " << results->res14 << "\n";
    std::cout << "r4=1, r5=0, r6=1, r7=0 (a/b weak): " << results->res15 << "\n";
    std::cout << "thread skipped: " << results->na << "\n";
    std::cout << "other: " << results->other << "\n";
  }
  return results->weak + results->res14 + results->res15;
}
