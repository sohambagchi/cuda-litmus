#include <iostream>
#include "litmus.cuh"
#include "functions.cuh"

__global__ void litmus_test(
    cuda::atomic<uint>* test_locations,
    uint* read_results,
    uint* shuffled_workgroups,
    cuda::atomic<uint>* barrier,
    uint* scratchpad,
    uint* scratch_locations,
    uint* stress_params) {

    uint shuffled_workgroup = shuffled_workgroups[blockIdx.x];
    if (shuffled_workgroup < stress_params[9]) {
        uint total_ids = blockDim.x * stress_params[9];
        uint id_0 = shuffled_workgroup * blockDim.x + threadIdx.x;
        uint new_workgroup = stripe_workgroup(shuffled_workgroup, threadIdx.x, stress_params[9]);
        uint id_1 = new_workgroup * blockDim.x + permute_id(threadIdx.x, stress_params[7], blockDim.x);
        uint x_0 = id_0 * stress_params[10] * 2;
        uint y_0 = permute_id(id_0, stress_params[8], total_ids) * stress_params[10] * 2 + stress_params[11];
        uint x_1 = id_1 * stress_params[10] * 2;
        uint y_1 = permute_id(id_1, stress_params[8], total_ids) * stress_params[10] * 2 + stress_params[11];

        if (stress_params[4]) {
            do_stress(scratchpad, scratch_locations, stress_params[5], stress_params[6]);
        }
        if (stress_params[0]) {
            spin(barrier, blockDim.x * stress_params[9]);
        }

	test_locations[x_0].store(1, cuda::memory_order_relaxed);
	//cuda::atomic_thread_fence(cuda::memory_order_release, cuda::thread_scope_device);
	test_locations[y_0].store(1, cuda::memory_order_relaxed);
	uint r0 = test_locations[y_1].load(cuda::memory_order_relaxed);
	//cuda::atomic_thread_fence(cuda::memory_order_acquire, cuda::thread_scope_device);
	uint r1 = test_locations[x_1].load(cuda::memory_order_relaxed);
	cuda::atomic_thread_fence(cuda::memory_order_seq_cst);
	read_results[id_1 * 2 + 1] = r1;
	read_results[id_1 * 2] = r0;
    } else if (stress_params[1]) {
        do_stress(scratchpad, scratch_locations, stress_params[2], stress_params[3]);
    }
}

__global__ void check_results (
    cuda::atomic<uint>* test_locations,
    uint* read_results,
    TestResults* test_results,
    uint* stress_params) {

    uint id_0 = blockIdx.x * blockDim.x + threadIdx.x;
    uint x_0 = id_0 * stress_params[10] * 2;
    uint mem_x_0 = test_locations[x_0];
    uint r0 = read_results[id_0 * 2];
    uint r1 = read_results[id_0 * 2 + 1];
    uint total_ids = blockDim.x * stress_params[9];
    uint y_0 = permute_id(id_0, stress_params[8], total_ids) * stress_params[10] * 2 + stress_params[11];
    uint mem_y_0 = test_locations[y_0];

    if ((r0 == 0 && r1 == 0)) {
	test_results->seq0.fetch_add(1);
    } else if ((r0 == 1 && r1 == 1)) {
	test_results->seq1.fetch_add(1);
    } else if ((r0 == 0 && r1 == 1)) {
	test_results->interleaved0.fetch_add(1);
    } else if ((r0 == 1 && r1 == 0)) {
	test_results->weak.fetch_add(1);
    }
}

int host_check_results(TestResults* results, bool print) {
  if (print) {
    std::cout << "r0=0, r1=0 (seq): " << results->seq0 << "\n";
    std::cout << "r0=1, r1=1 (seq): " << results->seq1 << "\n";
    std::cout << "r0=0, r1=1 (interleaved): " << results->interleaved0 << "\n";
    std::cout << "r0=1, r1=0 (weak): " << results->weak << "\n";
  }
  return results->weak;
}

