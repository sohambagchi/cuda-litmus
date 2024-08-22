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
	uint id_0_final = id_0 % total_ids;
        bool id_0_first_half = id_0 / total_ids == 0;
        uint new_workgroup = stripe_workgroup(shuffled_workgroup, threadIdx.x, stress_params[9]);
        uint id_1 = new_workgroup * blockDim.x + permute_id(threadIdx.x, stress_params[7], blockDim.x);
	uint id_1_final = id_1 % total_ids;
        bool id_1_first_half = id_1 / total_ids == 0;

	uint mem_0;
        if (id_0_first_half) {
            mem_0 = id_0_final * stress_params[10] * 2;
        } else {
            mem_0 = permute_id(id_0_final, stress_params[8], total_ids) * stress_params[10] * 2 + stress_params[11];
        }
        uint x_1 = (id_1_final) * stress_params[10] * 2;
        uint y_1 = (permute_id(id_1_final, stress_params[8], total_ids)) * stress_params[10] * 2 + stress_params[11];

        if (stress_params[4]) {
            do_stress(scratchpad, scratch_locations, stress_params[5], stress_params[6]);
        }
        if (stress_params[0]) {
            spin(barrier, blockDim.x * stress_params[9]);
        }

	test_locations[mem_0].store(1, cuda::memory_order_relaxed); // write to either x or y depending on thread

	if (id_0_final != id_1_final) {
            if (id_1_first_half) { // one observer thread reads x then y
		uint r0 = test_locations[x_1].load(cuda::memory_order_relaxed);
		uint r1 = test_locations[y_1].load(cuda::memory_order_relaxed);
	        cuda::atomic_thread_fence(cuda::memory_order_seq_cst);
	        read_results[id_1_final * 4] = r0;
	        read_results[id_1_final * 4 + 1] = r1;
            } else { // other observer thread reads y then x
		uint r2 = test_locations[y_1].load(cuda::memory_order_relaxed);
		uint r3 = test_locations[x_1].load(cuda::memory_order_relaxed);
	        cuda::atomic_thread_fence(cuda::memory_order_seq_cst);
	        read_results[id_1_final * 4 + 2] = r2;
	        read_results[id_1_final * 4 + 3] = r3;
            }
        }
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
    if (id_0 < (blockDim.x * stress_params[9])/2) {
        uint x_0 = id_0 * stress_params[10] * 2;
	uint mem_x_0 = test_locations[x_0];
        uint r0 = read_results[id_0 * 4];
        uint r1 = read_results[id_0 * 4 + 1];
        uint r2 = read_results[id_0 * 4 + 2];
        uint r3 = read_results[id_0 * 4 + 3];

        if (r0 == 0 && r1 == 0 && r2 == 0 && r3 == 0) { // both observers run first
	    test_results->seq0.fetch_add(1);
        } else if (r0 == 1 && r1 == 1 && r2 == 1 && r3 == 1) { // both observers run last
	    test_results->seq1.fetch_add(1);
        } else if (r0 == 0 && r1 == 0 && r2 == 1 && r3 == 1) { // first observer runs first
	    test_results->seq2.fetch_add(1);
        } else if (r0 == 1 && r1 == 1 && r2 == 0 && r3 == 0) { // second observer runs first
	    test_results->seq3.fetch_add(1);
        } else if (r0 == r1 && r2 != r3) { // second observer interleaved
	    test_results->seq_inter0.fetch_add(1);
        } else if (r0 != r1 && r2 == r3) { // first observer interleaved
	    test_results->seq_inter1.fetch_add(1);
        } else if (r0 == 0 && r1 == 1 && r2 == 0 && r3 == 1) { // both interleaved
	    test_results->interleaved0.fetch_add(1);
        } else if (r0 == 0 && r1 == 1 && r2 == 1 && r3 == 0) { // both interleaved
	    test_results->interleaved1.fetch_add(1);
        } else if (r0 == 1 && r1 == 0 && r2 == 0 && r3 == 1) { // both interleaved
	    test_results->interleaved2.fetch_add(1);
        } else if (r0 == 1 && r1 == 0 && r2 == 1 && r3 == 0) { // observer threads see x/y in different orders
	    test_results->weak.fetch_add(1);
        } else {
	    test_results->other.fetch_add(1);
        }
    }
}

int host_check_results(TestResults* results, bool print) {
    if (print) {

        std::cout << "r0=0, r1=0, r2=0, r3=0 (seq): " << results->seq0 << "\n";
        std::cout << "r0=1, r1=1, r2=1, r3=1 (seq): " << results->seq1 << "\n";
	std::cout << "r0=0, r1=0, r2=1, r3=1 (seq): " << results->seq2 << "\n";
	std::cout << "r0=1, r1=1, r2=0, r3=0 (seq): " << results->seq3 << "\n";
	std::cout << "r0 == r1, r2 != r3 (seq/interleaved): " << results->seq_inter0 << "\n";
	std::cout << "r0 != r1, r2 == r3 (interleaved/seq): " << results->seq_inter1 << "\n";
	std::cout << "r0=0, r1=1, r2=0, r3=1 (interleaved): " << results->interleaved0 << "\n";
	std::cout << "r0=0, r1=1, r2=1, r3=0 (interleaved): " << results->interleaved1 << "\n";
	std::cout << "r0=1, r1=0, r2=0, r3=1 (interleaved): " << results->interleaved2 << "\n";
	std::cout << "r0=1, r1=0, r2=1, r3=0 (weak): " << results->weak << "\n";
	std::cout << "other: " << results->other << "\n";
    }
    return results->weak;
}

