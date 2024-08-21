// litmus.cuh
#ifndef LITMUS_CUH
#define LITMUS_CUH
#endif

#include <cuda_runtime.h>
#include <cuda/atomic>

typedef struct {
    cuda::atomic<uint> seq0;
    cuda::atomic<uint>  seq1;
    cuda::atomic<uint>  seq2;
    cuda::atomic<uint>  seq3;
    cuda::atomic<uint>  seq_inter0;
    cuda::atomic<uint>  seq_inter1;
    cuda::atomic<uint> interleaved0;
    cuda::atomic<uint> interleaved1;
    cuda::atomic<uint> interleaved2;
    cuda::atomic<uint> weak;
    cuda::atomic<uint> other;
} TestResults;

__global__ void litmus_test(
    cuda::atomic<uint>* test_locations,
    uint* read_results,
    uint* shuffled_workgroups,
    cuda::atomic<uint>* barrier,
    uint* scratchpad,
    uint* scratch_locations,
    uint* stress_params);

__global__ void check_results (
    cuda::atomic<uint>* test_locations,
    uint* read_results,
    TestResults* test_results,
    uint* stress_params);

int host_check_results(TestResults* results, bool print);

