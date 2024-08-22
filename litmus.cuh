// litmus.cuh
#ifndef LITMUS_CUH
#define LITMUS_CUH
#endif

#include <cuda_runtime.h>
#include <cuda/atomic>

typedef struct {
    cuda::atomic<uint, cuda::thread_scope_device> seq0;
    cuda::atomic<uint, cuda::thread_scope_device>  seq1;
    cuda::atomic<uint, cuda::thread_scope_device>  seq2;
    cuda::atomic<uint, cuda::thread_scope_device>  seq3;
    cuda::atomic<uint, cuda::thread_scope_device>  seq_inter0;
    cuda::atomic<uint, cuda::thread_scope_device>  seq_inter1;
    cuda::atomic<uint, cuda::thread_scope_device> interleaved0;
    cuda::atomic<uint, cuda::thread_scope_device> interleaved1;
    cuda::atomic<uint, cuda::thread_scope_device> interleaved2;
    cuda::atomic<uint, cuda::thread_scope_device> weak;
    cuda::atomic<uint, cuda::thread_scope_device> other;
} TestResults;

__global__ void litmus_test(
    cuda::atomic<uint, cuda::thread_scope_device>* test_locations,
    uint* read_results,
    uint* shuffled_workgroups,
    cuda::atomic<uint, cuda::thread_scope_device>* barrier,
    uint* scratchpad,
    uint* scratch_locations,
    uint* stress_params);

__global__ void check_results (
    cuda::atomic<uint, cuda::thread_scope_device>* test_locations,
    uint* read_results,
    TestResults* test_results,
    uint* stress_params);

int host_check_results(TestResults* results, bool print);

