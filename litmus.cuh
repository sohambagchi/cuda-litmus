// litmus.cuh
#ifndef LITMUS_CUH
#define LITMUS_CUH
#endif

#include <cuda_runtime.h>
#include <cuda/atomic>

#ifdef SCOPE_DEVICE
typedef cuda::atomic<uint, cuda::thread_scope_device> d_atomic_uint;
#elif defined(SCOPE_BLOCK)
typedef cuda::atomic<uint, cuda::thread_scope_block> d_atomic_uint;
#elif defined(SCOPE_SYSTEM)
typedef cuda::atomic<uint, cuda::thread_scope_system> d_atomic_uint;
#else
typedef cuda::atomic<uint> d_atomic_uint; // default, which is system too
#endif


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

typedef struct {
    bool barrier;
    bool mem_stress;
    int mem_stress_iterations;
    int mem_stress_pattern;
    bool pre_stress;
    int pre_stress_iterations;
    int pre_stress_pattern;
    int permute_thread;
    int permute_location;
    int testing_workgroups;
    int mem_stride;
    int mem_offset;
} KernelParams;

__global__ void litmus_test(
    d_atomic_uint* test_locations,
    uint* read_results,
    uint* shuffled_workgroups,
    cuda::atomic<uint, cuda::thread_scope_device>* barrier,
    uint* scratchpad,
    uint* scratch_locations,
    KernelParams* kernel_params);

__global__ void check_results (
    d_atomic_uint* test_locations,
    uint* read_results,
    TestResults* test_results,
    KernelParams* kernel_params);

int host_check_results(TestResults* results, bool print);

