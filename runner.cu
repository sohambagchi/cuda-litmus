#include <iostream>
#include <chrono>
#include <unistd.h>
#include <cuda_runtime.h>
#include <atomic>
#include <cuda/atomic>
#include "functions.cuh"

// CUDA kernel for vector addition
__global__ void vectorAdd(const float *A, const float *B, float *C, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

typedef struct {
    int numOutputs;
    int numMemLocations;
    int numResults;
    int permuteLocation;
    int aliasedMemory;
    int workgroupMemory;
    int checkMemory;
} TestParams; 

typedef struct {
    int testIterations;
    int testingWorkgroups;
    int maxWorkgroups;
    int workgroupSize;
    int shufflePct;
    int barrierPct;
    int stressLineSize;
    int stressTargetLines;
    int scratchMemorySize;
    int memStride;
    int memStressPct;
    int memStressIterations;
    int memStressPattern;
    int preStressPct;
    int preStressIterations;
    int preStressPattern;
    int stressAssignmentStrategy;
    int permuteThread;
} StressParams;

typedef struct {
    cuda::atomic<uint> seq0;
    cuda::atomic<uint>  seq1;
    cuda::atomic<uint> interleaved;
    cuda::atomic<uint> weak;
} TestResults;


__global__ void check_results (
    uint* test_locations,
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
	test_results->interleaved.fetch_add(1);
    } else if ((r0 == 1 && r1 == 0)) {
	test_results->weak.fetch_add(1);
    }
}

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
	test_locations[y_0].store(1, cuda::memory_order_relaxed);
	uint r0 = test_locations[y_1].load(cuda::memory_order_relaxed);
	uint r1 = test_locations[x_1].load(cuda::memory_order_relaxed);
	read_results[id_1 * 2 + 1] = r1;
	read_results[id_1 * 2] = r0;
    } else if (stress_params[1]) {
        do_stress(scratchpad, scratch_locations, stress_params[2], stress_params[3]);
    }
}

int parseTestParamsFile(const char *filename, TestParams *config) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        perror("Error opening file");
        return -1;
    }

    char line[256];
    while (fgets(line, sizeof(line), file)) {
        char key[64];
        int value;
        if (sscanf(line, "%63[^=]=%d", key, &value) == 2) {
            if (strcmp(key, "numOutputs") == 0) config->numOutputs = value;
            else if (strcmp(key, "numMemLocations") == 0) config->numMemLocations = value;
            else if (strcmp(key, "numResults") == 0) config->numResults = value;
            else if (strcmp(key, "permuteLocation") == 0) config->permuteLocation = value;
            else if (strcmp(key, "aliasedMemory") == 0) config->aliasedMemory = value;
            else if (strcmp(key, "workgroupMemory") == 0) config->workgroupMemory = value;
            else if (strcmp(key, "checkMemory") == 0) config->checkMemory = value;
        }
    }

    fclose(file);
    return 0;
}

int parseStressParamsFile(const char *filename, StressParams *config) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        perror("Error opening file");
        return -1;
    }

    char line[256];
    while (fgets(line, sizeof(line), file)) {
        char key[64];
        int value;
        if (sscanf(line, "%63[^=]=%d", key, &value) == 2) {
            if (strcmp(key, "testIterations") == 0) config->testIterations = value;
            else if (strcmp(key, "testingWorkgroups") == 0) config->testingWorkgroups = value;
            else if (strcmp(key, "maxWorkgroups") == 0) config->maxWorkgroups = value;
            else if (strcmp(key, "workgroupSize") == 0) config->workgroupSize = value;
            else if (strcmp(key, "shufflePct") == 0) config->shufflePct = value;
            else if (strcmp(key, "barrierPct") == 0) config->barrierPct = value;
            else if (strcmp(key, "stressLineSize") == 0) config->stressLineSize = value;
            else if (strcmp(key, "stressTargetLines") == 0) config->stressTargetLines = value;
            else if (strcmp(key, "scratchMemorySize") == 0) config->scratchMemorySize = value;
            else if (strcmp(key, "memStride") == 0) config->memStride = value;
            else if (strcmp(key, "memStressPct") == 0) config->memStressPct = value;
            else if (strcmp(key, "memStressIterations") == 0) config->memStressIterations = value;
            else if (strcmp(key, "memStressPattern") == 0) config->memStressPattern = value;
            else if (strcmp(key, "preStressPct") == 0) config->preStressPct = value;
            else if (strcmp(key, "preStressIterations") == 0) config->preStressIterations = value;
            else if (strcmp(key, "preStressPattern") == 0) config->preStressPattern = value;
            else if (strcmp(key, "stressAssignmentStrategy") == 0) config->stressAssignmentStrategy = value;
            else if (strcmp(key, "permuteThread") == 0) config->permuteThread = value;
        }
    }

    fclose(file);
    return 0;
}

void run(StressParams stressParams, TestParams testParams, bool print_results) {
    int testingThreads = stressParams.workgroupSize * stressParams.testingWorkgroups;
    int testLocSize = testingThreads * testParams.numMemLocations * stressParams.memStride;

    uint *testLocations;
    cudaMalloc(&testLocations, testLocSize * sizeof(uint));

    uint *readResults;
    cudaMalloc(&readResults, testParams.numOutputs * testingThreads * sizeof(uint));

    uint *h_testResults = (uint *)malloc(testParams.numResults * sizeof(uint));
    uint *d_testResults;
    cudaMalloc(&d_testResults, testParams.numResults * sizeof(uint));

    uint *h_shuffledWorkgroups = (uint *)malloc(stressParams.maxWorkgroups * sizeof(uint));
    uint *d_shuffledWorkgroups;
    cudaMalloc(&d_shuffledWorkgroups, stressParams.maxWorkgroups * sizeof(uint));

    uint *barrier;
    cudaMalloc(&barrier, 1 * sizeof(uint));

    uint *scratchpad;
    cudaMalloc(&scratchpad, stressParams.scratchMemorySize * sizeof(uint));

    uint *h_scratchLocations = (uint *)malloc(stressParams.maxWorkgroups * sizeof(uint));
    uint *d_scratchLocations;
    cudaMalloc(&d_shuffledWorkgroups, stressParams.maxWorkgroups * sizeof(uint));

    uint *h_stressParams = (uint *)malloc(12 * sizeof(uint));
    uint *d_stressParams;
    cudaMalloc(&d_stressParams, 12 * sizeof(uint));

    // run iterations
    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();
    int weakBehaviors = 0;
    int totalBehaviors = 0;

    float testTime = 0;



    int N = 100000; // Number of elements in each vector
    size_t size = N * sizeof(float);

    // Allocate host memory
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    // Initialize vectors on the host
    for (int i = 0; i < N; i++) {
        h_A[i] = i * 1.0f;
        h_B[i] = i * 2.0f;
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy vectors from host memory to device memory
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Number of threads per block
    int threadsPerBlock = 256;

    // Number of blocks in grid
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the vectorAdd CUDA kernel
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Copy the result vector from device memory to host memory
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Verify the result
    for (int i = 0; i < N; i++) {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
            std::cerr << "Result verification failed at element " << i << "!" << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    std::cout << "Test PASSED" << std::endl;

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);
}

int main(int argc, char *argv[]) {
    char* stress_params_file = nullptr;
    char* test_params_file = nullptr;
    bool print_results = false;

    int c;
    while ((c = getopt(argc, argv, "xs:t:")) != -1)
    switch (c)
    {
    case 's':
      stress_params_file = optarg;
      break;
    case 't':
      test_params_file = optarg;
      break;
    case 'x':
      print_results = true;
      break;
    case '?':
      if (optopt == 's' || optopt == 't')
        std::cerr << "Option -" << optopt << "requires an argument\n";
      else
        std::cerr << "Unknown option" << optopt << "\n";
      return 1;
    default:
      abort();
    }

    if (stress_params_file == nullptr) {
        std::cerr << "Stress param file (-s) must be set\n";
        return 1;
    }

    if (test_params_file == nullptr) {
        std::cerr << "Test param file (-t) must be set\n";
        return 1;
    }

    StressParams stressParams;
    if (parseStressParamsFile(stress_params_file, &stressParams) != 0) {
        return 1;
    }

    TestParams testParams;
    if (parseTestParamsFile(test_params_file, &testParams) != 0) {
        return 1;
    }
    run(stressParams, testParams, print_results);
}

