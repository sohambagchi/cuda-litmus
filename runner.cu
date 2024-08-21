#include <iostream>
#include <set>
#include <chrono>
#include <iomanip>
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
	test_results->interleaved.fetch_add(1);
    } else if ((r0 == 1 && r1 == 0)) {
	test_results->weak.fetch_add(1);
    }
}

int host_check_results(TestResults* results, bool print) {
  if (print) {
    std::cout << "r0=0, r1=0 (seq): " << results->seq0 << "\n";
    std::cout << "r0=1, r1=1 (seq): " << results->seq1 << "\n";
    std::cout << "r0=0, r1=1 (interleaved): " << results->interleaved << "\n";
    std::cout << "r0=1, r1=0 (weak): " << results->weak << "\n";
  }
  return results->weak;
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

/** Returns a value between the min and max (inclusive). */
int setBetween(int min, int max) {
  if (min == max) {
    return min;
  } else {
    int size = rand() % (max - min + 1);
    return min + size;
  }
}

bool percentageCheck(int percentage) {
  return rand() % 100 < percentage;
}

/** Assigns shuffled workgroup ids, using the shufflePct to determine whether the ids should be shuffled this iteration. */
void setShuffledWorkgroups(uint* h_shuffledWorkgroups, int numWorkgroups, int shufflePct) {
  for (int i = 0; i < numWorkgroups; i++) {
    h_shuffledWorkgroups[i] = i; 
  }
  if (percentageCheck(shufflePct)) {
    for (int i = numWorkgroups - 1; i > 0; i--) {
      int swap = rand() % (i + 1);
      int temp = h_shuffledWorkgroups[i];
      h_shuffledWorkgroups[i] = h_shuffledWorkgroups[swap];
      h_shuffledWorkgroups[swap] = temp;
    }
  }
}

/** Sets the stress regions and the location in each region to be stressed. Uses the stress assignment strategy to assign
  * workgroups to specific stress locations. Assignment strategy 0 corresponds to a "round-robin" assignment where consecutive
  * threads access separate scratch locations, while assignment strategy 1 corresponds to a "chunking" assignment where a group
  * of consecutive threads access the same location.
  */
void setScratchLocations(uint* h_locations, int numWorkgroups, StressParams params) {
  std::set<int> usedRegions;
  int numRegions = params.scratchMemorySize / params.stressLineSize;
  for (int i = 0; i < params.stressTargetLines; i++) {
    int region = rand() % numRegions;
    while(usedRegions.count(region))
      region = rand() % numRegions;
    int locInRegion = rand() % params.stressLineSize;
    switch (params.stressAssignmentStrategy) {
      case 0:
        for (int j = i; j < numWorkgroups; j += params.stressTargetLines) {
	  h_locations[j] = (region * params.stressLineSize) + locInRegion;
        }
        break;
      case 1:
        int workgroupsPerLocation = numWorkgroups/params.stressTargetLines;
        for (int j = 0; j < workgroupsPerLocation; j++) {
	  h_locations[i*workgroupsPerLocation + j] = (region * params.stressLineSize) + locInRegion;
        }
        if (i == params.stressTargetLines - 1 && numWorkgroups % params.stressTargetLines != 0) {
          for (int j = 0; j < numWorkgroups % params.stressTargetLines; j++) {
	    h_locations[numWorkgroups - j - 1] = (region * params.stressLineSize) + locInRegion;
          }
        }
        break;
    }
  }
}

/** These parameters vary per iteration, based on a given percentage. */
void setDynamicStressParams(uint* h_stressParams, StressParams params) {
  if (percentageCheck(params.barrierPct)) {
    h_stressParams[0] = 1;
  } else {
    h_stressParams[0] = 0;
  }
  if (percentageCheck(params.memStressPct)) {
    h_stressParams[1] = 1;
  } else {
    h_stressParams[1] = 0;
  }
  if (percentageCheck(params.preStressPct)) {
    h_stressParams[4] = 1;
  } else {
    h_stressParams[4] = 0;
  }
}

/** These parameters are static for all iterations of the test. Aliased memory is used for coherence tests. */
void setStaticStressParams(uint* h_stressParams, StressParams stressParams, TestParams testParams) {
  h_stressParams[2] = stressParams.memStressIterations;
  h_stressParams[3] = stressParams.memStressPattern;
  h_stressParams[5] = stressParams.preStressIterations;
  h_stressParams[6] = stressParams.preStressPattern;
  h_stressParams[7] = stressParams.permuteThread;
  h_stressParams[8] = testParams.permuteLocation;
  h_stressParams[9] = stressParams.testingWorkgroups;
  h_stressParams[10] = stressParams.memStride;

  if (testParams.aliasedMemory == 1) {
    h_stressParams[11] = 0;
  } else {
    h_stressParams[11] = stressParams.memStride;
  }
}

void run(StressParams stressParams, TestParams testParams, bool print_results) {
    int testingThreads = stressParams.workgroupSize * stressParams.testingWorkgroups;

    int testLocSize = testingThreads * testParams.numMemLocations * stressParams.memStride * sizeof(uint);
    cuda::atomic<uint> *testLocations;
    cudaMalloc(&testLocations, testLocSize);

    int readResultsSize = testParams.numOutputs * testingThreads * sizeof(uint);
    uint *readResults;
    cudaMalloc(&readResults, readResultsSize);

    TestResults *h_testResults = (TestResults *)malloc(sizeof(TestResults));
    TestResults *d_testResults;
    cudaMalloc(&d_testResults, sizeof(TestResults));

    int shuffledWorkgroupsSize = stressParams.maxWorkgroups * sizeof(uint);
    uint *h_shuffledWorkgroups = (uint *)malloc(shuffledWorkgroupsSize);
    uint *d_shuffledWorkgroups;
    cudaMalloc(&d_shuffledWorkgroups, shuffledWorkgroupsSize);

    int barrierSize = sizeof(uint);
    cuda::atomic<uint> *barrier;
    cudaMalloc(&barrier, barrierSize);

    int scratchpadSize = stressParams.scratchMemorySize * sizeof(uint);
    uint *scratchpad;
    cudaMalloc(&scratchpad, scratchpadSize);

    int scratchLocationsSize = stressParams.maxWorkgroups * sizeof(uint);
    uint *h_scratchLocations = (uint *)malloc(scratchLocationsSize);
    uint *d_scratchLocations;
    cudaMalloc(&d_scratchLocations, scratchLocationsSize);

    int stressParamsSize = 12 * sizeof(uint);
    uint *h_stressParams = (uint *)malloc(stressParamsSize);
    uint *d_stressParams;
    cudaMalloc(&d_stressParams, stressParamsSize);
    setStaticStressParams(h_stressParams, stressParams, testParams);

    // run iterations
    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();
    int weakBehaviors = 0;

    for (int i = 0; i < stressParams.testIterations; i++) {
        int numWorkgroups = setBetween(stressParams.testingWorkgroups, stressParams.maxWorkgroups);

	// clear memory
	cudaMemset(testLocations, 0, testLocSize);
	cudaMemset(d_testResults, 0, sizeof(TestResults));
	cudaMemset(readResults, 0, readResultsSize);
	cudaMemset(barrier, 0, barrierSize);
	cudaMemset(scratchpad, 0, scratchpadSize);
	setShuffledWorkgroups(h_shuffledWorkgroups, numWorkgroups, stressParams.shufflePct);
	cudaMemcpy(d_shuffledWorkgroups, h_shuffledWorkgroups, shuffledWorkgroupsSize, cudaMemcpyHostToDevice);
	setScratchLocations(h_scratchLocations, numWorkgroups, stressParams);
	cudaMemcpy(d_scratchLocations, h_scratchLocations, scratchLocationsSize, cudaMemcpyHostToDevice);
	setDynamicStressParams(h_stressParams, stressParams);
	cudaMemcpy(d_stressParams, h_stressParams, stressParamsSize, cudaMemcpyHostToDevice);

        litmus_test<<<numWorkgroups, stressParams.workgroupSize>>>(testLocations, readResults, d_shuffledWorkgroups, barrier, scratchpad, d_scratchLocations, d_stressParams);

	check_results<<<stressParams.testingWorkgroups, stressParams.workgroupSize>>>(testLocations, readResults, d_testResults, d_stressParams);

	cudaMemcpy(h_testResults, d_testResults, sizeof(TestResults), cudaMemcpyDeviceToHost);

	if (print_results) {
          std::cout << "Iteration " << i << "\n";
        }
        weakBehaviors += host_check_results(h_testResults, print_results);
    }
    
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = end - start;
    std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;
    std::cout << std::fixed << std::setprecision(0) << "Weak behavior rate: " << float(weakBehaviors)/duration.count() << " per second\n";

    int totalBehaviors = stressParams.testingWorkgroups * stressParams.workgroupSize * stressParams.testIterations;

    std::cout << "Weak behavior percentage: " << float(weakBehaviors)/float(totalBehaviors) * 100 << "%\n";
    std::cout << "Number of weak behaviors: " << weakBehaviors << "\n";

    // Free memory
    cudaFree(testLocations);
    cudaFree(readResults);
    cudaFree(d_testResults);
    free(h_testResults);
    cudaFree(d_shuffledWorkgroups);
    free(h_shuffledWorkgroups);
    cudaFree(barrier);
    cudaFree(scratchpad);
    cudaFree(d_scratchLocations);
    free(h_scratchLocations);
    cudaFree(d_stressParams);
    free(h_stressParams);
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

