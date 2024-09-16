#define DEFINE_IDS()                                                                                           \
  uint total_ids = blockDim.x; \
  uint id_0 = threadIdx.x; \
  uint id_1 = permute_id(threadIdx.x, kernel_params->permute_thread, blockDim.x); \
  uint id_2 = permute_id(id_1, kernel_params->permute_thread, blockDim.x); \
  uint wg_offset = shuffled_workgroup * blockDim.x;

#define RESULT_IDS() \
  uint total_ids = blockDim.x; \
  uint wg_offset = blockIdx.x * blockDim.x;
