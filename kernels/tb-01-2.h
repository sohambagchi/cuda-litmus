#define DEFINE_IDS()                                                                                           \
  uint total_ids = blockDim.x * kernel_params->testing_workgroups; \
  uint id_0 = shuffled_workgroup * blockDim.x + threadIdx.x; \
  uint id_1 = shuffled_workgroup * blockDim.x + permute_id(threadIdx.x, kernel_params->permute_thread, blockDim.x); \
  uint new_workgroup = stripe_workgroup(shuffled_workgroup, threadIdx.x, kernel_params->testing_workgroups); \
  uint id_2 = new_workgroup * blockDim.x + permute_id(threadIdx.x, kernel_params->permute_thread, blockDim.x); \
  uint wg_offset = 0;

#define RESULT_IDS() \
  uint total_ids = blockDim.x * kernel_params->testing_workgroups; \
  uint wg_offset = 0;
