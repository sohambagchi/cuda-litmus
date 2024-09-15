#define DEFINE_IDS()                                                                                                \
  uint total_ids = (blockDim.x * kernel_params->testing_workgroups) / 2;                                            \
  uint id_0 = shuffled_workgroup * blockDim.x + threadIdx.x;                                                        \
  uint id_0_final = id_0 % total_ids;                                                                               \
  bool id_0_first_half = id_0 / total_ids == 0;                                                                     \
  uint id_1 = shuffled_workgroup * blockDim.x + permute_id(threadIdx.x, kernel_params->permute_thread, blockDim.x); \
  uint id_1_final = id_1 % total_ids;                                                                               \
  bool id_1_first_half = id_1 / total_ids == 0;                                                                     \
  uint wg_offset = 0;