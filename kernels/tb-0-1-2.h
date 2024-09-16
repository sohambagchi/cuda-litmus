#define DEFINE_IDS()                                                                                           \
  uint total_ids = blockDim.x * kernel_params->testing_workgroups; \
  uint id_0 = shuffled_workgroup * blockDim.x + threadIdx.x; \
  uint new_workgroup = stripe_workgroup(shuffled_workgroup, threadIdx.x, kernel_params->testing_workgroups); \
  uint id_1 = new_workgroup * blockDim.x + threadIdx.x; \
  uint third_workgroup = stripe_workgroup(new_workgroup, threadIdx.x, kernel_params->testing_workgroups); \
  uint id_2 = third_workgroup * blockDim.x + threadIdx.x; \
  uint wg_offset = 0;
