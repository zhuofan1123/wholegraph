/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <cub/cub.cuh>
#include <cub/device/device_for.cuh>
#include <thrust/sequence.h>
#include <thrust/unique.h>

#include <wholememory/communicator.hpp>

#include "cuda_macros.hpp"
#include "logger.hpp"
#include "wholememory/integer_utils.hpp"
#include "wholememory_ops/register.hpp"
#include "wholememory_ops/temp_memory_handle.hpp"
#include <wholememory_ops/thrust_allocator.hpp>

namespace wholememory_ops {

template <typename IndexT>
__global__ void MapIndicesForHierarchyKernel(const IndexT* dev_cross_gather_recv_conti_map_ptr,
                                             const IndexT* dev_bucket_id_map,
                                             size_t indice_count,
                                             IndexT* dev_conti_id_map)
{
  for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < indice_count;
       idx += blockDim.x * gridDim.x) {
    dev_conti_id_map[idx] = dev_cross_gather_recv_conti_map_ptr[dev_bucket_id_map[idx]];
  }
}

template <typename IndexT>
__global__ void AddOffsetKernel(const IndexT* indices,
                                IndexT* output_indices,
                                int64_t indice_count,
                                int64_t offset)
{
  for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < indice_count;
       idx += blockDim.x * gridDim.x) {
    output_indices[idx] = indices[idx] + offset;
  }
}

template <typename IndexT>
void MapIndicesForHierarchyTempFunc(const void* dev_cross_gather_id_map,
                                    wholememory_array_description_t cross_gather_id_map_desc,
                                    const void* dev_bucket_id_map,
                                    wholememory_array_description_t bucket_id_map_desc,
                                    void* dev_conti_id_map,
                                    const int64_t* host_cross_gather_count,
                                    const int64_t* host_bucket_id_count,
                                    const int64_t* host_bucket_id_offset,
                                    const int64_t* host_recv_id_count,
                                    const int64_t* host_recv_id_offset,
                                    wholememory_comm_t wm_local_comm,
                                    wm_thrust_allocator* p_thrust_allocator,
                                    wholememory_env_func_t* p_env_fns,
                                    cudaStream_t stream)
{
  int local_size = -1, local_rank = -1;
  wholememory_communicator_get_size(&local_size, wm_local_comm);
  wholememory_communicator_get_rank(&local_rank, wm_local_comm);

  // add offset
  temp_memory_handle dev_cross_gather_conti_id_map_handle(p_env_fns);
  void* dev_cross_gather_conti_id_map_ptr = dev_cross_gather_conti_id_map_handle.device_malloc(
    cross_gather_id_map_desc.size, cross_gather_id_map_desc.dtype);
  // recv conti map
  temp_memory_handle dev_cross_gather_recv_conti_id_map_handle(p_env_fns);
  void* dev_cross_gather_recv_conti_id_map_ptr =
    dev_cross_gather_recv_conti_id_map_handle.device_malloc(bucket_id_map_desc.size,
                                                            bucket_id_map_desc.dtype);

  int64_t cross_gather_offset = 0;
  for (int i = 0; i < local_rank; i++) {
    cross_gather_offset += host_cross_gather_count[i];
  }

  static constexpr int BLOCK_SIZE = 128;
  int block_count = wholememory::div_rounding_up_unsafe(cross_gather_id_map_desc.size, BLOCK_SIZE);

  AddOffsetKernel<<<block_count, BLOCK_SIZE, 0, stream>>>(
    static_cast<const IndexT*>(dev_cross_gather_id_map),
    static_cast<IndexT*>(dev_cross_gather_conti_id_map_ptr),
    cross_gather_id_map_desc.size,
    cross_gather_offset);

  cudaStreamSynchronize(stream);
  wm_local_comm->alltoallv(dev_cross_gather_conti_id_map_ptr,
                           dev_cross_gather_recv_conti_id_map_ptr,
                           reinterpret_cast<const size_t*>(host_recv_id_count),
                           reinterpret_cast<const size_t*>(host_recv_id_offset),
                           reinterpret_cast<const size_t*>(host_bucket_id_count),
                           reinterpret_cast<const size_t*>(host_bucket_id_offset),
                           bucket_id_map_desc.dtype,
                           stream);
  wm_local_comm->sync_stream(stream);

  block_count = wholememory::div_rounding_up_unsafe(bucket_id_map_desc.size, BLOCK_SIZE);
  MapIndicesForHierarchyKernel<<<block_count, BLOCK_SIZE, 0, stream>>>(
    static_cast<const IndexT*>(dev_cross_gather_recv_conti_id_map_ptr),
    static_cast<const IndexT*>(dev_bucket_id_map),
    bucket_id_map_desc.size,
    static_cast<IndexT*>(dev_conti_id_map));
}

REGISTER_DISPATCH_ONE_TYPE(MapIndicesForHierarchyTempFunc, MapIndicesForHierarchyTempFunc, SINT3264)

wholememory_error_code_t map_indices_for_hierarchy(
  const void* dev_cross_gather_id_map,
  wholememory_array_description_t cross_gather_id_map_desc,
  const void* dev_bucket_id_map,
  wholememory_array_description_t bucket_id_map_desc,
  void* conti_id_map,
  int64_t* host_cross_gather_count,
  int64_t* host_bucket_id_count,
  int64_t* host_bucket_id_offset,
  int64_t* host_recv_id_count,
  int64_t* host_recv_id_offset,
  wholememory_comm_t wm_local_comm,
  wm_thrust_allocator* p_thrust_allocator,
  wholememory_env_func_t* p_env_fns,
  cudaStream_t stream)
{
  if (bucket_id_map_desc.size == 0) return WHOLEMEMORY_SUCCESS;
  try {
    DISPATCH_ONE_TYPE(bucket_id_map_desc.dtype,
                      MapIndicesForHierarchyTempFunc,
                      dev_cross_gather_id_map,
                      cross_gather_id_map_desc,
                      dev_bucket_id_map,
                      bucket_id_map_desc,
                      conti_id_map,
                      host_cross_gather_count,
                      host_bucket_id_count,
                      host_bucket_id_offset,
                      host_recv_id_count,
                      host_recv_id_offset,
                      wm_local_comm,
                      p_thrust_allocator,
                      p_env_fns,
                      stream);
    WM_CUDA_CHECK(cudaGetLastError());
  } catch (wholememory::cuda_error& wce) {
    WHOLEMEMORY_ERROR("map indices for hierarchy CUDA LOGIC Error %s\n", wce.what());
    return WHOLEMEMORY_CUDA_ERROR;
  }
  return WHOLEMEMORY_SUCCESS;
}

}  // namespace wholememory_ops
