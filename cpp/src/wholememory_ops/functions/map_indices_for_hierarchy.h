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
#pragma once

#include <wholememory/tensor_description.h>
#include <wholememory/wholememory.h>

#include <wholememory_ops/temp_memory_handle.hpp>
#include <wholememory_ops/thrust_allocator.hpp>

namespace wholememory_ops {

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
  cudaStream_t stream);

}  // namespace wholememory_ops
