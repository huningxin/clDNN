// Copyright (c) 2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "include/fetch.cl"
#include "include/data_types.cl"

#ifdef BATCH_AXIS
    #define VALUES_NUM INPUT0_BATCH_NUM
    #define AXIS 0
#endif
#ifdef FEATURE_AXIS
    #define VALUES_NUM INPUT0_FEATURE_NUM
    #define AXIS 1
#endif
#ifdef Y_AXIS
    #define VALUES_NUM INPUT0_SIZE_Y
    #define AXIS 2
#endif
#ifdef X_AXIS
    #define VALUES_NUM INPUT0_SIZE_X
    #define AXIS 3
#endif

#ifdef MAX_OUT
    #define COMPARE_SIGN <
    #define INPUT0_FILL_VAL INPUT0_VAL_MIN
#else
    #define COMPARE_SIGN >
    #define INPUT0_FILL_VAL INPUT0_VAL_MAX
#endif

KERNEL(arg_max_min_modified)(const __global INPUT0_TYPE* input
                                  ,__global OUTPUT_TYPE* output
#ifdef SECOND_OUTPUT_EXIST
                                  ,__global INPUT1_TYPE* second_output
#endif
                            )
{
#include "include/arg_max_min_common.cl"
    iav_type result[TOP_K];
    uint output_idx = (uint)get_global_id(0);

    if (output_idx >= OPERATION_NUM)
        return;

#ifdef BATCH_AXIS
    const uint b = 0;
    #ifdef OUTPUT_LAYOUT_YXFB
    const uint y = output_idx / (INPUT0_SIZE_X * INPUT0_FEATURE_NUM); // Y
    const uint x = output_idx / INPUT0_FEATURE_NUM % INPUT0_SIZE_X; // X
    const uint f = output_idx % INPUT0_FEATURE_NUM; // F
    #else
    const uint f = output_idx / (INPUT0_SIZE_Y * INPUT0_SIZE_X); // F
    const uint y = output_idx / INPUT0_SIZE_X % INPUT0_SIZE_Y; // Y
    const uint x = output_idx % INPUT0_SIZE_X; // X
    #endif
    uint indices[] = {b, f, y, x}; // BFYX
#endif
#ifdef FEATURE_AXIS
    const uint f = 0;
    #ifdef OUTPUT_LAYOUT_YXFB
    const uint y = output_idx / (INPUT0_SIZE_X * INPUT0_BATCH_NUM); // Y
    const uint x = output_idx / INPUT0_BATCH_NUM % INPUT0_SIZE_X; // X
    const uint b = output_idx % INPUT0_BATCH_NUM; // B
    #else
    const uint b = output_idx / (INPUT0_SIZE_Y * INPUT0_SIZE_X); // B
    const uint y = output_idx / INPUT0_SIZE_X % INPUT0_SIZE_Y;  // Y
    const uint x = output_idx % INPUT0_SIZE_X;  // X
    #endif
    uint indices[] = {b, f, y, x}; // BFYX
#endif
#ifdef Y_AXIS
    const uint y = 0;
    #ifdef OUTPUT_LAYOUT_YXFB
    const uint x = output_idx / (INPUT0_FEATURE_NUM * INPUT0_BATCH_NUM); // X
    const uint f = output_idx / INPUT0_BATCH_NUM % INPUT0_FEATURE_NUM; // F
    const uint b = output_idx % INPUT0_BATCH_NUM; // B
    #else
    const uint b = output_idx / (INPUT0_FEATURE_NUM * INPUT0_SIZE_X); // B
    const uint f = output_idx / (INPUT0_SIZE_X) % INPUT0_FEATURE_NUM; // F
    const uint x = output_idx % INPUT0_SIZE_X; // X
    #endif
    uint indices[] = {b, f, y, x}; // BFYX
#endif
#ifdef X_AXIS
    const uint x = 0;
    #ifdef OUTPUT_LAYOUT_YXFB
    const uint y = output_idx / (INPUT0_FEATURE_NUM * INPUT0_BATCH_NUM); // Y
    const uint f = output_idx / INPUT0_BATCH_NUM % INPUT0_FEATURE_NUM; // F
    const uint b = output_idx % INPUT0_BATCH_NUM; // B
    #else
    const uint b = output_idx / (INPUT0_FEATURE_NUM * INPUT0_SIZE_Y); // B
    const uint f = output_idx / (INPUT0_SIZE_Y) % INPUT0_FEATURE_NUM; // F
    const uint y = output_idx % INPUT0_SIZE_Y; // Y
    #endif
    uint indices[] = {b, f, y, x}; // BFYX
#endif

    INPUT0_TYPE val = input[GET_DATA_INDEX(INPUT0, indices[0], indices[1], indices[2], indices[3])];
    result[0].index = 0;
    result[0].value = val;
    for (uint i = 0; i < VALUES_NUM; ++i) {
        indices[AXIS] = i;
        INPUT0_TYPE in_data = input[GET_DATA_INDEX(INPUT0, indices[0], indices[1], indices[2], indices[3])];
        if (val COMPARE_SIGN in_data) {
            result[0].index = i;
            result[0].value = in_data;
            val = in_data;
        }
    }

    indices[AXIS] = 0;
    output[GET_DATA_INDEX(OUTPUT, indices[0], indices[1], indices[2], indices[3])] = TO_OUTPUT_TYPE(result[0].index);
}

#undef COMPARE_SIGN
#undef INPUT0_FILL_VAL
#undef VALUES_NUM
