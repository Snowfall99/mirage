/* Copyright 2026 CMU
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
#include "ampere/gelu_mul.cuh"
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdio>

using bfloat16 = type::bfloat16_t;

// gelu_mul: output[i, j] = gelu_tanh(input[i, j]) * input[i, OUTPUT_SIZE + j]
// input:  [num_tokens, 2 * OUTPUT_SIZE]  ([gate, up] concatenated on dim 1)
// output: [num_tokens, OUTPUT_SIZE]

template <typename T, int BATCH_SIZE, int OUTPUT_SIZE>
__global__ __launch_bounds__(256) void gelu_mul_wrapper(void const *input_ptr,
                                                        void *output_ptr,
                                                        int num_tokens) {
  constexpr int INTER_SIZE = OUTPUT_SIZE * 2;
  kernel::
      gelu_mul_task_impl<T, BATCH_SIZE, OUTPUT_SIZE, INTER_SIZE, OUTPUT_SIZE>(
          input_ptr, output_ptr, num_tokens);
}

template <typename T, int BATCH_SIZE, int OUTPUT_SIZE>
void launch_gelu_mul(void const *input_ptr, void *output_ptr, int num_tokens) {
  dim3 grid_dim(1, 1, 1);
  dim3 block_dim(256, 1, 1);

  gelu_mul_wrapper<T, BATCH_SIZE, OUTPUT_SIZE>
      <<<grid_dim, block_dim>>>(input_ptr, output_ptr, num_tokens);
  cudaDeviceSynchronize();
}

void gelu_mul_kernel(torch::Tensor input, torch::Tensor output) {

  using T = bfloat16;

  void const *input_ptr = input.data_ptr();
  void *output_ptr = output.data_ptr();

  int num_tokens = static_cast<int>(input.size(0));
  int output_size = static_cast<int>(output.size(1));

  TORCH_CHECK(input.dim() == 2 && output.dim() == 2,
              "gelu_mul expects 2D input/output tensors");
  TORCH_CHECK(input.size(1) == 2 * output.size(1),
              "input dim 1 must be 2 * output dim 1");
  TORCH_CHECK(input.size(0) == output.size(0),
              "input/output must have the same number of tokens");

  if (output_size == 768) {
    launch_gelu_mul<T, 1, 768>(input_ptr, output_ptr, num_tokens);
  } else if (output_size == 2048) {
    launch_gelu_mul<T, 1, 2048>(input_ptr, output_ptr, num_tokens);
  } else if (output_size == 4096) {
    launch_gelu_mul<T, 1, 4096>(input_ptr, output_ptr, num_tokens);
  } else {
    printf("Unsupported output_size: %d\n", output_size);
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gelu_mul", &gelu_mul_kernel, "GeLU(tanh) Mul kernel");
}
