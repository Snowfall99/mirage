"""Canonical PyTorch reference implementation for the gelu_mul (GeGLU) task.

gelu_mul takes a 2D tensor [num_tokens, 2 * intermediate_size] holding
[gate, up] concatenated on the last dim (same layout as silu_mul) and
computes:

    output[i, j] = gelu_pytorch_tanh(gate[i, j]) * up[i, j]

where gelu_pytorch_tanh matches PyTorch's gelu(x, approximate="tanh"):

    0.5 * x * (1 + tanh(0.7978845608028654 * (x + 0.044715 * x^3)))

The kernel computes the activation in float32 and casts back to bfloat16.
"""

import torch
from torch.nn import functional as F


def gelu_mul_ref(input: torch.Tensor) -> torch.Tensor:
    """Reference GeGLU: input is [..., 2 * intermediate_size] = [gate, up]."""
    output_size = input.shape[-1] // 2
    gate = input[..., :output_size].to(torch.float)
    up = input[..., output_size:].to(torch.float)
    out = F.gelu(gate, approximate="tanh") * up
    return out.to(input.dtype)
