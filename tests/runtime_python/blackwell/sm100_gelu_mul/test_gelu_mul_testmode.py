"""
Test: gelu_mul_layer via PersistentKernel test_mode.

Builds a minimal PersistentKernel containing a single gelu_mul layer
(GeGLU: gelu(gate, approximate="tanh") * up), compiles it, runs it once,
and compares the output to the PyTorch reference.

The input is a 2D tensor [num_tokens, 2 * intermediate_size] holding
[gate, up] concatenated on the last dim (same layout as silu_mul).

Run:
    python tests/runtime_python/blackwell/sm100_gelu_mul/test_gelu_mul_testmode.py
"""

import torch
import sys
import os

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pytorch_reference import gelu_mul_ref


def test_gelu_mul_testmode():
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(42)

    batch_size = 8
    intermediate_size = 2048
    fused_dim = 2 * intermediate_size

    print(f"\n{'='*60}")
    print(f"Test: gelu_mul layer (GeGLU)")
    print(f"  B={batch_size}, intermediate={intermediate_size}")

    # Input: [batch, 2*intermediate] = [gate, up]
    input_act = torch.randn(batch_size, fused_dim, dtype=dtype, device=device)
    gelu_mul_out = torch.zeros(
        batch_size, intermediate_size, dtype=dtype, device=device
    )

    # PyTorch reference
    ref = gelu_mul_ref(input_act)

    # Build PersistentKernel in test mode
    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    params["max_num_batched_tokens"] = batch_size
    params["max_num_batched_requests"] = batch_size
    pk = PersistentKernel(**params)

    input_dt = pk.attach_input(input_act, name="input")
    gelu_mul_dt = pk.attach_input(gelu_mul_out, name="gelu_mul_out")

    block_dim = (256, 1, 1) if pk.target_cc >= 90 else (128, 1, 1)
    num_tasks = 32
    assert intermediate_size % num_tasks == 0

    pk.gelu_mul_layer(
        input=input_dt,
        output=gelu_mul_dt,
        grid_dim=(num_tasks, 1, 1),
        block_dim=block_dim,
    )

    print("Compiling...")
    folder_path = os.path.dirname(__file__)
    pk.compile(output_dir=folder_path)

    print("Running...")
    pk()
    torch.cuda.synchronize()

    print(f"\ngelu_mul_out[0, :8]: {gelu_mul_out[0, :8]}")
    print(f"reference[0, :8]:    {ref[0, :8]}")

    max_diff = (gelu_mul_out.float() - ref.float()).abs().max().item()
    print(f"\nMax absolute diff: {max_diff:.6f}")

    if max_diff < 0.1:
        print("\nPASSED: gelu_mul produces correct output")
    else:
        print(f"\nFAILED: max diff {max_diff:.6f} exceeds 0.1 tolerance")
        sys.exit(1)

    pk.finalize()
    print("Test completed successfully!")


if __name__ == "__main__":
    test_gelu_mul_testmode()
