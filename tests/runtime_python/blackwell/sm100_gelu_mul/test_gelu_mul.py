import torch
import runtime_kernel_gelu_mul

from pytorch_reference import gelu_mul_ref

torch.set_printoptions(sci_mode=False)

g = torch.Generator(device="cuda").manual_seed(1234)

output_sizes = [768, 2048]
batch_sizes = [1, 8]

for output_size in output_sizes:
    for batch_size in batch_sizes:
        print(
            f"\n=== Testing batch_size = {batch_size} output_size = {output_size} ==="
        )

        input = torch.randn(
            (batch_size, output_size * 2),
            device="cuda",
            dtype=torch.bfloat16,
            generator=g,
        )
        output = torch.empty(
            (batch_size, output_size), device="cuda", dtype=torch.bfloat16
        )

        # MPK impl
        runtime_kernel_gelu_mul.gelu_mul(input, output)

        # Reference impl: gelu(gate, approximate="tanh") * up in float32
        torch_output = gelu_mul_ref(input)

        torch.testing.assert_close(
            output,
            torch_output,
            rtol=1e-2,
            atol=1e-2,
        )
        print("Test passed!")

        # Warm-up
        for _ in range(16):
            runtime_kernel_gelu_mul.gelu_mul(input, output)

        torch.cuda.synchronize()
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        repetitions = 1000
        starter.record()
        for rep in range(repetitions):
            runtime_kernel_gelu_mul.gelu_mul(input, output)
        ender.record()
        torch.cuda.synchronize()
        total_time = starter.elapsed_time(ender)
        avg_time = total_time / repetitions
        print(f"Average time over {repetitions} runs: {avg_time:.6f} ms")
