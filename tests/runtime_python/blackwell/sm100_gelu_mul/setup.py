from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
import shutil
from os import path

this_dir = os.path.dirname(os.path.abspath(__file__))

nvcc_path = shutil.which("nvcc")
if nvcc_path:
    cuda_home = os.path.dirname(os.path.dirname(nvcc_path))
else:
    cuda_home = "/usr/local/cuda"

cuda_include_dir = os.path.join(cuda_home, "include")
cuda_library_dirs = [
    os.path.join(cuda_home, "lib"),
    os.path.join(cuda_home, "lib64"),
    os.path.join(cuda_home, "lib64", "stubs"),
]

macros=[("MIRAGE_BACKEND_USE_CUDA", None), ("MIRAGE_FINGERPRINT_USE_CUDA", None)]

# Pick arch flags from the local GPU: datacenter Blackwell (SM100) gets the
# MIRAGE_GRACE_BLACKWELL configuration; anything else (e.g. SM120 RTX Pro
# 6000, where the gelu_mul kernel is equally valid) builds natively.
import torch

cc = torch.cuda.get_device_properties(0)
target_cc = cc.major * 10 + cc.minor
if target_cc == 100:
    arch_nvcc_flags = [
        "-gencode=arch=compute_100a,code=sm_100a",
        "-DMIRAGE_GRACE_BLACKWELL",
    ]
    arch_cxx_flags = ["-DMIRAGE_GRACE_BLACKWELL"]
else:
    arch_nvcc_flags = ["-arch=native", f"-DMPK_TARGET_CC={target_cc}"]
    arch_cxx_flags = []

setup(
    name='runtime_kernel_gelu_mul',
    ext_modules=[
        CUDAExtension(
            name='runtime_kernel_gelu_mul',
            sources=[
                os.path.join(this_dir, 'runtime_kernel_wrapper_sm100.cu'),
            ],
            depends=[
                os.path.join(this_dir, '../../../../include/mirage/persistent_kernel/tasks/ampere/gelu_mul.cuh'),
            ],
            define_macros=macros,
            include_dirs=[
                os.path.join(this_dir, '../../../../include/mirage/persistent_kernel/'),
                os.path.join(this_dir, '../../../../include/mirage/persistent_kernel/tasks/'),
                os.path.join(this_dir, '../../../../include'),
                os.path.join(this_dir, '../../../../deps/cutlass/include'),
                os.path.join(this_dir, '../../../../deps/cutlass/tools/util/include'),
            ],
            libraries=["cuda"],
            library_dirs=cuda_library_dirs,
            extra_compile_args={
                'cxx': arch_cxx_flags,
                'nvcc': ['-O3'] + arch_nvcc_flags,
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
