# Build:  pip install -v -e .   (or: python3 setup.py install)
# See build.sh for a wrapper that sets CUDA_HOME / host compiler / arch list.
import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from distutils.sysconfig import get_config_vars

# '-Wstrict-prototypes' is valid for C but not C++; strip it from the Python
# build flags so it doesn't error when compiling the .cpp sources. Guard against
# OPT being unset/None (happens on some Python 3.10 builds), which the original
# `(opt,) = get_config_vars('OPT')` form crashed on.
cfg_vars = get_config_vars()
opt = cfg_vars.get('OPT')
if opt:
    cleaned = " ".join(flag for flag in opt.split() if flag != '-Wstrict-prototypes')
    cfg_vars['OPT'] = cleaned
    os.environ['OPT'] = cleaned

# torch 2.x requires the extension to be built as C++17.
cxx_flags = ['-g', '-std=c++17']
nvcc_flags = ['-O2', '-std=c++17']

# Escape hatch for hosts stuck on a gcc newer than the CUDA toolkit supports
# (e.g. CUDA 11.8 + gcc-12). Prefer installing gcc-11 and setting CC/CXX; use
# this only if that isn't possible. Enable with SPTR_ALLOW_UNSUPPORTED_COMPILER=1.
if os.environ.get('SPTR_ALLOW_UNSUPPORTED_COMPILER', '0') == '1':
    nvcc_flags += ['-allow-unsupported-compiler']

# Target architectures are taken from the TORCH_CUDA_ARCH_LIST env var by
# CUDAExtension automatically; set it in your shell (e.g. "8.6") to speed builds.

setup(
    name='sptr',
    ext_modules=[
        CUDAExtension('sptr_cuda', [
            'src/sptr/pointops_api.cpp',
            'src/sptr/attention/attention_cuda.cpp',
            'src/sptr/attention/attention_cuda_kernel.cu',
            'src/sptr/precompute/precompute.cpp',
            'src/sptr/precompute/precompute_cuda_kernel.cu',
            'src/sptr/rpe/relative_pos_encoding_cuda.cpp',
            'src/sptr/rpe/relative_pos_encoding_cuda_kernel.cu',
            ],
            extra_compile_args={'cxx': cxx_flags, 'nvcc': nvcc_flags}
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
