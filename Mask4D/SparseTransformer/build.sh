#!/usr/bin/env bash
# Clean build of the sptr CUDA extension for torch 2.x / CUDA 11.8 on Python 3.10.
#
# Usage:
#   TORCH_CUDA_ARCH_LIST="8.6" ./build.sh        # set to YOUR gpu's compute capability
#
# Override CUDA_HOME if your 11.8 toolkit lives elsewhere:
#   CUDA_HOME=/usr/local/cuda-11.8 ./build.sh
set -euo pipefail

# --- CUDA toolkit (must match torch's CUDA: cu118) --------------------------
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-11.8}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

if [ ! -x "$CUDA_HOME/bin/nvcc" ]; then
  echo "ERROR: nvcc not found at $CUDA_HOME/bin/nvcc"
  echo "       Install the CUDA 11.8 toolkit or set CUDA_HOME to it."
  exit 1
fi

# --- Host compiler ----------------------------------------------------------
# CUDA 11.8's nvcc supports gcc <= 11. Ubuntu 22.04 often defaults to gcc-12,
# which produces cryptic errors that look like they come from the kernel headers.
# Pin gcc-11/g++-11 if available (apt install gcc-11 g++-11).
if command -v gcc-11 >/dev/null 2>&1 && command -v g++-11 >/dev/null 2>&1; then
  export CC=gcc-11
  export CXX=g++-11
  echo "Using host compiler: $(gcc-11 --version | head -1)"
else
  echo "WARNING: gcc-11 not found. nvcc 11.8 supports gcc <= 11."
  echo "         If the build fails in the kernel headers, run:"
  echo "             sudo apt install gcc-11 g++-11"
  echo "         or, as a last resort, set SPTR_ALLOW_UNSUPPORTED_COMPILER=1"
fi

# --- Target GPU architecture ------------------------------------------------
# Set TORCH_CUDA_ARCH_LIST to your GPU's compute capability for a faster build,
# e.g. 6.1 (1080Ti), 7.5 (2080Ti/T4), 8.0 (A100), 8.6 (3090/A6000), 9.0 (H100).
if [ -z "${TORCH_CUDA_ARCH_LIST:-}" ]; then
  echo "WARNING: TORCH_CUDA_ARCH_LIST not set; building for a default arch set."
  export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6"
fi
echo "Building for arch(es): $TORCH_CUDA_ARCH_LIST"

# --- Sanity print -----------------------------------------------------------
nvcc --version | tail -2
python -c "import torch, sys; print('python', sys.version.split()[0]); \
print('torch', torch.__version__, 'cuda', torch.version.cuda)"

# --- Clean rebuild ----------------------------------------------------------
rm -rf build/ *.egg-info
pip install -v -e .

echo
echo "Build finished. Verify with:"
echo "    python -c 'import sptr_cuda; import sptr; print(\"sptr OK\")'"
