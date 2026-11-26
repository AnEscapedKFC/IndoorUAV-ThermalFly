#!/usr/bin/env bash
set -eu
root_dir="$(cd "$(dirname "$0")/.." && pwd)"
upstream_dir="$root_dir/upstream/IndoorUAV-Agent/openpi"
export OPENPI_DATA_HOME="$root_dir/assets/cache"
export PYTHONPATH="$upstream_dir/src:$upstream_dir/packages/openpi-client/src"
export PYTHONDONTWRITEBYTECODE=1
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export JAX_COMPILATION_CACHE_DIR="$root_dir/env/jax-cache"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_autotune_level=0 --xla_gpu_cuda_data_dir=/usr/local/cuda-12.8}"
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
cuda_python_libs="${INDOORUAV_CUDA_LIBS:-/home/zr_fan/miniconda3/envs/gradnav_5070/lib/python3.10/site-packages/nvidia}"
export LD_LIBRARY_PATH="$cuda_python_libs/cudnn/lib:$cuda_python_libs/nccl/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
if [ "$#" -gt 0 ]; then
    exec "$root_dir/env/runtime/bin/python" "$@"
fi
exec "$root_dir/env/runtime/bin/python" -u "$root_dir/model_worker.py"
