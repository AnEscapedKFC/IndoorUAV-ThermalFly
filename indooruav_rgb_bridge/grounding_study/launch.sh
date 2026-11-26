#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MAX_JOBS=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false TRANSFORMERS_OFFLINE=1
exec .venv/bin/python -uB "$@"
