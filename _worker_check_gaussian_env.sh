#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "build_rasterizer_masked" ]]; then
  echo "=== Build diff-gaussian-rasterization_masked extension ==="

  # Try activate conda env "gaussian" to match CUDA/PyTorch build.
  try_sources=(
    "/opt/conda/etc/profile.d/conda.sh"
    "/opt/miniconda3/etc/profile.d/conda.sh"
    "/usr/local/miniconda3/etc/profile.d/conda.sh"
    "/home/tiger/miniconda3/etc/profile.d/conda.sh"
    "/opt/tiger/mlx_deploy/miniconda3/etc/profile.d/conda.sh"
    "/opt/tiger/mlx_deploy/miniconda3/envs/mlx/etc/profile.d/conda.sh"
  )
  for f in "${try_sources[@]}"; do
    if [ -f "$f" ]; then
      # shellcheck disable=SC1090
      source "$f" || true
    fi
  done
  if command -v conda >/dev/null 2>&1; then
    set +e
    conda activate gaussian >/dev/null 2>&1
    set -e
  fi

  echo "python: $(command -v python)"
  python -V

  cd "/mnt/bn/aidp-data-3d-lf1/xxt/merlin/gs/workspace/GS_dev/submodules/diff-gaussian-rasterization_masked"
  python setup.py build_ext --inplace
  echo "Build done."
  exit 0
fi

echo "=== CUDA (nvcc -V) ==="
if command -v nvcc >/dev/null 2>&1; then
  nvcc -V
else
  echo "nvcc not found in PATH"
fi

echo
echo "=== GPU (nvidia-smi -L) ==="
nvidia-smi -L || true

echo
echo "=== Try locate conda ==="

try_sources=(
  "/opt/conda/etc/profile.d/conda.sh"
  "/opt/miniconda3/etc/profile.d/conda.sh"
  "/usr/local/miniconda3/etc/profile.d/conda.sh"
  "/home/tiger/miniconda3/etc/profile.d/conda.sh"
  "/opt/tiger/mlx_deploy/miniconda3/etc/profile.d/conda.sh"
  "/opt/tiger/mlx_deploy/miniconda3/envs/mlx/etc/profile.d/conda.sh"
)

for f in "${try_sources[@]}"; do
  if [ -f "$f" ]; then
    # shellcheck disable=SC1090
    source "$f" || true
  fi
done

if command -v conda >/dev/null 2>&1; then
  echo "conda: $(command -v conda)"
  conda --version || true
  echo "Available envs (first 20 lines):"
  conda info --envs | sed -n '1,20p' || true
else
  echo "conda command not available (may be installed elsewhere)"
  echo "Searching common locations for env python..."
  for p in \
    /opt/tiger/mlx_deploy/miniconda3/envs/gaussian/bin/python \
    /opt/tiger/miniconda3/envs/gaussian/bin/python \
    /opt/miniconda3/envs/gaussian/bin/python \
    /usr/local/miniconda3/envs/gaussian/bin/python \
    /home/tiger/miniconda3/envs/gaussian/bin/python
  do
    if [ -x "$p" ]; then
      echo "FOUND env python: $p"
      "$p" -V
      "$p" -m pip -V || true
      exit 0
    fi
  done
fi

echo
echo "=== Try activate env: gaussian (if conda exists) ==="
if command -v conda >/dev/null 2>&1; then
  set +e
  conda activate gaussian >/dev/null 2>&1
  rc=$?
  set -e
  if [ $rc -ne 0 ]; then
    echo "Failed to conda activate gaussian (env may not exist)"
    exit 0
  fi
  echo "CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-}"
  which python
  python -V
  python -m pip -V || true
fi
