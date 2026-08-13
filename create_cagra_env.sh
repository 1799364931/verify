#!/usr/bin/env bash
# Create the Conda environment required by the CAGRA verification scripts.
set -euo pipefail

ENV_NAME="${1:-cagra-bench}"
# RTX 5090 (Blackwell, compute capability 12.0) needs CUDA 12.8 or newer.
# cuVS 26.06 officially publishes Conda packages for CUDA 12.9.
CUDA_VERSION="${CUDA_VERSION:-12.9}"

if ! command -v conda >/dev/null 2>&1; then
  echo "error: conda is not available in PATH" >&2
  exit 2
fi
if conda env list | awk 'NR > 2 {print $1}' | grep -Fxq "$ENV_NAME"; then
  echo "error: Conda environment '$ENV_NAME' already exists; choose a new name" >&2
  exit 2
fi

conda create --yes --name "$ENV_NAME" --override-channels \
  --channel rapidsai --channel conda-forge --channel nvidia \
  "python=3.11" "cuvs=26.06" "cupy" "numpy" "cuda-version=${CUDA_VERSION}"

conda run --name "$ENV_NAME" python - <<'PY'
import cupy as cp
import cuvs
from cuvs.neighbors import cagra
from cuvs.neighbors.mg import cagra as mg_cagra

print("cuvs", cuvs.__version__)
print("single_gpu_cagra", cagra.IndexParams)
print("multi_gpu_cagra", mg_cagra.IndexParams)
try:
    count = cp.cuda.runtime.getDeviceCount()
    print("visible_gpus", count)
    for device in range(count):
        props = cp.cuda.runtime.getDeviceProperties(device)
        name = props["name"]
        name = name.decode() if isinstance(name, bytes) else str(name)
        print(f"gpu_{device}", name, f"sm_{props['major']}{props['minor']}")
except cp.cuda.runtime.CUDARuntimeError as exc:
    print("visible_gpus unavailable:", exc)
PY

echo "created environment '$ENV_NAME'"
echo "run: conda run -n $ENV_NAME python build_graph.py --config cagra_config.json"
