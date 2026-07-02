#!/usr/bin/env bash
# Environment setup for axial-splitting-ml
#
# Creates (or reuses) the "seismo" conda environment, installs all pinned
# dependencies from requirements.txt via pip, registers a Jupyter kernel,
# and verifies the vendored modified SWSPy in swspy/ imports.
#
# Usage:   bash env.sh
# Then:    conda activate seismo
#
# Re-running is safe — the conda env is reused if it already exists, and
# `pip install -r requirements.txt` is a no-op for already-satisfied pins.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${CONDA_ENV_NAME:-seismo}"
PY_VERSION="${PYTHON_VERSION:-3.11}"

echo "==> Repository: $REPO_DIR"

# 1. Verify conda
if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found on PATH. Install Miniconda/Anaconda first." >&2
  exit 1
fi
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
echo "==> Using conda at $CONDA_BASE"

# 2. Create the conda environment if missing
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "==> Reusing existing conda environment '$ENV_NAME'"
else
  echo "==> Creating conda environment '$ENV_NAME' (python=$PY_VERSION)"
  conda create -n "$ENV_NAME" "python=$PY_VERSION" -y
fi

conda activate "$ENV_NAME"
echo "==> Activated '$ENV_NAME' ($(python -c 'import sys; print(sys.version.split()[0])'))"

# 3. Upgrade installer toolchain
echo "==> Upgrading pip / setuptools / wheel"
python -m pip install --quiet --upgrade pip setuptools wheel

# 4. Install requirements
if [ -f "$REPO_DIR/requirements.txt" ]; then
  echo "==> Installing requirements.txt"
  python -m pip install --quiet -r "$REPO_DIR/requirements.txt"
else
  echo "WARNING: requirements.txt not found; skipping pip install" >&2
fi

# 5. Vendored SWSPy is loaded via sys.path.insert(0, 'swspy') in the notebooks.
#    No separate install needed — but verify it imports.
echo "==> Verifying vendored SWSPy import"
( cd "$REPO_DIR" && python -c "
import sys, os
sys.path.insert(0, os.path.join('$REPO_DIR', 'swspy'))
import swspy
print('   SWSPy OK at', swspy.__file__)
" ) || {
  echo "WARNING: vendored swspy/ failed to import — check the submodule is initialised" >&2
  echo "         Try: git submodule update --init --recursive" >&2
}

# 6. Register a Jupyter kernel for this environment (idempotent)
if python -c "import ipykernel" >/dev/null 2>&1; then
  echo "==> Registering Jupyter kernel '$ENV_NAME'"
  python -m ipykernel install --user --name "$ENV_NAME" --display-name "Python ($ENV_NAME)" >/dev/null
fi

echo
echo "==> Done."
echo "    Activate with:   conda activate $ENV_NAME"
echo "    Launch Jupyter:  jupyter notebook"
echo "    In a notebook, select kernel: Python ($ENV_NAME)"
echo
echo "    Note: pylith_axial/ requires a separate PyLith install, activated"
echo "    via pylith_axial/activate_pylith.sh — not managed by this script."
