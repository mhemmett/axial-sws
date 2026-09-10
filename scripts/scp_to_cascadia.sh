#!/bin/bash
set -e
cd "$(dirname "$0")/.."

REMOTE="mhemmett@cascadia.ess.washington.edu"
PORT=7777
DEST="~/axial-splitting-ml"

echo "== 1/4: swspy/swspy/ =="
rsync -avz -e "ssh -p${PORT}" swspy/swspy/ "${REMOTE}:${DEST}/swspy/swspy/"

echo "== 2/4: swspy/swspy_mfast_adapt/ =="
rsync -avz -e "ssh -p${PORT}" swspy/swspy_mfast_adapt/ "${REMOTE}:${DEST}/swspy/swspy_mfast_adapt/"

echo "== 3/4: swspy top-level files =="
scp -P${PORT} swspy/setup.py swspy/requirements.txt swspy/README.md swspy/LICENSE swspy/pytest.ini \
  "${REMOTE}:${DEST}/swspy/"

echo "== 4/4: scripts + velocity model =="
scp -P${PORT} scripts/splitting_functions.py scripts/baillard_velocity.py scripts/pykonal_raytracer.py scripts/teanby_clustering.py \
  "${REMOTE}:${DEST}/scripts/"
scp -P${PORT} data/AXIAL_MODEL_3P_VELOCITY.S.mod.buf data/AXIAL_MODEL_3P_VELOCITY.S.mod.hdr \
  "${REMOTE}:${DEST}/data/"

echo "Done."
