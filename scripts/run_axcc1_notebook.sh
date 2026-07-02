#!/bin/bash
# Retry loop for running the AXCC1 batched splitting script.
# Re-runs on failure; file-exists checks inside the script ensure already-downloaded
# batches and completed splitting results are skipped, so each run picks up where
# it left off.

MAX_RETRIES=50
ATTEMPT=0
SCRIPT="axial_splitting_mldd_AXCC1_batched.py"

cd /Users/mhemmett/Seismology/axial-splitting-ml/scripts

while [ $ATTEMPT -lt $MAX_RETRIES ]; do
    ATTEMPT=$((ATTEMPT + 1))
    echo ""
    echo "============================================"
    echo "ATTEMPT $ATTEMPT / $MAX_RETRIES  — $(date)"
    echo "============================================"

    /opt/anaconda3/envs/seismo/bin/python -u "$SCRIPT" 2>&1

    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo ""
        echo "SUCCESS — script completed on attempt $ATTEMPT"
        exit 0
    else
        echo ""
        echo "FAILED (exit $EXIT_CODE) — will retry in 5s..."
        sleep 5
    fi
done

echo "Exceeded $MAX_RETRIES retries, giving up."
exit 1
