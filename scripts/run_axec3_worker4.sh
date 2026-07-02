#!/bin/bash
MAX_RETRIES=50
ATTEMPT=0
SCRIPT="axial_splitting_mldd_AXEC3_batched.py"
START=76
END=90

cd /Users/mhemmett/Seismology/axial-splitting-ml/scripts

while [ $ATTEMPT -lt $MAX_RETRIES ]; do
    ATTEMPT=$((ATTEMPT + 1))
    echo ""
    echo "============================================"
    echo "WORKER 4 (batches $START–$END) — ATTEMPT $ATTEMPT / $MAX_RETRIES  — $(date)"
    echo "============================================"

    /opt/anaconda3/envs/seismo/bin/python -u "$SCRIPT" --start-batch $START --end-batch $END 2>&1

    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo ""
        echo "SUCCESS — worker 4 completed on attempt $ATTEMPT"
        exit 0
    else
        echo ""
        echo "FAILED (exit $EXIT_CODE) — will retry in 5s..."
        sleep 5
    fi
done

echo "Exceeded $MAX_RETRIES retries, giving up."
exit 1
