#!/bin/bash
# Monitors system memory pressure. If memory usage exceeds THRESHOLD,
# kills the highest-memory AXEC3 worker to prevent system crash.

THRESHOLD=90        # % memory used to trigger a kill
CHECK_INTERVAL=30   # seconds between checks
LOG="/Users/mhemmett/Seismology/axial-splitting-ml/scripts/memory_monitor.log"

echo "[$(date)] Memory monitor started (threshold: ${THRESHOLD}%)" | tee -a "$LOG"

while true; do
    # macOS: use vm_stat + sysctl to get memory pressure %
    TOTAL_MEM=$(sysctl -n hw.memsize)
    TOTAL_PAGES=$(sysctl -n hw.physmem 2>/dev/null || echo $TOTAL_MEM)

    # Use vm_stat to get free + inactive pages
    VM=$(vm_stat)
    PAGE_SIZE=$(echo "$VM" | grep "page size" | awk '{print $8}')
    FREE_PAGES=$(echo "$VM" | grep "^Pages free" | awk '{print $3}' | tr -d '.')
    INACTIVE_PAGES=$(echo "$VM" | grep "^Pages inactive" | awk '{print $3}' | tr -d '.')
    SPECULATIVE_PAGES=$(echo "$VM" | grep "^Pages speculative" | awk '{print $3}' | tr -d '.')

    FREE_BYTES=$(( (FREE_PAGES + INACTIVE_PAGES + SPECULATIVE_PAGES) * PAGE_SIZE ))
    TOTAL_BYTES=$(sysctl -n hw.memsize)
    USED_BYTES=$(( TOTAL_BYTES - FREE_BYTES ))
    MEM_PCT=$(( USED_BYTES * 100 / TOTAL_BYTES ))

    # Also check for memory pressure via memory_pressure command
    PRESSURE=$(memory_pressure 2>/dev/null | grep "System memory pressure" | awk '{print $NF}')

    echo "[$(date)] Memory used: ${MEM_PCT}% | Pressure: ${PRESSURE:-unknown}" | tee -a "$LOG"

    if [ "$MEM_PCT" -ge "$THRESHOLD" ] || [ "$PRESSURE" = "Critical" ]; then
        echo "[$(date)] WARNING: Memory critical (${MEM_PCT}%, pressure=${PRESSURE}) — killing highest-memory worker" | tee -a "$LOG"

        # Find the highest-memory axec3 worker process
        VICTIM_PID=$(ps aux | grep "axial_splitting_mldd_AXEC3_batched.py" | grep -v grep \
            | sort -k4 -rn | head -1 | awk '{print $2}')

        if [ -n "$VICTIM_PID" ]; then
            VICTIM_INFO=$(ps aux | awk -v pid="$VICTIM_PID" '$2==pid {print $0}')
            echo "[$(date)] Killing PID $VICTIM_PID: $VICTIM_INFO" | tee -a "$LOG"
            kill "$VICTIM_PID"
        else
            echo "[$(date)] No AXEC3 worker found to kill" | tee -a "$LOG"
        fi
    fi

    sleep "$CHECK_INTERVAL"
done
