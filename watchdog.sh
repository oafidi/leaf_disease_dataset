#!/usr/bin/env bash
LOG_FILE="logs/watchdog.log"
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}
log "Watchdog started."
while true; do
    log "Waiting for 25 minutes..."
    sleep 1500
    log "Killing running HPO process..."
    pkill -f ga_hpo.py
    pkill -f de_hpo.py
    sleep 180
    pkill -9 -f ga_hpo.py 2>/dev/null
    pkill -9 -f de_hpo.py 2>/dev/null
    log "Process killed."
    log "Activating virtual environment..."
    source ../bin/activate

    TRIAL_COUNT=$(ls results_ga/seed_4/trials/ 2>/dev/null | wc -l)
    log "Trial count in results_ga/seed_4/trials/: $TRIAL_COUNT"

    if [ "$TRIAL_COUNT" -ge 100 ]; then
        log "Trial count >= 100, restarting de_hpo.py..."
        nohup python -u hpo/de_hpo.py >> logs/output.log 2>&1 &
        log "de_hpo.py restarted (PID=$!)."
    else
        log "Trial count < 100, restarting ga_hpo.py..."
        nohup python -u hpo/ga_hpo.py >> logs/output.log 2>&1 &
        log "ga_hpo.py restarted (PID=$!)."
    fi
done