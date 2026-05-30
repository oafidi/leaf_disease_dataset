#!/usr/bin/env bash

LOG_FILE="logs/watchdog.log"
DIR_TO_CHECK="results_pso/seed_4/trials/"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

log "Watchdog started."

while true; do
    log "Waiting for 25 minutes..."
    sleep 1500
    
    log "Killing running HPO process..."
    pkill -f pso_hpo.py
    pkill -f aco_hpo.py
    sleep 180
    pkill -9 -f pso_hpo.py 2>/dev/null
    pkill -9 -f aco_hpo.py 2>/dev/null
    log "Process killed."
    
    log "Activating virtual environment..."
    source ../bin/activate

    # 1. Vérifier si le dossier existe
    if [ ! -d "$DIR_TO_CHECK" ]; then
        log "Directory $DIR_TO_CHECK does not exist yet. Launching pso_hpo.py..."
        nohup python -u hpo/pso_hpo.py >> logs/output.log 2>&1 &
        log "pso_hpo.py started (PID=$!)."
    else
        # 2. S'il existe, on compte les fichiers (trials) à l'intérieur
        TRIAL_COUNT=$(find "$DIR_TO_CHECK" -maxdepth 1 -type f 2>/dev/null | wc -l)
        log "Trial count in $DIR_TO_CHECK: $TRIAL_COUNT"

        # 3. Logique de lancement en fonction du compteur
        if [ "$TRIAL_COUNT" -lt 100 ]; then
            log "Trial count < 100, restarting pso_hpo.py..."
            nohup python -u hpo/pso_hpo.py >> logs/output.log 2>&1 &
            log "pso_hpo.py restarted (PID=$!)."
        else
            log "Trial count >= 100, starting aco_hpo.py..."
            nohup python -u hpo/aco_hpo.py >> logs/output.log 2>&1 &
            log "aco_hpo.py started (PID=$!)."
        fi
    fi
done