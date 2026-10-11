#!/bin/bash
# Experiment queue worker, run from src_bis/:   bash mt3d/queue.sh <gpus> <port>
#   e.g. nohup bash mt3d/queue.sh 0,1 29511 &   and   nohup bash mt3d/queue.sh 2,3 29512 &
# Each worker pops the first line of ../logs/bis/queue.txt (main_mt3d.py arguments; '#' lines skipped),
# runs it on its GPUs, and waits for new lines when the queue is empty. Stop: touch ../logs/bis/queue.stop
GPUS=$1; PORT=$2
Q=../logs/bis/queue.txt; LOCK=../logs/bis/queue.lock
N=$(echo "$GPUS" | tr ',' '\n' | wc -l)
while [ ! -e ../logs/bis/queue.stop ]; do
    LINE=$(flock "$LOCK" bash -c "L=\$(grep -v '^#' $Q | grep -v '^\s*$' | head -n1); \
        if [ -n \"\$L\" ]; then grep -vxF -- \"\$L\" $Q > $Q.tmp; mv $Q.tmp $Q; fi; echo \"\$L\"")
    if [ -z "$LINE" ]; then sleep 60; continue; fi
    LOG=../logs/bis/q_$(date +%m%d_%H%M%S)_g${GPUS/,/}.log
    echo "$(date +%H:%M) [$GPUS] $LINE -> $LOG" >> ../logs/bis/queue_history.txt
    CUDA_VISIBLE_DEVICES=$GPUS ../.venv/bin/torchrun --standalone --nproc_per_node=$N --master_port=$PORT \
        main_mt3d.py $LINE > "$LOG" 2>&1
    sleep 5
done
