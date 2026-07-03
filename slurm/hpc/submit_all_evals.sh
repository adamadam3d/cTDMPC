#!/bin/bash
# Submit checkpoint-evaluation arrays for all thesis runs, in coarse-to-fine
# waves. Wave 1 gives every run a complete curve at 200k resolution first;
# waves 2 and 3 densify to 100k and then the full 50k grid. Each run's waves
# are chained with --dependency=afterany so the coarse pass always finishes
# before finer passes start — if the queue underdelivers before the deadline,
# every run still has a uniform curve at the density reached. evaluated.txt
# dedups across waves, so overlapping resubmissions are safe.
#
#   bash slurm/hpc/submit_all_evals.sh
#
# EDIT RUNS below to match the (encoder, seed) pairs going into the thesis.
# Runs are submitted in list order — put the most important seed of each
# encoder first so wave 1 covers all encoders as early as possible.

RUNS=(
    "task_id:3"
    "task_id:5"
    "task_id:6"
    "task_id:7"
    "task_id:8"
)

# Filename patterns for the waves: multiples of 200k end in <even digit>00000,
# the remaining 100k multiples in <odd digit>00000, and the 50k offsets in
# 50000. final.pt matches none of them — evaluate it in the high-episode
# final pass instead.
WAVES=(
    '*[02468]00000.pt'
    '*[13579]00000.pt'
    '*50000.pt'
)

BASE=/mnt/beegfs/data/AI-REEFSHIELD/tdm/cTDMPC/tdmpc2/logs/mt30
EVAL_EPISODES=${EVAL_EPISODES:-5}
SCRIPT=$(dirname "$0")/eval_checkpoints_array.sh

declare -A PREV_JOB
for w in "${!WAVES[@]}"; do
    for run in "${RUNS[@]}"; do
        ENC=${run%%:*}
        SEED=${run##*:}
        TRAIN_EXP=seed${SEED}_${ENC}_param5
        CKPT_DIR="$BASE/$SEED/$TRAIN_EXP/models/${WAVES[$w]}"
        DEP=""
        if [ -n "${PREV_JOB[$run]}" ]; then
            DEP="--dependency=afterany:${PREV_JOB[$run]}"
        fi
        JOB=$(sbatch --parsable $DEP \
            --export=ALL,ENC=$ENC,SEED=$SEED,TRAIN_EXP=$TRAIN_EXP,CKPT_DIR="$CKPT_DIR",EVAL_EPISODES=$EVAL_EPISODES \
            "$SCRIPT")
        PREV_JOB[$run]=$JOB
        echo "wave$((w+1)) ${ENC} seed${SEED}: job $JOB ${DEP:+(depends on ${DEP#--dependency=afterany:})}"
    done
done
