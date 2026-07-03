#!/bin/bash
# Smoke test for the checkpoint-evaluation pipeline. Submits ONE array task
# (1 GPU, 4 packed processes, 4h limit) that evaluates just the checkpoints
# matching *000000.pt at 1M-step spacing (10 checkpoints -> 2-3 per shard)
# with eval_episodes=2, against the pearl seed-3 run by default.
#
#   bash slurm/hpc/smoke_test_eval.sh            # default pearl seed 3
#   ENC=varibad SEED=4 bash slurm/hpc/smoke_test_eval.sh
#
# What to check in the eval_*_shard*.log files afterwards:
#   1. all 4 shard processes started, loaded checkpoints, and finished
#   2. 'Loaded ... episodes (20 per task) for gradient-conflict metrics'
#      appears and the data load takes seconds, not minutes (mmap path works)
#   3. per-checkpoint wall time: << 12 min/checkpoint per process at
#      eval_episodes=2 means 4-way packing is paying off; if it is close to
#      the unpacked time, resubmit the real runs with PROCS_PER_GPU=2
#   4. the points appear in wandb project 'eval50k-smoke' with the right
#      iteration x-axis
# The smoke test uses a separate wandb project and exp_name, so it leaves no
# trace in the real eval results (and its evaluated.txt does not dedup them).

ENC=${ENC:-pearl}
SEED=${SEED:-3}
TRAIN_EXP=${TRAIN_EXP:-seed${SEED}_${ENC}_param5}
BASE=/mnt/beegfs/data/AI-REEFSHIELD/tdm/cTDMPC/tdmpc2/logs/mt30

sbatch \
    --array=0 \
    --time=04:00:00 \
    --job-name=tdmpc2-evalsmoke \
    --export=ALL,ENC=$ENC,SEED=$SEED,TRAIN_EXP=smoke_${TRAIN_EXP},CKPT_DIR="$BASE/$SEED/$TRAIN_EXP/models/*000000.pt",EVAL_EPISODES=2,PROCS_PER_GPU=4,WANDB_PROJECT=eval50k-smoke \
    "$(dirname "$0")/eval_checkpoints_array.sh"
