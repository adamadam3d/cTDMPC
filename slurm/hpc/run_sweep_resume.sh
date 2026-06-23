#!/bin/bash
#SBATCH --job-name=tdmpc2-reefshield
#SBATCH --partition=gpu_ampere,gpu_volta
#SBATCH --account=deepl
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=adam.elsayed@dfki.de
#SBATCH --time=3-00:00:00
#SBATCH -N 1
#SBATCH --mem=60G
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH -D /mnt/beegfs/data/AI-REEFSHIELD/tdm/cTDMPC/tdmpc2/
#SBATCH --array=0-7

# Explicit (encoder, seed) per array task. Encoder-first within the main grid so
# the first wave covers all encoders @ seed 3, then seed 4, then the extra
# pearl-only seeds (2 and 5).
#   task: 0        1        2      3        4        5      6      7
ENC_LIST=( varibad  task_id  pearl  varibad  task_id  pearl  pearl  pearl )
SEED_LIST=(3        3        3      4        4        4      2      5     )

ENC=${ENC_LIST[$SLURM_ARRAY_TASK_ID]}
SEED=${SEED_LIST[$SLURM_ARRAY_TASK_ID]}
# WandB project mirrors the encoder (task_id -> taskID convention).
case $ENC in
    task_id) PROJ=taskID ;;
    *)       PROJ=$ENC ;;
esac

EXP_NAME=seed${SEED}_${ENC}_param5

echo "Array task $SLURM_ARRAY_TASK_ID -> context_encoder=$ENC seed=$SEED project=$PROJ exp_name=$EXP_NAME"

# ----------------------------------------------------------------------------
# Resume logic: train.py/offline_trainer.py accepts checkpoint=<path to .pt>
# and resumes from the iteration stored inside it. Checkpoints are written to
# logs/<task>/<seed>/<exp_name>/models/<step>.pt on a save_freq cadence, plus a
# final.pt when a run completes. If a previous run with this exact exp_name
# left checkpoints behind, pick the highest-step one and resume from it instead
# of starting over.
# ----------------------------------------------------------------------------
MODEL_DIR=/mnt/beegfs/data/AI-REEFSHIELD/tdm/cTDMPC/tdmpc2/logs/mt30/${SEED}/${EXP_NAME}/models
CHECKPOINT=""
if [ -d "$MODEL_DIR" ]; then
    LATEST_STEP=$(ls "$MODEL_DIR" 2>/dev/null \
        | grep -E '^[0-9]+\.pt$' \
        | sed -E 's/\.pt$//' \
        | sort -n \
        | tail -n 1)
    if [ -n "$LATEST_STEP" ]; then
        CHECKPOINT="${MODEL_DIR}/${LATEST_STEP}.pt"
        echo "Found existing checkpoint for exp_name=$EXP_NAME -> resuming from $CHECKPOINT"
    elif [ -f "${MODEL_DIR}/final.pt" ]; then
        CHECKPOINT="${MODEL_DIR}/final.pt"
        echo "Found completed run for exp_name=$EXP_NAME -> resuming from $CHECKPOINT"
    else
        echo "Model dir exists but no checkpoint found for exp_name=$EXP_NAME -> starting fresh"
    fi
else
    echo "No prior model dir for exp_name=$EXP_NAME -> starting fresh"
fi

CHECKPOINT_ARG=()
if [ -n "$CHECKPOINT" ]; then
    CHECKPOINT_ARG=(checkpoint=$CHECKPOINT)
fi

# 1. Grab your WandB API key from ~/.netrc using only the Python stdlib
#    (avoids the host's broken wandb/platformdirs install)
export WANDB_API_KEY=$(python3 -c "import netrc; print(netrc.netrc().authenticators('api.wandb.ai')[2])")

# Fail fast if the key didn't resolve, instead of silently running unauthenticated
if [ -z "$WANDB_API_KEY" ]; then
    echo "ERROR: WANDB_API_KEY is empty — check ~/.netrc has a 'machine api.wandb.ai' entry" >&2
    exit 1
fi

# 2. Force Singularity to pass this key into the container
export SINGULARITYENV_WANDB_API_KEY=$WANDB_API_KEY

singularity exec \
    -B /mnt/beegfs/ \
    --home /mnt/beegfs/data/AI-REEFSHIELD/tdm/cTDMPC/tdmpc2/ \
    --nv \
    /mnt/beegfs/public/images/tdmpc2.sif \
    python train.py \
        task=mt30 \
        model_size=5 \
        batch_size=256 \
        steps=3000000 \
        compile=true \
        wandb_project=sweep_param5 \
        eval_episodes=10 \
        eval_freq=500000 \
        exp_name=$EXP_NAME \
        wandb_entity=https-www-guc-edu-eg- \
        seed=$SEED \
        data_dir=/mnt/beegfs/data/AI-REEFSHIELD/tdm/mt30/mt30 \
        context_encoder=$ENC \
        "${CHECKPOINT_ARG[@]}"
