#!/bin/bash
#SBATCH --job-name=tdmpc2-reefshield
#SBATCH --partition=gpu_volta,gpu_ampere
#SBATCH --account=deepl
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=adam.elsayed@dfki.de
#SBATCH --time=3-00:00:00
#SBATCH -N 1
#SBATCH --mem=49G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -D /mnt/beegfs/data/AI-REEFSHIELD/tdm/cTDMPC/tdmpc2/
#SBATCH --array=0-7

# 4 context encoders x 2 seeds (3, 4) = 8 array tasks.
# Index layout: task = SLURM_ARRAY_TASK_ID
#   encoder = ENCODERS[task / 2]
#   seed    = SEEDS[task % 2]
ENCODERS=(pearl varibad supervised task_id)
# WandB project per encoder, mirroring the original convention (task_id -> taskID)
PROJECTS=(pearl varibad supervised taskID)
SEEDS=(3 4)

ENC=${ENCODERS[$((SLURM_ARRAY_TASK_ID / 2))]}
PROJ=${PROJECTS[$((SLURM_ARRAY_TASK_ID / 2))]}
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID % 2))]}

echo "Array task $SLURM_ARRAY_TASK_ID -> context_encoder=$ENC seed=$SEED project=$PROJ"

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
        wandb_project=$PROJ \
        exp_name=seed${SEED}_${ENC}_mt30_param5 \
        wandb_entity=https-www-guc-edu-eg- \
        seed=$SEED \
        data_dir=/mnt/beegfs/data/AI-REEFSHIELD/tdm/mt30/mt30 \
        context_encoder=$ENC
