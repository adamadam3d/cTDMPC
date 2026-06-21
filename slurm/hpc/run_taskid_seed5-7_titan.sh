#!/bin/bash
#SBATCH --job-name=tdmpc2-titan-taskid
#SBATCH --partition=gpu_titan
#SBATCH --account=deepl
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=adam.elsayed@dfki.de
#SBATCH --time=3-00:00:00
#SBATCH -N 1
#SBATCH --mem=60G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -D /mnt/beegfs/data/AI-REEFSHIELD/tdm/cTDMPC/tdmpc2/
#SBATCH --array=0-2

# context_encoder=task_id, seeds 5/6/7, on the Pascal Titan partition.
#   task 0 -> seed 5, task 1 -> seed 6, task 2 -> seed 7
# NOTE: Pascal (sm_61) + compile=true is the gamble here -- Triton/inductor may
# not support sm_61. If this job dies with a CUDA "no kernel image" / Triton
# compile error, re-run with compile=false (slower, but works on Pascal).
ENC=task_id
PROJ=taskID
SEEDS=(5 6 7)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "Running context_encoder=$ENC seed=$SEED project=$PROJ on $(hostname)"

# Grab the WandB API key from ~/.netrc using only the Python stdlib
export WANDB_API_KEY=$(python3 -c "import netrc; print(netrc.netrc().authenticators('api.wandb.ai')[2])")
if [ -z "$WANDB_API_KEY" ]; then
    echo "ERROR: WANDB_API_KEY is empty — check ~/.netrc has a 'machine api.wandb.ai' entry" >&2
    exit 1
fi
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
        eval_episodes=10 \
        eval_freq=500000 \
        exp_name=seed${SEED}_${ENC}_param5 \
        wandb_entity=https-www-guc-edu-eg- \
        seed=$SEED \
        data_dir=/mnt/beegfs/data/AI-REEFSHIELD/tdm/mt30/mt30 \
        context_encoder=$ENC
