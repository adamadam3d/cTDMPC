#!/bin/bash
#SBATCH --job-name=tdmpc2-supervised
# Partition preference: L40S and RTXA6000 first (the user's primary targets,
# both 3-day partitions), then the remaining GPUs ordered fastest -> slowest.
# SLURM tries these left-to-right and runs on the first that can schedule the job.
#
# NOTE ON TIME LIMITS: L40S / RTXA6000 / A100* / V100 / RTX3090 allow up to
# 3-00:00:00, but B200 / H200* / H100* / RTXB6000 cap at 1-00:00:00. With
# --time=3-00:00:00 below the 1-day partitions will simply be skipped (the job
# can't fit), so in practice it lands on L40S/RTXA6000/A100/V100/RTX3090. If you
# want it to also use the H100/H200/B200 class, drop --time to 1-00:00:00 and
# rely on checkpoint/resume for the full 3M-step run.
#SBATCH --partition=L40S,RTXA6000,B200,H200,H200-PCI,H100,H100-PCI,A100-80GB,A100-40GB,A100-PCI,RTXB6000,V100-32GB,RTX3090
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=adam.elsayed@dfki.de
#SBATCH --time=3-00:00:00
#SBATCH -N 1
#SBATCH --mem=60G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -D /home/adelsayed/cTDMPC/tdmpc2/
#SBATCH --array=0-5

# context_encoder=supervised, seeds 3..8 (one per array task).
#   task 0 -> seed 3, task 1 -> seed 4, ... task 5 -> seed 8
ENC=supervised
PROJ=supervised
SEEDS=(3 4 5 6 7 8)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "Array task $SLURM_ARRAY_TASK_ID -> context_encoder=$ENC seed=$SEED project=$PROJ on $(hostname)"

# WandB API key (inlined per request). Replace with your real key.
export WANDB_API_KEY=PUT_YOUR_WANDB_API_KEY_HERE
if [ -z "$WANDB_API_KEY" ] || [ "$WANDB_API_KEY" = "PUT_YOUR_WANDB_API_KEY_HERE" ]; then
    echo "ERROR: WANDB_API_KEY not set — edit this sbatch and paste your key." >&2
    exit 1
fi

# Container (enroot via SLURM's pyxis plugin). The squashfs image and the data
# live under /fscratch/adelsayed; the code lives in ~/cTDMPC. Bind both in.
SQSH=/fscratch/adelsayed/tdmpc2
DATA_DIR=/fscratch/adelsayed/TDdata
CODE_DIR=/home/adelsayed/cTDMPC/tdmpc2

srun \
    --container-image=$SQSH \
    --container-mounts=/fscratch/adelsayed:/fscratch/adelsayed,/home/adelsayed:/home/adelsayed \
    --container-workdir=$CODE_DIR \
    --export=ALL,WANDB_API_KEY=$WANDB_API_KEY \
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
        data_dir=$DATA_DIR \
        context_encoder=$ENC
