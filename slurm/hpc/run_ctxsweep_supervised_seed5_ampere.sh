#!/bin/bash
#SBATCH --job-name=tdmpc2-ctxsweep
#SBATCH --partition=gpu_ampere
#SBATCH --account=deepl
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=adam.elsayed@dfki.de
#SBATCH --time=3-00:00:00
#SBATCH -N 1
#SBATCH --mem=60G
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH -D /mnt/beegfs/data/AI-REEFSHIELD/tdm/cTDMPC/tdmpc2/
#SBATCH --array=0-6

# ----------------------------------------------------------------------------
# Context-encoder ablation sweep for the supervised encoder on MT30 (seed 5).
# Thesis E4: one-factor-at-a-time around the baseline config already covered
# by run_supervised_seed3-4-5-7-8_ampere.sh (exp_name=seed5_supervised_param5:
# context_window=100, task_dim=96, context_loss=ce, context_coef=1.0) -- that
# run IS the baseline/reference point for this sweep and is intentionally not
# repeated here. Logs to a NEW WandB project so it does not pollute the
# existing seed sweep or the reward/value/consistency/horizon paramsweep.
#
# Ablated knobs (thesis Section 4.2 / 5.3):
#   - context_loss   : ce (default, classification) vs. nce (supervised InfoNCE)
#   - context_window : K, length of the online context FIFO
#   - task_dim       : d_c, the context-embedding dimension
#   - context_coef   : alpha, weight of the context loss
#
#   task 0 -> nce        (context_loss=nce,  everything else baseline)
#   task 1 -> window25   (context_window=25,  everything else baseline)
#   task 2 -> window200  (context_window=200, everything else baseline)
#   task 3 -> dim32      (task_dim=32,        everything else baseline)
#   task 4 -> dim192     (task_dim=192,       everything else baseline)
#   task 5 -> coeflow    (context_coef=0.1,   everything else baseline)
#   task 6 -> coefhigh   (context_coef=10.0,  everything else baseline)
# ----------------------------------------------------------------------------
ENC=supervised
PROJ=supervised_ctxsweep   # <-- NEW project (auto-created on first run)
SEED=5

NAMES=(nce window25 window200 dim32 dim192 coeflow coefhigh)
CONTEXT_LOSSES=(nce ce ce ce ce ce ce)
CONTEXT_WINDOWS=(100 25 200 100 100 100 100)
TASK_DIMS=(96 96 96 32 192 96 96)
CONTEXT_COEFS=(1.0 1.0 1.0 1.0 1.0 0.1 10.0)

I=$SLURM_ARRAY_TASK_ID
NAME=${NAMES[$I]}
CONTEXT_LOSS=${CONTEXT_LOSSES[$I]}
CONTEXT_WINDOW=${CONTEXT_WINDOWS[$I]}
TASK_DIM=${TASK_DIMS[$I]}
CONTEXT_COEF=${CONTEXT_COEFS[$I]}

EXP_NAME=seed${SEED}_${ENC}_ctxsweep_${NAME}

echo "Array task $I -> $NAME | context_loss=$CONTEXT_LOSS context_window=$CONTEXT_WINDOW task_dim=$TASK_DIM context_coef=$CONTEXT_COEF | seed=$SEED project=$PROJ on $(hostname)"

# ----------------------------------------------------------------------------
# Resume logic: train.py accepts checkpoint=<path to .pt> and resumes from the
# iteration stored inside it. Checkpoints live in
# logs/<task>/<seed>/<exp_name>/models/<step>.pt (+ final.pt when done). If a
# previous run with this exact exp_name left checkpoints, resume from the
# highest-step one instead of restarting.
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

# Grab the WandB API key from ~/.netrc using only the Python stdlib
# (avoids the host's broken wandb/platformdirs install)
export WANDB_API_KEY=$(python3 -c "import netrc; print(netrc.netrc().authenticators('api.wandb.ai')[2])")
if [ -z "$WANDB_API_KEY" ]; then
    echo "ERROR: WANDB_API_KEY is empty — check ~/.netrc has a 'machine api.wandb.ai' entry" >&2
    exit 1
fi
export SINGULARITYENV_WANDB_API_KEY=$WANDB_API_KEY

# ----------------------------------------------------------------------------
# Thread Limiting Logic: Prevent CPU oversubscription on shared nodes
# Restricts math libraries to the exact number of allocated CPUs per task
# ----------------------------------------------------------------------------
export SINGULARITYENV_OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export SINGULARITYENV_MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export SINGULARITYENV_OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export SINGULARITYENV_NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

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
        exp_name=$EXP_NAME \
        wandb_entity=https-www-guc-edu-eg- \
        seed=$SEED \
        data_dir=/mnt/beegfs/data/AI-REEFSHIELD/tdm/mt30/mt30 \
        context_encoder=$ENC \
        context_loss=$CONTEXT_LOSS \
        context_window=$CONTEXT_WINDOW \
        task_dim=$TASK_DIM \
        context_coef=$CONTEXT_COEF \
        "${CHECKPOINT_ARG[@]}"
