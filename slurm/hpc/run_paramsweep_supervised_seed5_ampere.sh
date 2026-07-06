#!/bin/bash
#SBATCH --job-name=tdmpc2-paramsweep
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
#SBATCH --array=0-4

# ----------------------------------------------------------------------------
# Parameter test for the supervised context encoder on MT30 (seed 5).
# Logs to a NEW WandB project so it does not pollute the existing seed sweep.
#
# Each array task runs ONE parameter configuration. Config-only changes; no
# code is modified. Knobs come from the diagnosis report:
#   - reward_coef / value_coef : reward & Q heads had the highest grad conflict
#                                yet the lowest loss weight (starves MPC planning)
#   - consistency_coef         : dominates the loss (20) -> rebalance test
#   - horizon                  : longer lookahead for sparse/hard-exploration tasks
#
#   task 0 -> baseline   (reward=0.1 value=0.1 consistency=20 horizon=3)   [reference]
#   task 1 -> rewardup   (reward=0.5 value=0.3 consistency=20 horizon=3)
#   task 2 -> horizon5   (reward=0.1 value=0.1 consistency=20 horizon=5)
#   task 3 -> combined   (reward=0.5 value=0.3 consistency=10 horizon=5)
#   task 4 -> conslow    (reward=0.1 value=0.1 consistency=10 horizon=3)
# ----------------------------------------------------------------------------
ENC=supervised
PROJ=supervised_paramsweep   # <-- NEW project (auto-created on first run)
SEED=5

NAMES=(baseline rewardup horizon5 combined conslow)
REWARD_COEFS=(0.1 0.5 0.1 0.5 0.1)
VALUE_COEFS=(0.1 0.3 0.1 0.3 0.1)
CONSISTENCY_COEFS=(20 20 20 10 10)
HORIZONS=(3 3 5 5 3)

I=$SLURM_ARRAY_TASK_ID
NAME=${NAMES[$I]}
REWARD_COEF=${REWARD_COEFS[$I]}
VALUE_COEF=${VALUE_COEFS[$I]}
CONSISTENCY_COEF=${CONSISTENCY_COEFS[$I]}
HORIZON=${HORIZONS[$I]}

EXP_NAME=seed${SEED}_${ENC}_paramsweep_${NAME}

echo "Array task $I -> $NAME | reward_coef=$REWARD_COEF value_coef=$VALUE_COEF consistency_coef=$CONSISTENCY_COEF horizon=$HORIZON | seed=$SEED project=$PROJ on $(hostname)"

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
        reward_coef=$REWARD_COEF \
        value_coef=$VALUE_COEF \
        consistency_coef=$CONSISTENCY_COEF \
        horizon=$HORIZON \
        "${CHECKPOINT_ARG[@]}"
