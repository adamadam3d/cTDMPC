#!/bin/bash
#SBATCH --job-name=tdmpc2-evalckpt
#SBATCH --partition=gpu_ampere,gpu_volta
#SBATCH --account=deepl
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=adam.elsayed@dfki.de
#SBATCH --time=2-00:00:00
#SBATCH -N 1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH -D /mnt/beegfs/data/AI-REEFSHIELD/tdm/cTDMPC/tdmpc2/
#SBATCH --array=0-3

# Evaluate the saved checkpoints (save_freq=50k) of ONE training run, sharded
# across the array. Eval is MuJoCo-bound and leaves the GPU mostly idle, so
# each array task packs PROCS_PER_GPU (default 4) eval processes onto its GPU;
# with --array=0-3 that is 16 shards total. All shards log into the same wandb
# group (same task/exp_name), so the curves overlay into one. Already-evaluated
# checkpoints are skipped via <work_dir>/evaluated.txt — safe to resubmit, and
# waves with different CKPT_DIR globs compose (see submit_all_evals.sh).
#
# Which run to evaluate is picked at submit time (defaults below):
#   sbatch --export=ALL,ENC=pearl,SEED=3 slurm/hpc/eval_checkpoints_array.sh
# Override TRAIN_EXP / CKPT_DIR directly for runs that don't follow the
# seed${SEED}_${ENC}_param5 naming of the sweep scripts. CKPT_DIR may be a
# directory or a glob pattern (quote it so the shell does not expand it).
#
# Sizing: one mt30 checkpoint at eval_episodes=10 is ~30 GPU-min (~15 at 5),
# so 16 shards of a full 200-checkpoint run are ~3-6h each at eval_episodes=5
# (per-process slowdown from 4-way GPU sharing eats into the ideal 4x).

ENC=${ENC:-pearl}
SEED=${SEED:-3}
TRAIN_EXP=${TRAIN_EXP:-seed${SEED}_${ENC}_param5}
CKPT_DIR=${CKPT_DIR:-/mnt/beegfs/data/AI-REEFSHIELD/tdm/cTDMPC/tdmpc2/logs/mt30/${SEED}/${TRAIN_EXP}/models}
EVAL_EPISODES=${EVAL_EPISODES:-10}
PROCS_PER_GPU=${PROCS_PER_GPU:-4}
WANDB_PROJECT=${WANDB_PROJECT:-eval50k}
# Distinct exp_name keeps the eval run (wandb group, work_dir, evaluated.txt)
# separate from the training run it evaluates.
EVAL_EXP=eval50k_${TRAIN_EXP}

TOTAL_SHARDS=$(( SLURM_ARRAY_TASK_COUNT * PROCS_PER_GPU ))
echo "Array task $SLURM_ARRAY_TASK_ID/$SLURM_ARRAY_TASK_COUNT ($PROCS_PER_GPU procs/GPU, $TOTAL_SHARDS shards) -> ${ENC} seed=${SEED}"
echo "Checkpoints: $CKPT_DIR"

# Grab the WandB API key from ~/.netrc using only the Python stdlib
# (avoids the host's broken wandb/platformdirs install)
export WANDB_API_KEY=$(python3 -c "import netrc; print(netrc.netrc().authenticators('api.wandb.ai')[2])")
if [ -z "$WANDB_API_KEY" ]; then
    echo "ERROR: WANDB_API_KEY is empty — check ~/.netrc has a 'machine api.wandb.ai' entry" >&2
    exit 1
fi
export SINGULARITYENV_WANDB_API_KEY=$WANDB_API_KEY

for (( p=0; p<PROCS_PER_GPU; p++ )); do
    SHARD=$(( SLURM_ARRAY_TASK_ID * PROCS_PER_GPU + p ))
    # Separate hydra run dirs: simultaneous launches would otherwise collide on
    # hydra's timestamped default output directory.
    singularity exec \
        -B /mnt/beegfs/ \
        --home /mnt/beegfs/data/AI-REEFSHIELD/tdm/cTDMPC/tdmpc2/ \
        --nv \
        /mnt/beegfs/public/images/tdmpc2.sif \
        python evaluate_checkpoints.py \
            task=mt30 \
            model_size=5 \
            "context_encoder=$ENC" \
            "seed=$SEED" \
            "checkpoint=$CKPT_DIR" \
            "checkpoint_shard=$SHARD" \
            "num_shards=$TOTAL_SHARDS" \
            "eval_episodes=$EVAL_EPISODES" \
            data_dir=/mnt/beegfs/data/AI-REEFSHIELD/tdm/mt30/mt30 \
            grad_conflict_episodes=20 \
            compile=true \
            save_video=false \
            "exp_name=$EVAL_EXP" \
            "wandb_project=$WANDB_PROJECT" \
            wandb_entity=https-www-guc-edu-eg- \
            "hydra.run.dir=logs/hydra/${EVAL_EXP}/shard${SHARD}" \
        > "eval_${EVAL_EXP}_shard${SHARD}.log" 2>&1 &
done
wait
