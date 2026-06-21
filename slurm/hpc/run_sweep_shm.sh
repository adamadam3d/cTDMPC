#!/bin/bash
#SBATCH --job-name=tdmpc2-reefshield
#SBATCH --partition=gpu_ampere,gpu_volta
#SBATCH --account=deepl
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=adam.elsayed@dfki.de
#SBATCH --time=3-00:00:00
#SBATCH -N 1
# NOTE: tmpfs (/dev/shm) usage is charged to the job's memory cgroup on most
# SLURM setups. We stage the dataset (~tens of GB) into /dev/shm AND still load
# it into the per-process buffer, so --mem must cover BOTH. Bumped well above the
# 48G baseline. If tasks OOM at staging, raise this further.
#SBATCH --mem=120G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -D /mnt/beegfs/data/AI-REEFSHIELD/tdm/cTDMPC/tdmpc2/
#SBATCH --array=0-7

# ============================================================================
# /dev/shm dataset staging prototype
# ----------------------------------------------------------------------------
# WHAT THIS DOES: copies the mt30 *.pt files from beegfs into a node-local
# RAM-backed tmpfs (/dev/shm) ONCE per node, then points data_dir there.
#
# WHY: when several array tasks start on the same node, they otherwise each read
# the whole dataset from beegfs concurrently -> network contention + slow start.
# Staging once into RAM makes every task on that node load at memory speed.
#
# WHAT IT DOES NOT DO: it does NOT reduce per-process RAM. Each run still copies
# the data into its own LazyTensorStorage (buffer.py is unchanged). The shared
# /dev/shm copy is ON TOP of the per-process copies. This is a load-speed
# prototype / stepping stone, not the packing/RAM fix.
# ============================================================================

SRC_DIR=/mnt/beegfs/data/AI-REEFSHIELD/tdm/mt30/mt30
# Per-user shared path so all array tasks on a node hit the SAME files (one copy
# in the page cache). Keep it stable across tasks of this sweep.
SHM_DIR=/dev/shm/${USER}_mt30
LOCK=/dev/shm/${USER}_mt30.lock
DONE_MARKER=${SHM_DIR}/.stage_done

# 4 context encoders x 2 seeds (3, 4) = 8 array tasks.
ENCODERS=(pearl varibad supervised task_id)
PROJECTS=(pearl varibad supervised taskID)
SEEDS=(3 4)

ENC=${ENCODERS[$((SLURM_ARRAY_TASK_ID / 2))]}
PROJ=${PROJECTS[$((SLURM_ARRAY_TASK_ID / 2))]}
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID % 2))]}

echo "Array task $SLURM_ARRAY_TASK_ID -> context_encoder=$ENC seed=$SEED project=$PROJ on $(hostname)"

# ----------------------------------------------------------------------------
# Stage dataset into /dev/shm exactly once per node, coordinated with flock so
# concurrent array tasks on the same node don't all copy at the same time.
# ----------------------------------------------------------------------------
stage_dataset() {
    mkdir -p "$SHM_DIR"
    # Serialize staging across same-node tasks; the lock is held only for the
    # check + copy, not for the whole training run.
    exec {lockfd}>"$LOCK"
    flock "$lockfd"

    if [ -f "$DONE_MARKER" ]; then
        echo "Dataset already staged in $SHM_DIR (found .stage_done) — skipping copy."
        flock -u "$lockfd"
        return 0
    fi

    # Size guard: make sure /dev/shm has room before copying.
    local need_kb avail_kb
    need_kb=$(du -sk "$SRC_DIR" | awk '{print $1}')
    avail_kb=$(df -k /dev/shm | awk 'NR==2 {print $4}')
    echo "Dataset size: $((need_kb/1024)) MB; /dev/shm free: $((avail_kb/1024)) MB"
    if [ "$need_kb" -ge "$avail_kb" ]; then
        echo "ERROR: not enough space in /dev/shm to stage dataset — falling back to beegfs." >&2
        flock -u "$lockfd"
        return 1
    fi

    echo "Staging $SRC_DIR -> $SHM_DIR ..."
    cp "$SRC_DIR"/*.pt "$SHM_DIR"/ && touch "$DONE_MARKER"
    local rc=$?
    flock -u "$lockfd"
    return $rc
}

DATA_DIR=$SRC_DIR
if stage_dataset; then
    DATA_DIR=$SHM_DIR
    echo "Using staged data_dir=$DATA_DIR"
else
    echo "Staging failed/skipped — using beegfs data_dir=$DATA_DIR"
fi

# 1. Grab your WandB API key from ~/.netrc using only the Python stdlib
export WANDB_API_KEY=$(python3 -c "import netrc; print(netrc.netrc().authenticators('api.wandb.ai')[2])")
if [ -z "$WANDB_API_KEY" ]; then
    echo "ERROR: WANDB_API_KEY is empty — check ~/.netrc has a 'machine api.wandb.ai' entry" >&2
    exit 1
fi
export SINGULARITYENV_WANDB_API_KEY=$WANDB_API_KEY

# Bind /dev/shm into the container so the staged files are visible inside it.
singularity exec \
    -B /mnt/beegfs/ \
    -B /dev/shm/ \
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
        eval_episodes=5 \
        exp_name=seed${SEED}_${ENC}_param5 \
        wandb_entity=https-www-guc-edu-eg- \
        seed=$SEED \
        data_dir=$DATA_DIR \
        context_encoder=$ENC

# ----------------------------------------------------------------------------
# Cleanup note: /dev/shm is NOT auto-cleared and persists until reboot. We do
# NOT delete here because sibling array tasks on the same node may still be
# using the staged copy. After the whole array finishes, reclaim the RAM with:
#     srun -w <node> rm -rf /dev/shm/${USER}_mt30 /dev/shm/${USER}_mt30.lock
# (run once per node that the array touched). Check nodes via: sacct -j <jobid>
# ----------------------------------------------------------------------------
