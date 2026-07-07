#!/bin/bash
#SBATCH --job-name=tdmpc2-eval-carl
#SBATCH --partition=gpu_ampere
#SBATCH --account=deepl
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=adam.elsayed@dfki.de
#SBATCH --time=2-00:00:00
#SBATCH -N 1
#SBATCH --mem=120G
#SBATCH --cpus-per-task=28
#SBATCH --gres=gpu:1
#SBATCH -D /mnt/beegfs/data/AI-REEFSHIELD/tdm/cTDMPC/tdmpc2/
#SBATCH --array=0-23

# CARL generalization eval over ALL checkpoints of each training run, tuned for
# throughput like eval_taskid4_supervised_50k_speedimprovement.sh: dense
# process packing + CUDA MPS + checkpoint sharding inside evaluate_carl.py.
#
#     sbatch slurm/hpc/eval_carl_sweep_taskid_supervised.sh
#
# Runs `evaluate_carl.py -B -F --sweep` on every checkpoint in each combo's
# models/ dir: a magnitude sweep (auto:5 -> baseline to the physically-extreme
# CARL context bound in 5 steps) over the FULL (-F) set of CARL-wrappable
# dm_control tasks in the mt30 training set. Per-checkpoint metrics are logged
# to wandb at `iteration` (evaluate_checkpoints.py convention) and appended to
# carl_metrics_shard<i>.csv; finished checkpoints are tracked in
# carl_evaluated.txt so crashed/resubmitted jobs skip done work.
#
#   * task_id     seeds 3,4,5 -> wandb project taskid_generalizable
#   * supervised  seeds 3,4,5 -> wandb project supervised_generalizable
#
# Layout: 6 combos x GPUS_PER_COMBO=4 array tasks (1 GPU each) = array 0-23.
# Each array task packs PROCS_PER_GPU=8 evaluate_carl.py processes onto its GPU
# under MPS -> num_shards = 4*8 = 32 disjoint checkpoint shards per combo. The
# workload is launch-bound (small kernels, MuJoCo stepping on CPU), so MPS-packed
# processes give near-linear throughput; see the speedimprovement script header
# for the measurements behind these numbers.
#
# NOTE: num_shards depends on PROCS_PER_GPU and GPUS_PER_COMBO -- do NOT run two
# submissions with different packings for the SAME combo concurrently, or the
# shards would evaluate overlapping checkpoint sets.
#
# DEPENDENCY: tdmpc2.sif does NOT ship the `carl` package. It is installed once
# to the beegfs dir below and injected via PYTHONPATH. To (re)install:
#   singularity exec -B /mnt/beegfs/ \
#       --home /mnt/beegfs/data/AI-REEFSHIELD/tdm/cTDMPC/tdmpc2/ \
#       /mnt/beegfs/public/images/tdmpc2.sif \
#       python -m pip install --no-deps --target=/mnt/beegfs/data/AI-REEFSHIELD/tdm/pip_extras carl-bench
#   (the -B bind is required -- without it the --target path is read-only)
# (--no-deps on purpose: the container's dm_control/gymnasium/mujoco pins must
# not be upgraded; install any genuinely-missing dep the same way, one by one.)
CARL_PYTHONPATH=/mnt/beegfs/data/AI-REEFSHIELD/tdm/pip_extras

# One entry per combo: "<context_encoder> <seed> <wandb_project>"
COMBOS=(
    "task_id    3 taskid_generalizable"
    "task_id    4 taskid_generalizable"
    "task_id    5 taskid_generalizable"
    "supervised 3 supervised_generalizable"
    "supervised 4 supervised_generalizable"
    "supervised 5 supervised_generalizable"
)
GPUS_PER_COMBO=4
# Overridable, e.g. to push further if CPULoad is still well under the alloc'd
# cores mid-job:
#   sbatch --cpus-per-task=32 --export=ALL,PROCS_PER_GPU=12 \
#          slurm/hpc/eval_carl_sweep_taskid_supervised.sh
PROCS_PER_GPU=${PROCS_PER_GPU:-8}

COMBO=(${COMBOS[$(( SLURM_ARRAY_TASK_ID / GPUS_PER_COMBO ))]})
ENC=${COMBO[0]}
SEED=${COMBO[1]}
PROJ=${COMBO[2]}
GPU_IDX=$(( SLURM_ARRAY_TASK_ID % GPUS_PER_COMBO ))
NUM_SHARDS=$(( GPUS_PER_COMBO * PROCS_PER_GPU ))

TRAIN_EXP=seed${SEED}_${ENC}_param5
CKPT_DIR=/mnt/beegfs/data/AI-REEFSHIELD/tdm/cTDMPC/tdmpc2/logs/mt30/${SEED}/${TRAIN_EXP}/models
EVAL_EXP=carl_${PROJ}_${TRAIN_EXP}

echo "Array task $SLURM_ARRAY_TASK_ID -> encoder=$ENC seed=$SEED project=$PROJ, shards $((GPU_IDX*PROCS_PER_GPU))-$((GPU_IDX*PROCS_PER_GPU+PROCS_PER_GPU-1)) of $NUM_SHARDS"
echo "Checkpoints: $CKPT_DIR"

# Grab the WandB API key from ~/.netrc using only the Python stdlib
# (avoids the host's broken wandb/platformdirs install)
export WANDB_API_KEY=$(python3 -c "import netrc; print(netrc.netrc().authenticators('api.wandb.ai')[2])")
if [ -z "$WANDB_API_KEY" ]; then
    echo "ERROR: WANDB_API_KEY is empty — check ~/.netrc has a 'machine api.wandb.ai' entry" >&2
    exit 1
fi
export SINGULARITYENV_WANDB_API_KEY=$WANDB_API_KEY

# CARL lives outside the image (see DEPENDENCY note above). Fail fast with a
# clear message if the one-time install has not been done yet.
if [ ! -d "$CARL_PYTHONPATH/carl" ]; then
    echo "ERROR: carl not found at $CARL_PYTHONPATH — run the pip install command in this script's header first" >&2
    exit 1
fi
export SINGULARITYENV_PYTHONPATH=$CARL_PYTHONPATH${SINGULARITYENV_PYTHONPATH:+:$SINGULARITYENV_PYTHONPATH}

# Simultaneous torch.compile runs exhaust host RAM (std::bad_alloc in
# inductor): each process spawns ~one compile worker per core by default.
# Cap the workers per process; the launch stagger below does the rest.
export SINGULARITYENV_TORCHINDUCTOR_COMPILE_THREADS=2

# CUDA MPS is ON by default: it lets the PROCS_PER_GPU processes' small kernels
# run concurrently on the GPU instead of time-slicing one context -- the right
# tool for this launch-bound workload. Disable with:
#   sbatch --export=ALL,USE_MPS=0 slurm/hpc/eval_carl_sweep_taskid_supervised.sh
# The control daemon runs on the host (outside the container); its pipe/log
# directories are bind-mounted and exported into the container so the client
# processes inside singularity can find it. One daemon per array task (i.e. per
# GPU, since --gres=gpu:1), torn down in a trap so it always quits even if the
# eval loop below fails partway through.
if [ "${USE_MPS:-1}" = "1" ]; then
    export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}/pipe
    export CUDA_MPS_LOG_DIRECTORY=/tmp/mps_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}/log
    mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
    # Daemon keeps the SLURM-allocated physical GPU (CUDA_VISIBLE_DEVICES as set
    # by SLURM) so the MPS server attaches to the right device.
    nvidia-cuda-mps-control -d
    trap 'echo quit | nvidia-cuda-mps-control' EXIT
    export SINGULARITYENV_CUDA_MPS_PIPE_DIRECTORY=$CUDA_MPS_PIPE_DIRECTORY
    export SINGULARITYENV_CUDA_MPS_LOG_DIRECTORY=$CUDA_MPS_LOG_DIRECTORY
    # CRITICAL: the MPS server re-indexes its single allocated GPU to local id 0,
    # so the client procs must ask MPS for id 0 (see speedimprovement script).
    export SINGULARITYENV_CUDA_VISIBLE_DEVICES=0
    MPS_BIND="-B $CUDA_MPS_PIPE_DIRECTORY -B $CUDA_MPS_LOG_DIRECTORY"
    echo "CUDA MPS enabled: $CUDA_MPS_PIPE_DIRECTORY"
else
    MPS_BIND=""
fi

for (( p=0; p<PROCS_PER_GPU; p++ )); do
    SHARD=$(( GPU_IDX * PROCS_PER_GPU + p ))
    # Stagger launches so the compile phases do not overlap: process 0 pays
    # the compile cost, later processes hit its inductor cache in /tmp
    # (node-local, shared by all processes) and start almost warm.
    sleep $(( p == 0 ? 0 : 180 ))
    # Separate hydra run dirs: simultaneous launches would otherwise collide on
    # hydra's timestamped default output directory.
    singularity exec \
        -B /mnt/beegfs/ \
        $MPS_BIND \
        --home /mnt/beegfs/data/AI-REEFSHIELD/tdm/cTDMPC/tdmpc2/ \
        --nv \
        /mnt/beegfs/public/images/tdmpc2.sif \
        python evaluate_carl.py -B -F --sweep \
            task=mt30 \
            model_size=5 \
            context_encoder=$ENC \
            "seed=$SEED" \
            "checkpoint=$CKPT_DIR" \
            "checkpoint_shard=$SHARD" \
            "num_shards=$NUM_SHARDS" \
            eval_episodes=10 \
            compile=true \
            cudagraphs=false \
            save_video=false \
            "exp_name=$EVAL_EXP" \
            wandb_project=$PROJ \
            wandb_entity=https-www-guc-edu-eg- \
            "hydra.run.dir=logs/hydra/${EVAL_EXP}/shard${SHARD}" \
        > "eval_${EVAL_EXP}_shard${SHARD}.log" 2>&1 &
done
wait
