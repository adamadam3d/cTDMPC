#!/bin/bash
#SBATCH --job-name=tdmpc2-eval-speed
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
#SBATCH --array=15-19

# SPEED IMPROVEMENT variant of eval_taskid4_supervised_50k.sh.
#
#     sbatch slurm/hpc/eval_taskid4_supervised_50k_speedimprovement.sh
#
# Same eval logic and same combo list as the base script, but tuned for the
# 128-cpu / 515G A100 node (hpc-dnode04). The base script's measured state on
# that node was: GPU-Util 99% but VRAM ~6% and CPULoad ~18/128 -- i.e. the job
# was launch-bound and barely loading the machine. This variant packs denser
# and enables MPS by default so the node actually gets used:
#
#   * PROCS_PER_GPU defaults to 8 (was 4) -> 32 eval procs/node.
#   * USE_MPS defaults to 1 so the packed procs' kernels run concurrently
#     instead of time-slicing (only helps a launch-bound job, which this is).
#   * --cpus-per-task=28 (was 16) -> ~112 cores across 4 tasks, ~3.5/proc.
#   * --mem=120G (was 96G) -> ~480G across 4 tasks, ~15G/proc.
#   * gpu_ampere only (the fat node lives there).
#
# Everything below (sharding, checkpoint-resume via evaluated.txt, wandb
# routing, CSV output) is identical to the base script, so you can resubmit
# either after a crash and done checkpoints are skipped. Do NOT run this and
# the base script for the SAME combo concurrently -- num_shards depends on
# PROCS_PER_GPU, so different packings would evaluate overlapping checkpoints.

# One entry per combo: "<context_encoder> <seed> <wandb_project>"
COMBOS=(
    "task_id    4 eval50k"
    "supervised 7 supervised_eval"
    "supervised 4 supervised_eval"
    "supervised 5 supervised_eval"
    "supervised 3 supervised_eval"
)
GPUS_PER_SEED=4
# Denser packing default for this speed variant. Still overridable, e.g. to
# push further if CPULoad is still well under the alloc'd cores mid-job:
#   sbatch --cpus-per-task=32 --export=ALL,PROCS_PER_GPU=12 \
#          slurm/hpc/eval_taskid4_supervised_50k_speedimprovement.sh
# NOTE: num_shards depends on this value, so do not mix packings within one
# combo at the same time -- concurrent jobs with different PROCS_PER_GPU would
# evaluate overlapping checkpoint sets.
PROCS_PER_GPU=${PROCS_PER_GPU:-8}

COMBO=(${COMBOS[$(( SLURM_ARRAY_TASK_ID / GPUS_PER_SEED ))]})
ENC=${COMBO[0]}
SEED=${COMBO[1]}
PROJ=${COMBO[2]}
GPU_IDX=$(( SLURM_ARRAY_TASK_ID % GPUS_PER_SEED ))
NUM_SHARDS=$(( GPUS_PER_SEED * PROCS_PER_GPU ))

TRAIN_EXP=seed${SEED}_${ENC}_param5
CKPT_DIR=/mnt/beegfs/data/AI-REEFSHIELD/tdm/cTDMPC/tdmpc2/logs/mt30/${SEED}/${TRAIN_EXP}/models
EVAL_EXP=${PROJ}_${TRAIN_EXP}

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

# Simultaneous torch.compile runs exhaust host RAM (std::bad_alloc in
# inductor): each process spawns ~one compile worker per core by default.
# Cap the workers per process; the launch stagger below does the rest.
export SINGULARITYENV_TORCHINDUCTOR_COMPILE_THREADS=2

# CUDA MPS is ON by default: it lets the PROCS_PER_GPU processes' small kernels
# run concurrently on the GPU instead of time-slicing one context. This is the
# right tool here -- the workload is launch-bound (nvidia-smi shows gpu-util 99%
# but memory-controller ~7% and only ~4GB VRAM), so without MPS the packed procs
# serialize and 8-way packing yields only ~2.9x throughput. Disable with:
#   sbatch --export=ALL,USE_MPS=0 slurm/hpc/eval_taskid4_supervised_50k_speedimprovement.sh
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
    # so the client procs must ask MPS for id 0. Inheriting the physical
    # CUDA_VISIBLE_DEVICES (e.g. 3) makes them request a device not in MPS's
    # 1-device set -> "Invalid CUDA_VISIBLE_DEVICES" and torch.cuda.is_available()
    # returns False. The per-task pipe dir still routes to the correct GPU.
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
        python evaluate_checkpoints.py \
            task=mt30 \
            model_size=5 \
            context_encoder=$ENC \
            "seed=$SEED" \
            "checkpoint=$CKPT_DIR" \
            "checkpoint_shard=$SHARD" \
            "num_shards=$NUM_SHARDS" \
            eval_episodes=10 \
            data_dir=/mnt/beegfs/data/AI-REEFSHIELD/tdm/mt30/mt30 \
            grad_conflict_episodes=20 \
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
