#!/bin/bash
#SBATCH --job-name=tdmpc2-eval50k
#SBATCH --partition=gpu_ampere,gpu_volta
#SBATCH --account=deepl
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=adam.elsayed@dfki.de
#SBATCH --time=2-00:00:00
#SBATCH -N 1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH -D /mnt/beegfs/data/AI-REEFSHIELD/tdm/cTDMPC/tdmpc2/
#SBATCH --array=0-19

# Evaluate every 50k checkpoint of the task_id runs, seeds 3/5/6/7/8, in one
# submission:
#
#     sbatch slurm/hpc/eval_taskid_50k.sh
#
# 20 array tasks = 5 seeds x 4 GPUs. Each task takes one GPU and packs 4 eval
# processes onto it (eval is MuJoCo-bound, the GPU is mostly idle), so each
# seed's ~200 checkpoints are split into 16 disjoint shards of ~13.
#
# Metrics go to wandb (project eval50k, one group per seed, x-axis iteration)
# AND to logs/mt30/<seed>/eval50k_seed<seed>_task_id_param5/metrics_shard*.csv.
# Finished checkpoints are recorded in evaluated.txt next to the CSVs, so if
# anything crashes or hits the time limit, just resubmit the same command --
# done work is skipped.
#
# Sizing: ~30 min/checkpoint unpacked at eval_episodes=10; with 4-way packing
# expect each shard to finish in well under a day.

SEEDS=(3 5 6 7 8)
GPUS_PER_SEED=4
# Overridable at submit time for denser packing on big nodes, e.g. on the
# 128-cpu/500G A100 node:
#   sbatch --partition=gpu_ampere --cpus-per-task=32 --mem=120G \
#          --export=ALL,PROCS_PER_GPU=8 slurm/hpc/eval_taskid_50k.sh
# (CLI flags override the #SBATCH directives above.) NOTE: num_shards depends
# on this value, so do not mix packings within one seed at the same time --
# concurrent jobs with different PROCS_PER_GPU would evaluate overlapping
# checkpoint sets.
PROCS_PER_GPU=${PROCS_PER_GPU:-4}

SEED=${SEEDS[$(( SLURM_ARRAY_TASK_ID / GPUS_PER_SEED ))]}
GPU_IDX=$(( SLURM_ARRAY_TASK_ID % GPUS_PER_SEED ))
NUM_SHARDS=$(( GPUS_PER_SEED * PROCS_PER_GPU ))

TRAIN_EXP=seed${SEED}_task_id_param5
CKPT_DIR=/mnt/beegfs/data/AI-REEFSHIELD/tdm/cTDMPC/tdmpc2/logs/mt30/${SEED}/${TRAIN_EXP}/models
EVAL_EXP=eval50k_${TRAIN_EXP}

echo "Array task $SLURM_ARRAY_TASK_ID -> seed=$SEED, shards $((GPU_IDX*PROCS_PER_GPU))-$((GPU_IDX*PROCS_PER_GPU+PROCS_PER_GPU-1)) of $NUM_SHARDS"
echo "Checkpoints: $CKPT_DIR"

# Grab the WandB API key from ~/.netrc using only the Python stdlib
# (avoids the host's broken wandb/platformdirs install)
export WANDB_API_KEY=$(python3 -c "import netrc; print(netrc.netrc().authenticators('api.wandb.ai')[2])")
if [ -z "$WANDB_API_KEY" ]; then
    echo "ERROR: WANDB_API_KEY is empty — check ~/.netrc has a 'machine api.wandb.ai' entry" >&2
    exit 1
fi
export SINGULARITYENV_WANDB_API_KEY=$WANDB_API_KEY

# Four simultaneous torch.compile runs exhaust host RAM (std::bad_alloc in
# inductor): each process spawns ~one compile worker per core by default.
# Cap the workers per process; the launch stagger below does the rest.
export SINGULARITYENV_TORCHINDUCTOR_COMPILE_THREADS=2

# Optional: CUDA MPS lets the PROCS_PER_GPU processes' kernels run
# concurrently on the GPU instead of time-slicing it. Only worth it if the
# GPU is launch-bound (system.gpu.*.gpu ~100% but system.gpu.*.memory in the
# single digits in wandb's System tab, as observed here) -- MPS cannot help
# a workload that is already compute-bound. Opt in with:
#   sbatch --export=ALL,USE_MPS=1 slurm/hpc/eval_taskid_50k.sh
# The control daemon runs on the host (outside the container); its pipe/log
# directories are bind-mounted and exported into the container so the client
# processes inside singularity can find it. One daemon per array task (i.e.
# per GPU, since --gres=gpu:1), torn down in a trap so it always quits even
# if the eval loop below fails partway through.
if [ "${USE_MPS:-0}" = "1" ]; then
    export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}/pipe
    export CUDA_MPS_LOG_DIRECTORY=/tmp/mps_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}/log
    mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
    nvidia-cuda-mps-control -d
    trap 'echo quit | nvidia-cuda-mps-control' EXIT
    export SINGULARITYENV_CUDA_MPS_PIPE_DIRECTORY=$CUDA_MPS_PIPE_DIRECTORY
    export SINGULARITYENV_CUDA_MPS_LOG_DIRECTORY=$CUDA_MPS_LOG_DIRECTORY
    MPS_BIND="-B $CUDA_MPS_PIPE_DIRECTORY -B $CUDA_MPS_LOG_DIRECTORY"
    echo "CUDA MPS enabled: $CUDA_MPS_PIPE_DIRECTORY"
else
    MPS_BIND=""
fi

for (( p=0; p<PROCS_PER_GPU; p++ )); do
    SHARD=$(( GPU_IDX * PROCS_PER_GPU + p ))
    # Stagger launches so the compile phases do not overlap: process 0 pays
    # the compile cost, later processes hit its inductor cache in /tmp
    # (node-local, shared by all 4 processes) and start almost warm.
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
            context_encoder=task_id \
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
            wandb_project=eval50k \
            wandb_entity=https-www-guc-edu-eg- \
            "hydra.run.dir=logs/hydra/${EVAL_EXP}/shard${SHARD}" \
        > "eval_${EVAL_EXP}_shard${SHARD}.log" 2>&1 &
done
wait
