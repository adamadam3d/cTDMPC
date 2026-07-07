#!/bin/bash
#SBATCH --job-name=tdmpc2-probe-carl
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
#SBATCH --array=0-5

# Context-recovery probe (TODO.md #3, RQ1 evidence): pairs the inferred context
# embedding z_ctx with ground-truth CARL context values.
#
#     sbatch slurm/hpc/eval_carl_probe_taskid_supervised.sh
#
# Runs `evaluate_carl.py -F --probe $PROBE_EPISODES` on the FINAL checkpoint of
# each combo: per task, N episodes (default 200), each under a freshly sampled
# context with every numeric feature perturbed INDEPENDENTLY within the task's
# physically-feasible box (mult ~ U(1-s_max, 1+s_max)) -- unlike the lockstep
# --sweep, this keeps per-dimension linear-probe R^2 attributable. One row per
# episode (raw ground-truth context + z_ctx snapshots at t=25/50/100/250 + the
# last-100-step mean) is appended to context_probe_<task>_shard<i>.csv in the
# eval work_dir; finished (checkpoint, task) pairs are tracked per shard in
# context_probe_done_shard<i>.txt so resubmissions skip done work.
#
# The task_id combos are the NEGATIVE CONTROL: their embedding is a fixed
# per-task lookup, so probe R^2 ~ 0 by construction. Run them anyway -- that
# contrast is itself a result.
#
# The 8 eval processes run wandb-free (enable_wandb=false: probe rows are wide
# tabular data, not scalar metrics); after they all finish, this array task
# uploads the combo's CSVs as ONE versioned wandb Artifact (type=dataset) in
# project carl_context_probe. The offline analysis (cross-validated per-task
# linear probe R^2, silhouette score) can read the local CSVs or the artifact.
#
# Layout: 6 combos x GPUS_PER_COMBO=1 array task = array 0-5. Each array task
# packs PROCS_PER_GPU=8 evaluate_carl.py processes onto its GPU under MPS ->
# num_shards = 8 disjoint EPISODE shards per combo (probe mode shards the
# per-task episode budget, not the checkpoint list). At the default 200
# episodes/task x 8 mt30 CARL tasks that is 200 episodes per shard.
#
# Episode budget override:
#   sbatch --export=ALL,PROBE_EPISODES=400 slurm/hpc/eval_carl_probe_taskid_supervised.sh
#
# DEPENDENCY: same as eval_carl_sweep_taskid_supervised.sh -- tdmpc2.sif does
# not ship `carl`; see that script's header for the one-time pip install.
CARL_PYTHONPATH=/mnt/beegfs/data/AI-REEFSHIELD/tdm/pip_extras

# One entry per combo: "<context_encoder> <seed>". Alternated supervised/task_id
# per seed so both encoders' data lands interleaved.
COMBOS=(
    "supervised 3"
    "task_id    3"
    "supervised 4"
    "task_id    4"
    "supervised 5"
    "task_id    5"
)
GPUS_PER_COMBO=1
PROCS_PER_GPU=${PROCS_PER_GPU:-8}
PROBE_EPISODES=${PROBE_EPISODES:-200}

COMBO=(${COMBOS[$(( SLURM_ARRAY_TASK_ID / GPUS_PER_COMBO ))]})
ENC=${COMBO[0]}
SEED=${COMBO[1]}
GPU_IDX=$(( SLURM_ARRAY_TASK_ID % GPUS_PER_COMBO ))
NUM_SHARDS=$(( GPUS_PER_COMBO * PROCS_PER_GPU ))

TRAIN_EXP=seed${SEED}_${ENC}_param5
CKPT_DIR=/mnt/beegfs/data/AI-REEFSHIELD/tdm/cTDMPC/tdmpc2/logs/mt30/${SEED}/${TRAIN_EXP}/models
EVAL_EXP=probe_carl_${TRAIN_EXP}

echo "Array task $SLURM_ARRAY_TASK_ID -> encoder=$ENC seed=$SEED probe_episodes=$PROBE_EPISODES, shards $((GPU_IDX*PROCS_PER_GPU))-$((GPU_IDX*PROCS_PER_GPU+PROCS_PER_GPU-1)) of $NUM_SHARDS"
echo "Checkpoints: $CKPT_DIR (probe uses the final one only)"

# WandB key for the post-run artifact upload only (the eval processes
# themselves run with enable_wandb=false). Same ~/.netrc mechanism as
# eval_carl_sweep_taskid_supervised.sh.
export WANDB_API_KEY=$(python3 -c "import netrc; print(netrc.netrc().authenticators('api.wandb.ai')[2])")
if [ -z "$WANDB_API_KEY" ]; then
    echo "ERROR: WANDB_API_KEY is empty — check ~/.netrc has a 'machine api.wandb.ai' entry" >&2
    exit 1
fi
export SINGULARITYENV_WANDB_API_KEY=$WANDB_API_KEY

# CARL lives outside the image (see DEPENDENCY note above). Fail fast with a
# clear message if the one-time install has not been done yet.
if [ ! -d "$CARL_PYTHONPATH/carl" ]; then
    echo "ERROR: carl not found at $CARL_PYTHONPATH — run the pip install command in eval_carl_sweep_taskid_supervised.sh's header first" >&2
    exit 1
fi
export SINGULARITYENV_PYTHONPATH=$CARL_PYTHONPATH${SINGULARITYENV_PYTHONPATH:+:$SINGULARITYENV_PYTHONPATH}

# Simultaneous torch.compile runs exhaust host RAM (std::bad_alloc in
# inductor): each process spawns ~one compile worker per core by default.
# Cap the workers per process; the launch stagger below does the rest.
export SINGULARITYENV_TORCHINDUCTOR_COMPILE_THREADS=2

# CUDA MPS is ON by default, same rationale and mechanics as
# eval_carl_sweep_taskid_supervised.sh (launch-bound workload; one daemon per
# array task, torn down in a trap). Disable with:
#   sbatch --export=ALL,USE_MPS=0 slurm/hpc/eval_carl_probe_taskid_supervised.sh
if [ "${USE_MPS:-1}" = "1" ]; then
    export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}/pipe
    export CUDA_MPS_LOG_DIRECTORY=/tmp/mps_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}/log
    mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
    nvidia-cuda-mps-control -d
    trap 'echo quit | nvidia-cuda-mps-control' EXIT
    export SINGULARITYENV_CUDA_MPS_PIPE_DIRECTORY=$CUDA_MPS_PIPE_DIRECTORY
    export SINGULARITYENV_CUDA_MPS_LOG_DIRECTORY=$CUDA_MPS_LOG_DIRECTORY
    # CRITICAL: the MPS server re-indexes its single allocated GPU to local id 0.
    export SINGULARITYENV_CUDA_VISIBLE_DEVICES=0
    MPS_BIND="-B $CUDA_MPS_PIPE_DIRECTORY -B $CUDA_MPS_LOG_DIRECTORY"
    echo "CUDA MPS enabled: $CUDA_MPS_PIPE_DIRECTORY"
else
    MPS_BIND=""
fi

for (( p=0; p<PROCS_PER_GPU; p++ )); do
    SHARD=$(( GPU_IDX * PROCS_PER_GPU + p ))
    # Stagger launches so the compile phases do not overlap: process 0 pays
    # the compile cost, later processes hit its inductor cache in /tmp.
    sleep $(( p == 0 ? 0 : 180 ))
    singularity exec \
        -B /mnt/beegfs/ \
        $MPS_BIND \
        --home /mnt/beegfs/data/AI-REEFSHIELD/tdm/cTDMPC/tdmpc2/ \
        --nv \
        /mnt/beegfs/public/images/tdmpc2.sif \
        python evaluate_carl.py -F --probe $PROBE_EPISODES \
            task=mt30 \
            model_size=5 \
            context_encoder=$ENC \
            "seed=$SEED" \
            "checkpoint=$CKPT_DIR" \
            "checkpoint_shard=$SHARD" \
            "num_shards=$NUM_SHARDS" \
            compile=true \
            cudagraphs=false \
            save_video=false \
            enable_wandb=false \
            "exp_name=$EVAL_EXP" \
            "hydra.run.dir=logs/hydra/${EVAL_EXP}/shard${SHARD}" \
        > "eval_${EVAL_EXP}_shard${SHARD}.log" 2>&1 &
done
wait

# All shards of this combo are done -> upload the combo's probe CSVs as one
# versioned wandb Artifact. Runs once per array task (= per combo), so there is
# no cross-process race on the artifact; a resubmission after a partial failure
# simply logs a new version with the now-complete file set.
WORK_DIR=logs/mt30/${SEED}/${EVAL_EXP}
singularity exec \
    -B /mnt/beegfs/ \
    --home /mnt/beegfs/data/AI-REEFSHIELD/tdm/cTDMPC/tdmpc2/ \
    /mnt/beegfs/public/images/tdmpc2.sif \
    python - <<PYEOF
import glob
import wandb

files = sorted(glob.glob('${WORK_DIR}/context_probe_*_shard*.csv'))
if not files:
    raise SystemExit('ERROR: no probe CSVs found in ${WORK_DIR} — nothing to upload')
run = wandb.init(project='carl_context_probe', entity='https-www-guc-edu-eg-',
                 name='${ENC}-seed${SEED}', job_type='probe-upload',
                 group='mt30-probe', tags=['mt30', '${ENC}', 'seed:${SEED}'])
art = wandb.Artifact(
    'context_probe_seed${SEED}_${ENC}', type='dataset',
    description='RQ1 context-recovery probe: one row per episode pairing z_ctx '
                'snapshots with ground-truth CARL context values '
                '(evaluate_carl.py --probe ${PROBE_EPISODES}, final checkpoint of ${TRAIN_EXP}).')
for f in files:
    art.add_file(f)
run.log_artifact(art)
run.finish()
print(f'Uploaded {len(files)} CSVs as artifact context_probe_seed${SEED}_${ENC}')
PYEOF

