#!/bin/bash
#SBATCH --job-name=mps-sanity
#SBATCH --partition=gpu_ampere
#SBATCH --account=deepl
#SBATCH --time=00:15:00
#SBATCH -N 1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -D /mnt/beegfs/data/AI-REEFSHIELD/tdm/cTDMPC/tdmpc2/

# Minimal MPS handshake test. Does NOT run eval -- it only answers:
#   "can a process inside the container see CUDA while MPS is active?"
# Iterate on this (15 min, 1 GPU) instead of the real array.
#
#   sbatch slurm/hpc/mps_sanity.sh ; tail -f slurm-<jobid>.out
#
# It runs the CUDA check THREE times so you can localise the break:
#   [1] host,      no MPS   -> baseline: proves the GPU is visible at all
#   [2] container, no MPS   -> proves singularity --nv works (this already
#                              works in your real runs)
#   [3] container, with MPS -> the combination that failed. If only this one
#                              prints False, MPS is the culprit (as suspected).

SIF=/mnt/beegfs/public/images/tdmpc2.sif
CHECK='import torch; print("cuda.is_available =", torch.cuda.is_available(), "| device_count =", torch.cuda.device_count())'

echo "=================================================================="
echo "SLURM_JOB_ID=$SLURM_JOB_ID  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi -L
echo "=================================================================="

echo "[1] HOST, no MPS -------------------------------------------------"
python3 -c "$CHECK" 2>&1 || echo "  (host has no torch -- skip, not fatal)"

echo "[2] CONTAINER, no MPS --------------------------------------------"
singularity exec -B /mnt/beegfs/ --nv "$SIF" python -c "$CHECK"

echo "[3] CONTAINER, WITH MPS ------------------------------------------"
# Start the MPS control daemon on the host. It inherits CUDA_VISIBLE_DEVICES
# from this shell (SLURM set it to the allocated GPU), which is what the MPS
# server needs to attach to the right device.
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_${SLURM_JOB_ID}/pipe
export CUDA_MPS_LOG_DIRECTORY=/tmp/mps_${SLURM_JOB_ID}/log
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"

echo "  starting daemon (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES) ..."
nvidia-cuda-mps-control -d
trap 'echo quit | nvidia-cuda-mps-control 2>/dev/null' EXIT

# Prove the daemon is actually alive and responding BEFORE launching a client.
# If this errors, the daemon never came up -> that is your bug, not the client.
sleep 2
echo "  daemon ping: $(echo get_default_active_thread_percentage | nvidia-cuda-mps-control 2>&1)"

# Export the pipe/log dirs into the container and bind-mount them so the
# client inside singularity can find the server the host daemon spawned.
export SINGULARITYENV_CUDA_MPS_PIPE_DIRECTORY=$CUDA_MPS_PIPE_DIRECTORY
export SINGULARITYENV_CUDA_MPS_LOG_DIRECTORY=$CUDA_MPS_LOG_DIRECTORY
singularity exec \
    -B /mnt/beegfs/ \
    -B "$CUDA_MPS_PIPE_DIRECTORY" \
    -B "$CUDA_MPS_LOG_DIRECTORY" \
    --nv "$SIF" python -c "$CHECK"

echo "=================================================================="
echo "MPS server log (errors here explain a [3]=False):"
cat "$CUDA_MPS_LOG_DIRECTORY"/server.log 2>/dev/null || echo "  (no server.log -- server never started)"
echo "MPS control log:"
cat "$CUDA_MPS_LOG_DIRECTORY"/control.log 2>/dev/null || echo "  (no control.log)"
echo "=================================================================="
