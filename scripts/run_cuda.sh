#!/bin/bash
#SBATCH --job-name=tvbo_cuda
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --output=cuda_%j.log
#SBATCH --error=cuda_%j.err

# BIH Cluster CUDA Job Script
# ============================
# Submit with: sbatch run_cuda.sh

echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "GPU info:"
nvidia-smi

# Load modules (adjust for your cluster)
module load cuda/12.0 2>/dev/null || module load cuda
module load python/3.11 2>/dev/null || module load python

# Activate virtual environment
source ~/venvs/pycuda/bin/activate 2>/dev/null || {
    echo "Creating pycuda venv..."
    python -m venv ~/venvs/pycuda
    source ~/venvs/pycuda/bin/activate
    pip install pycuda numpy
}

# Run simulation
python run_cuda_cluster.py rwongwang_kernel.cu 68 10000

echo "Job finished at $(date)"
