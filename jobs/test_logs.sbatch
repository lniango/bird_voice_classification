#!/bin/bash
#SBATCH --job-name=test-logs
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:02:00
#SBATCH --output=/SCRATCH/l26niang/bird_classifier/logs/test-logs-%j.out
#SBATCH --error=/SCRATCH/l26niang/bird_classifier/logs/test-logs-%j.err

set -x
echo "TEST SLURM : début"
hostname
pwd
source /SCRATCH/l26niang/bird-env/bin/activate
python -c "import torch; print('PyTorch:', torch.__version__)"
echo "TEST SLURM : fin"
