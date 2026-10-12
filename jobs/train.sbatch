#!/bin/bash
#SBATCH --job-name=bird-cnn
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --nodelist=sl-mee-br-206
#SBATCH --output=/SCRATCH/l26niang/bird_classifier/logs/bird-cnn-%j.out
#SBATCH --error=/SCRATCH/l26niang/bird_classifier/logs/bird-cnn-%j.err

set -e

cd /SCRATCH/l26niang/bird_classifier

source /SCRATCH/l26niang/bird-env/bin/activate
export PYTHONPATH=src
export CUBLAS_WORKSPACE_CONFIG=:4096:8

echo "Job ID : $SLURM_JOB_ID"
echo "Nœud : $SLURMD_NODENAME"
echo "Debut : $(date)"
nvidia-smi

python -u scripts/train.py

echo "Fin: $(date)"
