#!/bin/bash
#SBATCH --job-name=bird-mels
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/SCRATCH/l26niang/bird_classifier/logs/precompute-%j.out
#SBATCH --error=/SCRATCH/l26niang/bird_classifier/logs/precompute-%j.err
#SBATCH --nodelist=sl-mee-br-206


set -e

cd /SCRATCH/l26niang/bird_classifier
source /SCRATCH/l26niang/bird-env/bin/activate

export PYTHONPATH=src

echo "Job : $SLURM_JOB_ID"
echo "Nœud : $SLURMD_NODENAME"
echo "Début : $(date)"

python -u scripts/precompute_mels.py

echo "Fin : $(date)"
