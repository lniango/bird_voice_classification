#!/bin/bash
#SBATCH --job-name=diag-env3
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:02:00
#SBATCH --output=/SCRATCH/l26niang/bird_classifier/logs/diag-env3-%j.out
#SBATCH --error=/SCRATCH/l26niang/bird_classifier/logs/diag-env3-%j.err

set -x
echo "Début du diagnostic"
hostname
id
ls -ld /SCRATCH/l26niang/bird-env
/SCRATCH/l26niang/bird-env/bin/python --version
echo "Fin du diagnostic"
