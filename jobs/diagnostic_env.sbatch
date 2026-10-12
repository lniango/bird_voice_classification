#!/bin/bash
#SBATCH --job-name=diag-env
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:02:00
#SBATCH --output=/SCRATCH/l26niang/bird_classifier/logs/diag-env-%j.out
#SBATCH --error=/SCRATCH/l26niang/bird_classifier/logs/diag-env-%j.err


echo "=== DIAGNOSTIC ==="
date
hostname
id

echo "=== ACCES SCRATCH ==="
ls -ld /SCRATCH
ls -ld /SCRATCH/l26niang
ls -ld /SCRATCH/l26niang/bird-env
ls -l /SCRATCH/l26niang/bird-env/bin/activate

echo "=== TEST PYTHON DIRECT ==="
/SCRATCH/l26niang/bird-env/bin/python --version
echo "Code retour Python : $?"

echo "=== FIN ==="
