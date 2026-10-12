#!/bin/bash
#SBATCH --job-name=diag-env4
#SBATCH --partition=Odyssey_GPU
#SBATCH --nodelist=sl-mee-br-206
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:02:00
#SBATCH --output=/SCRATCH/l26niang/bird_classifier/logs/diag-env4-%j.out
#SBATCH --error=/SCRATCH/l26niang/bird_classifier/logs/diag-env4-%j.err

set -x
echo "Début du diagnostic"
hostname
id
ls -ld /SCRATCH/l26niang/bird-env
/SCRATCH/l26niang/bird-env/bin/python --version
echo "Fin du diagnostic"
