#!/bin/bash
#SBATCH --job-name=test-train
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:03:00
#SBATCH --mem=8G
#SBATCH --output=/SCRATCH/l26niang/bird_classifier/logs/test-train-%j.out
#SBATCH --error=/SCRATCH/l26niang/bird_classifier/logs/test-train-%j.err

set -e
cd /SCRATCH/l26niang/bird_classifier
source /SCRATCH/l26niang/bird-env/bin/activate

echo "Étape 1 : environnement"
which python

echo "Étape 2 : GPU"
nvidia-smi

echo "Étape 3 : syntaxe Python"
python -m py_compile scripts/train.py

echo "Étape 4 : démarrage du script"
python -u scripts/train.py
