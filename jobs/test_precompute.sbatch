#!/bin/bash
#SBATCH --job-name=test-mels
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --output=/SCRATCH/l26niang/bird_classifier/logs/test-mels-%j.out
#SBATCH --error=/SCRATCH/l26niang/bird_classifier/logs/test-mels-%j.err

echo "ETAPE 1 : démarrage"
hostname
date

echo "ETAPE 2 : accès au projet"
cd /SCRATCH/l26niang/bird_classifier || exit 10
pwd

echo "ETAPE 2-2 : identité et accès"
id
ls -ld /SCRATCH /SCRATCH/l26niang /SCRATCH/l26niang/bird-env
ls -l /SCRATCH/l26niang/bird-env/bin/activate

echo "ETAPE 3 : activation de Python"
source /SCRATCH/l26niang/bird-env/bin/activate || exit 11


echo "ETAPE 3 : activation de Python"
source /SCRATCH/l26niang/bird-env/bin/activate || exit 11
which python
python --version

echo "ETAPE 4 : vérification des dépendances"
python -c "import torch; print('Torch:', torch.__version__)"
python -c "import torchaudio; print('Torchaudio:', torchaudio.__version__)"
python -c "import tqdm; print('tqdm OK')"

echo "ETAPE 5 : fin du test"
