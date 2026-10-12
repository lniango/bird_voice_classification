#!/bin/bash
#SBATCH --job-name=diag-env5
#SBATCH --partition=Odyssey_GPU
#SBATCH --nodelist=sl-mee-br-206
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:03:00
#SBATCH --output=/SCRATCH/l26niang/bird_classifier/logs/diag-env5-%j.out
#SBATCH --error=/SCRATCH/l26niang/bird_classifier/logs/diag-env5-%j.err

set -e
cd /SCRATCH/l26niang/bird_classifier

echo "=== Activation de l'environnement ==="
source /SCRATCH/l26niang/bird-env/bin/activate
which python
python --version

echo "=== PyTorch ==="
python -c "import torch; print(torch.__version__); print('CUDA disponible :', torch.cuda.is_available())"

echo "=== Torchaudio ==="
python -c "import torchaudio; print(torchaudio.__version__)"

echo "=== TQDM ==="
python -c "import tqdm; print('tqdm OK')"

echo "=== Fin du diagnostic ==="
