#!/bin/bash
#SBATCH --job-name=diag-env2
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:02:00

OUT=/SCRATCH/l26niang/bird_classifier/diagnostic-resultat.txt

{
    echo "=== DIAGNOSTIC ==="
    date
    hostname
    id

    echo "=== ENVIRONNEMENT ==="
    ls -ld /SCRATCH/l26niang/bird-env
    ls -l /SCRATCH/l26niang/bird-env/bin/activate

    echo "=== PYTHON DIRECT ==="
    /SCRATCH/l26niang/bird-env/bin/python --version
    echo "Code retour Python : $?"

    echo "=== ACTIVATION ==="
    source /SCRATCH/l26niang/bird-env/bin/activate
    echo "Code retour activation : $?"
    which python
    python --version
} > "$OUT" 2>&1
