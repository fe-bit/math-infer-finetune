#!/bin/bash
# sbatch ./inference/rewoo_train/execute.sh
#SBATCH --job-name=MA-Inference-ReWOO-Train
#SBATCH -o ./inference/rewoo_train/jobs/%x.%j.out
#SBATCH -e ./inference/rewoo_train/jobs/%x.%j.err
#SBATCH -D ./
#SBATCH --time=10:30:00
#SBATCH --partition=AMD
#SBATCH --comment=""

source env/bin/activate
python3 inference/rewoo_train/main.py
