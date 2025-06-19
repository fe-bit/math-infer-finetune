#!/bin/bash
# sbatch ./inference/rewoo_train_calculator_only/execute.sh
#SBATCH --job-name=MA-Inference-ReWOO-Train
#SBATCH -o ./inference/rewoo_train_calculator_only/jobs/%x.%j.out
#SBATCH -e ./inference/rewoo_train_calculator_only/jobs/%x.%j.err
#SBATCH -D ./
#SBATCH --time=15:30:00
#SBATCH --partition=AMD
#SBATCH --comment=""

export PYTHONUNBUFFERED=1

source env/bin/activate
python3 inference/rewoo_train_calculator_only/main.py
