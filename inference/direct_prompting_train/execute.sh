#!/bin/bash
# sbatch ./inference/direct_prompting_train/execute.sh
#SBATCH --job-name=MA-Inference-DP-Train
#SBATCH -o ./inference/direct_prompting_train/jobs/%x.%j.out
#SBATCH -e ./inference/direct_prompting_train/jobs/%x.%j.err
#SBATCH -D ./
#SBATCH --time=10:30:00
#SBATCH --partition=AMD
#SBATCH --comment=""

export PYTHONUNBUFFERED=1
source env/bin/activate
python3 inference/direct_prompting_train/main.py
