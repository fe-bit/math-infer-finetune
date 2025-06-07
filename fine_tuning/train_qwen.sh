#!/bin/bash
# sbatch ./fine_tuning/train_qwen.sh
#SBATCH --job-name=MA-Train-Qwen
#SBATCH -o ./fine_tuning/jobs/train/%x.%j.out
#SBATCH -e ./fine_tuning/jobs/train/%x.%j.err
#SBATCH -D ./
#SBATCH --time=10:30:00
#SBATCH --partition=AMD
#SBATCH --comment=""

export OMP_NUM_THREADS=16
export SLURM_CPUS_PER_TASK=16

source env/bin/activate
python3 fine_tuning/train_qwen.py
