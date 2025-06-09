#!/bin/bash
# sbatch ./fine_tuning/train_qwen.sh
#SBATCH --job-name=MA-Train-Qwen
#SBATCH -o ./fine_tuning/jobs/train/%x.%j.out
#SBATCH -e ./fine_tuning/jobs/train/%x.%j.err
#SBATCH -D ./
#SBATCH --time=6:30:00
#SBATCH --partition=NvidiaAll
#SBATCH --comment=""

export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32
export NUMEXPR_NUM_THREADS=32
export SLURM_CPUS_PER_TASK=32

source env/bin/activate
python3 fine_tuning/train.py Qwen/Qwen2.5-0.5B-Instruct
