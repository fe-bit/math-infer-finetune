#!/bin/bash
# sbatch ./fine_tuning/train.sh
#SBATCH --job-name=MA-Train
#SBATCH -o ./fine_tuning/jobs/train/%x.%j.out
#SBATCH -e ./fine_tuning/jobs/train/%x.%j.err
#SBATCH -D ./
#SBATCH --time=3:30:00
#SBATCH --partition=All
#SBATCH --comment=""

export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16
export SLURM_CPUS_PER_TASK=16

source env/bin/activate
python3 fine_tuning/train.py
