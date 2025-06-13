#!/bin/bash
# sbatch ./fine_tuning/train_llama.sh
#SBATCH --job-name=MA-Train-Llama
#SBATCH -o ./fine_tuning/jobs/train/llama/%x.%j.out
#SBATCH -e ./fine_tuning/jobs/train/llama/%x.%j.err
#SBATCH -D ./
#SBATCH --time=10:30:00
#SBATCH --partition=NvidiaAll
#SBATCH --comment=""

export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16
export SLURM_CPUS_PER_TASK=16

source env/bin/activate
python3 fine_tuning/train.py meta-llama/Llama-3.2-1B-Instruct --resume
