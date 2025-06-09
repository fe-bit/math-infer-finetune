#!/bin/bash
# sbatch ./fine_tuning/train_smollm_big.sh
#SBATCH --job-name=MA-Train-SmolLM-1.7B
#SBATCH -o ./fine_tuning/jobs/train/smollm/%x.%j.out
#SBATCH -e ./fine_tuning/jobs/train/smollm/%x.%j.err
#SBATCH -D ./
#SBATCH --time=16:30:00
#SBATCH --partition=AMD
#SBATCH --comment=""

export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16
export SLURM_CPUS_PER_TASK=16

source env/bin/activate
python3 fine_tuning/train.py HuggingFaceTB/SmolLM2-1.7B-Instruct
