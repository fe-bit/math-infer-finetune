#!/bin/bash
# sbatch ./fine_tuning/evaluate.sh
#SBATCH --job-name=MA-Evaluate
#SBATCH -o ./fine_tuning/jobs/evaluate/%x.%j.out
#SBATCH -e ./fine_tuning/jobs/evaluate/%x.%j.err
#SBATCH -D ./
#SBATCH --time=02:00:00
#SBATCH --partition=NvidiaAll
#SBATCH --comment=""

source env/bin/activate
python3 fine_tuning/evaluate.py --model-name Qwen/Qwen2.5-0.5B-Instruct
