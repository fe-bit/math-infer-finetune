#!/bin/bash
# sbatch ./inference/direct_prompting_test/execute.sh
#SBATCH --job-name=MA-Inference-DP-Test
#SBATCH -o ./inference/direct_prompting_test/jobs/%x.%j.out
#SBATCH -e ./inference/direct_prompting_test/jobs/%x.%j.err
#SBATCH -D ./
#SBATCH --time=10:30:00
#SBATCH --partition=NvidiaAll
#SBATCH --comment=""

source env/bin/activate
python3 inference/direct_prompting_test/main.py