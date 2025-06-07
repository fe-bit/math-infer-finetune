#!/bin/bash
#SBATCH --job-name=MA-Evaluate
#SBATCH -o ./jobs/evaluate/%x.%j.out
#SBATCH -e ./jobs/evaluate/%x.%j.err
#SBATCH -D ./
#SBATCH --time=02:00:00
#SBATCH --partition=AMD
#SBATCH --comment=""

source env/bin/activate
python3 evaluate.py
