#!/bin/bash
# sbatch ./inference/rewoo_test/execute.sh
#SBATCH --job-name=ReWOO-Test
#SBATCH -o ./inference/rewoo_test/jobs/%x.%j.out
#SBATCH -e ./inference/rewoo_test/jobs/%x.%j.err
#SBATCH -D ./
#SBATCH --time=10:30:00
#SBATCH --partition=NvidiaAll
#SBATCH --comment=""

source env/bin/activate
python3 inference/rewoo_test/main.py
