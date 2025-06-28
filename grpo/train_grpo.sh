#!/bin/bash
# sbatch ./grpo/train_grpo.sh
#SBATCH --job-name=GRPO-Train
#SBATCH -o ./grpo/jobs/train/smollm/%x.%j.out
#SBATCH -e ./grpo/jobs/train/smollm/%x.%j.err
#SBATCH -D ./
#SBATCH --time=10:30:00
#SBATCH --partition=NvidiaAll
#SBATCH --comment=""

source env/bin/activate
# accelerate launch grpo/train_grpo.py
python3 grpo/main.py --model-name HuggingFaceTB/SmolLM2-135M-Instruct --first-n 50