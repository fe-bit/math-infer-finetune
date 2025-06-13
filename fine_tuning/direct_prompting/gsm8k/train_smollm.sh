#!/bin/bash
# sbatch ./fine_tuning/direct_prompting/gsm8k/train_smollm.sh
#SBATCH --job-name=SmolLm-GSM8K-Train
#SBATCH -o ./fine_tuning/direct_prompting/gsm8k/jobs/train/smollm/%x.%j.out
#SBATCH -e ./fine_tuning/direct_prompting/gsm8k/jobs/train/smollm/%x.%j.err
#SBATCH -D ./
#SBATCH --time=10:30:00
#SBATCH --partition=NvidiaAll
#SBATCH --comment=""

export TMPDIR=$HOME/tmp
mkdir -p $TMPDIR

# Add aggressive memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export CUDA_LAUNCH_BLOCKING=0
export PYTHONUNBUFFERED=1

source env/bin/activate

# For single GPU training, no need for distributed setup
python3 ./fine_tuning/direct_prompting/gsm8k/train.py HuggingFaceTB/SmolLM2-360M-Instruct
python3 fine_tuning/direct_prompting/gsm8k/evaluate.py --model-name HuggingFaceTB/SmolLM2-360M-Instruct --first-n 50 --with-peft