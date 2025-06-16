#!/bin/bash
# sbatch ./fine_tuning/direct_prompting/synthetic/train_smollm1.7B.sh
#SBATCH --job-name=SmolLM2-1.7B-Synthetic-Train
#SBATCH -o ./fine_tuning/direct_prompting/synthetic/jobs/train/HuggingFaceTB/SmolLM2-1.7B-Instruct/%x.%j.out
#SBATCH -e ./fine_tuning/direct_prompting/synthetic/jobs/train/HuggingFaceTB/SmolLM2-1.7B-Instruct/%x.%j.err
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

python3 ./fine_tuning/direct_prompting/synthetic/train.py HuggingFaceTB/SmolLM2-1.7B-Instruct --quantized
python3 fine_tuning/direct_prompting/synthetic/evaluate.py --model-name HuggingFaceTB/SmolLM2-1.7B-Instruct --first-n 50 --with-peft --quantized