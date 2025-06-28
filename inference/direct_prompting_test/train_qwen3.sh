#!/bin/bash
# sbatch ./inference/direct_prompting_test/train_qwen3.sh
#SBATCH --job-name=Qwen3-ReWOO-Train
#SBATCH -o ./inference/direct_prompting_test/jobs/train/qwen3/%x.%j.out
#SBATCH -e ./inference/direct_prompting_test/jobs/train/qwen3/%x.%j.err
#SBATCH -D ./
#SBATCH --time=10:30:00
#SBATCH --partition=NvidiaAll
#SBATCH --comment=""

# Add aggressive memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export CUDA_LAUNCH_BLOCKING=0
export PYTHONUNBUFFERED=1

lscpu

source env/bin/activate

python3 ./inference/direct_prompting_test/train.py Qwen/Qwen3-0.6B --quantized
python3 inference/direct_prompting_test/main.py --model-name Qwen/Qwen3-0.6B --first-n 50