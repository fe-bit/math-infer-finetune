#!/bin/bash
# sbatch ./inference/direct_prompting_test_gsm8k_training/train_qwen.sh
#SBATCH --job-name=Qwen-ReWOO-Train
#SBATCH -o ./inference/direct_prompting_test_gsm8k_training/jobs/train/qwen/%x.%j.out
#SBATCH -e ./inference/direct_prompting_test_gsm8k_training/jobs/train/qwen/%x.%j.err
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

python3 ./inference/direct_prompting_test_gsm8k_training/train.py Qwen/Qwen2.5-0.5B-Instruct
python3 inference/direct_prompting_test_gsm8k_training/main.py --model-name Qwen/Qwen2.5-0.5B-Instruct --first-n 50