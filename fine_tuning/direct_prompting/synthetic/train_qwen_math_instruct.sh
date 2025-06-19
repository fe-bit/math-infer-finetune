#!/bin/bash
# sbatch ./fine_tuning/direct_prompting/synthetic/train_qwen_math_instruct.sh
#SBATCH --job-name=Qwen2.5:1.5B-Math-Instruct-Synthetic-Train
#SBATCH -o ./fine_tuning/direct_prompting/synthetic/jobs/train/Qwen/Qwen2.5-Math-1.5B-Instruct/%x.%j.out
#SBATCH -e ./fine_tuning/direct_prompting/synthetic/jobs/train/Qwen/Qwen2.5-Math-1.5B-Instruct/%x.%j.err
#SBATCH -D ./
#SBATCH --time=10:30:00
#SBATCH --partition=NvidiaAll
#SBATCH --comment=""

export TMPDIR=$HOME/tmp
mkdir -p $TMPDIR

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export PYTHONUNBUFFERED=1

source env/bin/activate

python3 ./fine_tuning/direct_prompting/synthetic/train_advanced.py Qwen/Qwen2.5-Math-1.5B-Instruct --quantized
python3 fine_tuning/direct_prompting/synthetic/evaluate.py --model-name Qwen/Qwen2.5-Math-1.5B-Instruct --first-n 50 --with-peft --quantized