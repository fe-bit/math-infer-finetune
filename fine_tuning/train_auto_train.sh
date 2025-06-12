#!/bin/bash
# sbatch ./fine_tuning/train_auto_train.sh
#SBATCH --job-name=MA-Train-Qwen-Autotrain
#SBATCH -o ./fine_tuning/jobs/train/qwen/%x.%j.out
#SBATCH -e ./fine_tuning/jobs/train/qwen/%x.%j.err
#SBATCH -D ./
#SBATCH --time=6:30:00
#SBATCH --partition=NvidiaAll
#SBATCH --nodes=1                # Use only 1 node for now
#SBATCH --ntasks-per-node=1      # Use only 1 GPU for now
#SBATCH --cpus-per-task=8
#SBATCH --comment=""

export TMPDIR=$HOME/tmp
mkdir -p $TMPDIR

# Add aggressive memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export CUDA_LAUNCH_BLOCKING=0

source env/bin/activate

# For single GPU training, no need for distributed setup
python ./fine_tuning/train.py Qwen/Qwen2.5-0.5B-Instruct
python3 fine_tuning/evaluate.py --model-name Qwen/Qwen2.5-0.5B-Instruct