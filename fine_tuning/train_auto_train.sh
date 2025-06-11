#!/bin/bash
# sbatch ./fine_tuning/train_auto_train.sh
#SBATCH --job-name=MA-Train-Qwen-Autotrain
#SBATCH -o ./fine_tuning/jobs/train/qwen/%x.%j.out
#SBATCH -e ./fine_tuning/jobs/train/qwen/%x.%j.err
#SBATCH -D ./
#SBATCH --time=6:30:00
#SBATCH --partition=NvidiaAll
#SBATCH --nodes=2
#SBATCH --comment=""

export TMPDIR=$HOME/tmp
mkdir -p $TMPDIR

source env/bin/activate

MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n1)
MASTER_PORT=12345
export PYTHONUNBUFFERED=1

export MASTER_ADDR
export MASTER_PORT
export WORLD_SIZE=2
export RANK=$SLURM_PROCID

# accelerate launch --multi_gpu autotrain --config ./fine_tuning/my_config.yaml
# accelerate launch \
#   --multi_gpu \
#   --num_machines $SLURM_JOB_NUM_NODES \
#   --machine_rank $SLURM_PROCID \
#   --main_process_ip $(scontrol show hostname $SLURM_NODELIST | head -n1) \
#   --main_process_port 12345 \
#   autotrain --config ./fine_tuning/my_config.yaml


accelerate launch --multi_gpu --num_processes $WORLD_SIZE --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT autotrain --config ./fine_tuning/my_config.yaml
