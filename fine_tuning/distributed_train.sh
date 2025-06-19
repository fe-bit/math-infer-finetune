#!/bin/bash
# sbatch ./fine_tuning/distributed_train.sh
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=DDP-Llama
#SBATCH -o ./fine_tuning/jobs/train/meta-llama/%x.%j.out
#SBATCH -e ./fine_tuning/jobs/train/meta-llama/%x.%j.err
#SBATCH -D ./
#SBATCH --time=6:30:00
#SBATCH --partition=NvidiaAll
#SBATCH --comment=""

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export CUDA_LAUNCH_BLOCKING=0

source env/bin/activate

# ...existing code...

# Get the list of nodes allocated to this job
nodes=($(scontrol show hostnames $SLURM_JOB_NODELIST))
MASTER_HOSTNAME=${nodes[0]}
MASTER_IP=$(getent hosts ${MASTER_HOSTNAME} | awk '{ print $1 }')

echo "Master node hostname: $MASTER_HOSTNAME"
echo "Master node IP: $MASTER_IP"
echo "All nodes: ${nodes[@]}"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "Node rank: $SLURM_NODEID"
echo "Task ID: $SLURM_PROCID"

# Add debugging info
echo "CUDA devices available: $(nvidia-smi -L | wc -l)"
echo "Current directory: $(pwd)"
echo "Python path: $(which python3)"
echo "Hostname: $(hostname)"

# Only on master node, write both config files
if [ $SLURM_NODEID -eq 0 ]; then
  echo "Writing accelerate configs with main_process_ip: $MASTER_IP"

  cat <<EOF > ./fine_tuning/accelerate_config0.yaml
compute_environment: LOCAL_MACHINE
deepspeed_config: {}
distributed_type: MULTI_GPU
downcast_bf16: 'no'
fsdp_config: {}
machine_rank: 0
main_process_ip: "${MASTER_IP}"
main_process_port: 29500
num_machines: ${SLURM_JOB_NUM_NODES}
num_processes: 2
use_cpu: false
EOF

  cat <<EOF > ./fine_tuning/accelerate_config1.yaml
compute_environment: LOCAL_MACHINE
deepspeed_config: {}
distributed_type: MULTI_GPU
downcast_bf16: 'no'
fsdp_config: {}
machine_rank: 1
main_process_ip: "${MASTER_IP}"
main_process_port: 29500
num_machines: ${SLURM_JOB_NUM_NODES}
num_processes: 2
use_cpu: false
EOF
fi


# Launch distributed training
accelerate launch \
  --config_file ./fine_tuning/accelerate_config$SLURM_NODEID.yaml \
  ./fine_tuning/train.py meta-llama/Llama-3.2-1B-Instruct

# Run evaluation only on the master node (node rank 0)
if [ $SLURM_NODEID -eq 0 ]; then
    echo "Running evaluation on master node..."
    python3 fine_tuning/evaluate.py --model-name meta-llama/Llama-3.2-1B-Instruct
fi