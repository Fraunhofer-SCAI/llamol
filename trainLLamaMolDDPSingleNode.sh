#!/bin/bash
#SBATCH --mem=32gb                    # Total memory limit
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=<HOW MANY GPUS>
#SBATCH --cpus-per-task=2
#SBATCH --partition=<YOUR PARTITION>
#SBATCH --gres=gpu:a100:<HOW MANY GPUS>
#SBATCH --time=2-00:00:00               # Time limit 2-hrs:min:sec days

export WORLD_SIZE=2
export OMP_NUM_THREADS=8
### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
PORT=54357
export MASTER_ADDR="$master_addr:$PORT"


# TODO: Change FULL_PATH_TO_CONDA to the binary where the conda folder is: see https://github.com/conda/conda/issues/8536 
conda activate FULL_PATH_TO_CONDA/torch2-llamol
module load CUDA/11.7.0
module load GCC/8.3.0

# TODO: Change this to the folder you cloned the repo in 
cd ~/llamol

srun torchrun --standalone --max_restarts=3  --nnodes=1 --nproc_per_node=2 --rdzv-id=$SLURM_JOB_ID --rdzv-backend=c10d  --rdzv-endpoint="$master_addr:$PORT" train.py train=llama2-M-Full > "train_runs/run_$SLURM_JOB_ID.out" 