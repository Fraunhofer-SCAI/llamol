#!/bin/bash
#SBATCH --mem=32gb                    # Total memory limit
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=<YOUR PARTITION>
#SBATCH --gres=gpu:a100:1
#SBATCH --time=2-00:00:00               # Time limit 2-hrs:min:sec days

export CUDA_VISIBLE_DEVICES=0

# TODO: Change FULL_PATH_TO_CONDA to the binary where the conda folder is: see https://github.com/conda/conda/issues/8536 
conda activate FULL_PATH_TO_CONDA/torch2-llamol
module load CUDA/11.7.0
module load GCC/7.1.0-2.28

cd ~/llama2-mol

srun python train.py train=llama2-M-Full-RSS > "train_runs/run_$SLURM_JOB_ID.out" 