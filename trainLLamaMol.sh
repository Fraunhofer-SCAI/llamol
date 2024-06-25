#!/bin/bash
#SBATCH --mem=32gb                    # Total memory limit
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=gpu-mix
#SBATCH --gres=gpu:a100:1
#SBATCH --time=2-00:00:00               # Time limit 2-hrs:min:sec days

export MAMBA_EXE='/home/ndobberstein/.local/bin/micromamba';
export MAMBA_ROOT_PREFIX='/home/ndobberstein/micromamba';
__mamba_setup="$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX" 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__mamba_setup"
else
    alias micromamba="$MAMBA_EXE"  # Fallback on help from mamba activate
fi
unset __mamba_setup
# <<< mamba initialize <<<

export CUDA_VISIBLE_DEVICES=0

micromamba activate torch2-llamol

module load CUDA/11.7.0
module load GCC/7.1.0-2.28

cd /home/ndobberstein/AI/llamol-paper
mkdir -p ./train_runs
srun python train.py train=llama2-M-Full-RSS > "train_runs/run_$SLURM_JOB_ID.out" 