#!/bin/bash
#SBATCH --mem=32gb                    # Total memory limit
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=gpu-mix
#SBATCH --gres=gpu:v100:1
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

cd /home/ndobberstein/AI/llamol-paper

array=( logp sascore mol_weight )


srun python sample.py --sample_range "UNK" --num_samples 20000 --num_samples_per_step 1000 --ckpt_path "out/llama2-M-Full-RSS-Canonical.pt"  --max_new_tokens 256 --cmp_dataset_path="data/OrganiX13.parquet"
# for i in "${array[@]}"
# do
#     # For the logp {2,4,6}
#     # srun python sample.py --sample_range "2D" --num_samples 10000 --num_samples_per_step 1000 --ckpt_path "out/llama2-M-Full-RSS-Canonical.pt" --context_cols "$i" --max_new_tokens 256 --cmp_dataset_path="data/OrganiX13.parquet"

#     # For logp [-2,7;1]
#     srun python sample.py --sample_range "1D" --num_samples 10000 --num_samples_per_step 1000 --ckpt_path "out/llama2-M-Full-RSS-Canonical.pt" --context_cols "$i" --max_new_tokens 256 --cmp_dataset_path="data/OrganiX13.parquet"
# done


# 2 Combinations
# srun python sample.py --sample_range "2D" --num_samples 10000 --num_samples_per_step 1000 --seed 4321 --ckpt_path "out/llama2-M-Full-RSS-Canonical.pt" --context_cols logp sascore --max_new_tokens 256 --cmp_dataset_path="data/OrganiX13.parquet"
# srun  python sample.py --sample_range "2D" --num_samples 10000 --num_samples_per_step 1000 --seed 4321 --ckpt_path "out/llama2-M-Full-RSS-Canonical.pt" --context_cols logp mol_weight --max_new_tokens 256 --cmp_dataset_path="data/OrganiX13.parquet"
# srun  python sample.py --sample_range "2D" --num_samples 10000 --num_samples_per_step 1000 --seed 4321 --ckpt_path "out/llama2-M-Full-RSS-Canonical.pt" --context_cols sascore mol_weight --max_new_tokens 256 --cmp_dataset_path="data/OrganiX13.parquet"

# # # # All 3
# srun python sample.py --sample_range "3D" --num_samples 10000 --num_samples_per_step 1000 --ckpt_path "out/llama2-M-Full-RSS-Canonical.pt" --context_cols logp sascore mol_weight --max_new_tokens 256 --cmp_dataset_path="data/OrganiX13.parquet" --seed 4312

# srun python sample.py --num_samples 20000 --num_samples_per_step 1000 --ckpt_path "out/llama2-M-Full-RSS.pt"  --max_new_tokens 256 --cmp_dataset_path="data/OrganiX13.parquet"
# for i in "${array[@]}"
# do
#     srun python sample.py --num_samples 10000 --num_samples_per_step 1000 --ckpt_path "out/llama2-M-Full-RSS.pt" --context_cols "$i" --max_new_tokens 256 --cmp_dataset_path="data/OrganiX13.parquet"
# done

# 2 Combinations
# python sample.py --num_samples 10000 --num_samples_per_step 1000 --seed 4321 --ckpt_path "out/llama2-M-Full-RSS.pt" --context_cols logp sascore --max_new_tokens 256 --cmp_dataset_path="data/OrganiX13.parquet"
# python sample.py --num_samples 10000 --num_samples_per_step 1000 --seed 4321 --ckpt_path "out/llama2-M-Full-RSS.pt" --context_cols logp mol_weight --max_new_tokens 256 --cmp_dataset_path="data/OrganiX13.parquet"
# python sample.py --num_samples 10000 --num_samples_per_step 1000 --seed 4321 --ckpt_path "out/llama2-M-Full-RSS.pt" --context_cols sascore mol_weight --max_new_tokens 256 --cmp_dataset_path="data/OrganiX13.parquet"

# # # # All 3
# python sample.py --num_samples 10000 --num_samples_per_step 1000 --ckpt_path "out/llama2-M-Full-RSS.pt" --context_cols logp sascore mol_weight --max_new_tokens 256 --cmp_dataset_path="data/OrganiX13.parquet" --seed 4312