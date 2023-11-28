#!/bin/bash

# TODO: Change FULL_PATH_TO_CONDA to the binary where the conda folder is: see https://github.com/conda/conda/issues/8536 
conda activate FULL_PATH_TO_CONDA/torch2-llamol

array=( logp sascore mol_weight )
# python sample.py --num_samples 20000 --num_samples_per_step 1000 --ckpt_path "out/llama2-M-Full-RSS.pt"  --max_new_tokens 256 --cmp_dataset_path="data/OrganiX13.parquet"
# for i in "${array[@]}"
# do
#     python sample.py --num_samples 10000 --num_samples_per_step 500 --kv_caching --ckpt_path "out/llama2-M-Full-RSS.pt" --context_cols "$i" --max_new_tokens 256 --cmp_dataset_path="data/OrganiX13.parquet"
# done

# 2 Combinations
python sample.py --num_samples 1000 --seed 4321 --kv_caching --ckpt_path "out/llama2-M-Full-RSS.pt" --context_cols logp sascore --max_new_tokens 256 --cmp_dataset_path="data/OrganiX13.parquet"
python sample.py --num_samples 1000 --seed 4321 --kv_caching --ckpt_path "out/llama2-M-Full-RSS.pt" --context_cols logp mol_weight --max_new_tokens 256 --cmp_dataset_path="data/OrganiX13.parquet"
python sample.py --num_samples 1000 --seed 4321 --kv_caching --ckpt_path "out/llama2-M-Full-RSS.pt" --context_cols sascore mol_weight --max_new_tokens 256 --cmp_dataset_path="data/OrganiX13.parquet"

# # # All 3
# python sample.py --num_samples 1000 --ckpt_path "out/llama2-M-Full-RSS.pt" --context_cols logp sascore mol_weight --kv_caching --max_new_tokens 256 --cmp_dataset_path="data/OrganiX13.parquet" --seed 4312