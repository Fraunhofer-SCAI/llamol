#!/bin/bash

# TODO: Change FULL_PATH_TO_CONDA to the binary where the conda folder is: see https://github.com/conda/conda/issues/8536 

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
context_smiles=("c1ccccc1" "s1cccc1" "CC1=CSC=C1" "CCO" "CC=O" "CC(=O)OC1=CC=CC=C1C(=O)O" "CC(=O)NC1=CC=C(C=C1)O" "CN1C=NC2=C1C(=O)N(C(=O)N2C)C" "CN1CCC23C4C1CC5=C2C(=C(C=C5)O)OC3C(C=C4)O" "OC(=O)C(C)c1ccc(cc1)CC(C)C")
# context_smiles=("CN1CCC23C4C1CC5=C2C(=C(C=C5)O)OC3C(C=C4)O" "CN1CC[C@]23[C@@H]4[C@H]1CC5=C2C(=C(C=C5)O)O[C@H]3[C@H](C=C4)O" "CN1CCC23C4C1CC5=C2C(=C(C=C5)OC)OC3C(=O)CC4" "CN1CC[C@]23[C@@H]4[C@H]1CC5=C2C(=C(C=C5)OC)O[C@H]3C(=O)CC4" )
# context_smiles=("C1=CSC=C1" )
# context_smiles=("C1=CSC=C1" "CC=O" "CC(=O)NC1=CC=C(C=C1)O" "CN1C=NC2=C1C(=O)N(C(=O)N2C)C")
for smi in "${context_smiles[@]}"; do
    # Only fragment generation
    output=$(python sample.py --sample_range "UNK" --num_samples 1000 --ckpt_path "out/llama2-M-Full-RSS-Canonical.pt" --max_new_tokens 256 --cmp_dataset_path="data/OrganiX13.parquet" --context_smi "$smi")
    
    # Fragment and LogP
    output=$(python sample.py --sample_range "TOK" --num_samples 1000 --ckpt_path "out/llama2-M-Full-RSS-Canonical.pt" --max_new_tokens 256 --cmp_dataset_path="data/OrganiX13.parquet" --context_smi "$smi" --context_cols "logp" )

    # Fragment and Sascore
    output=$(python sample.py --sample_range "TOK" --num_samples 1000 --ckpt_path "out/llama2-M-Full-RSS-Canonical.pt" --max_new_tokens 256 --cmp_dataset_path="data/OrganiX13.parquet" --context_smi "$smi" --context_cols "sascore" )
    
    # Fragment and Mol weight
    output=$(python sample.py --sample_range "TOK" --num_samples 1000 --ckpt_path "out/llama2-M-Full-RSS-Canonical.pt" --max_new_tokens 256 --cmp_dataset_path="data/OrganiX13.parquet" --context_smi "$smi" --context_cols "mol_weight" )

    echo "SMI: $smi"
    echo "----------------------"
done

echo "Multi Property"
context_smiles=("C1=CSC=C1" "CC=O" "CC(=O)NC1=CC=C(C=C1)O" "CN1C=NC2=C1C(=O)N(C(=O)N2C)C")
for smi in "${context_smiles[@]}"; do
    # Multi Fragment Condition    
    # Logp + Sascore
    output=$(python sample.py --sample_range "TOK" --num_samples 1000 --ckpt_path "out/llama2-M-Full-RSS-Canonical.pt" --max_new_tokens 256 --cmp_dataset_path="data/OrganiX13.parquet" --context_smi "$smi" --context_cols "logp" "sascore" )


    # Logp + Mol Weight
    output=$(python sample.py --sample_range "TOK" --num_samples 1000 --ckpt_path "out/llama2-M-Full-RSS-Canonical.pt" --max_new_tokens 256 --cmp_dataset_path="data/OrganiX13.parquet" --context_smi "$smi" --context_cols "logp" "mol_weight" )

    # Sascore + Mol Weight    
    output=$(python sample.py --sample_range "TOK" --num_samples 1000 --ckpt_path "out/llama2-M-Full-RSS-Canonical.pt" --max_new_tokens 256 --cmp_dataset_path="data/OrganiX13.parquet" --context_smi "$smi" --context_cols "sascore" "mol_weight" )

    # Logp +  Sascore + Mol Weight    
    output=$(python sample.py --sample_range "TOK" --num_samples 1000 --ckpt_path "out/llama2-M-Full-RSS-Canonical.pt" --max_new_tokens 256 --cmp_dataset_path="data/OrganiX13.parquet" --context_smi "$smi" --context_cols "logp" "sascore" "mol_weight" )


    echo "SMI: $smi"
    echo "----------------------"
done
