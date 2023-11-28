#!/bin/bash

# TODO: Change FULL_PATH_TO_CONDA to the binary where the conda folder is: see https://github.com/conda/conda/issues/8536 
conda activate FULL_PATH_TO_CONDA/torch2-llamol


# context_smiles=("c1ccccc1" "s1cccc1" "C1=CSC=C1" "CC1=CSC=C1" "C1=CC=C2C(=C1)C3=CC=CC=C3S2" "CCO" "CC=O" "CC(=O)OC1=CC=CC=C1C(=O)O" "CC(=O)NC1=CC=C(C=C1)O" "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O" "OC(=O)C(C)c1ccc(cc1)CC(C)C" "C1C(=O)NC(=O)NC1=O" "CN1C=NC2=C1C(=O)N(C(=O)N2C)C" "CN1CCC23C4C1CC5=C2C(=C(C=C5)O)OC3C(C=C4)O" "CN1CCC23C4C1CC5=C2C(=C(C=C5)OC)OC3C(=O)CC4")
# context_smiles=("CN1CCC23C4C1CC5=C2C(=C(C=C5)O)OC3C(C=C4)O" "CN1CC[C@]23[C@@H]4[C@H]1CC5=C2C(=C(C=C5)O)O[C@H]3[C@H](C=C4)O" "CN1CCC23C4C1CC5=C2C(=C(C=C5)OC)OC3C(=O)CC4" "CN1CC[C@]23[C@@H]4[C@H]1CC5=C2C(=C(C=C5)OC)O[C@H]3C(=O)CC4" )
# context_smiles=("C1=CSC=C1" )
context_smiles=("C1=CSC=C1" "CC=O" "CC(=O)NC1=CC=C(C=C1)O" "CN1C=NC2=C1C(=O)N(C(=O)N2C)C")
for smi in "${context_smiles[@]}"; do
    # Only fragment generation
    # output=$(python sample.py --num_samples 1000 --ckpt_path "out/llama2-M-Full-RSS.pt" --max_new_tokens 256 --cmp_dataset_path="data/OrganiX13.parquet" --context_smi "$smi")
    
    # Fragment and LogP
    # output=$(python sample.py --num_samples 1000 --ckpt_path "out/llama2-M-Full-RSS.pt" --max_new_tokens 256 --cmp_dataset_path="data/OrganiX13.parquet" --context_smi "$smi" --context_cols "logp" )

    # Fragment and Sascore
    # output=$(python sample.py --num_samples 1000 --ckpt_path "out/llama2-M-Full-RSS.pt" --max_new_tokens 256 --cmp_dataset_path="data/OrganiX13.parquet" --context_smi "$smi" --context_cols "sascore" )
    
    # Fragment and Mol weight
    # output=$(python sample.py --num_samples 1000 --ckpt_path "out/llama2-M-Full-RSS.pt" --max_new_tokens 256 --cmp_dataset_path="data/OrganiX13.parquet" --context_smi "$smi" --context_cols "mol_weight" )

    # Multi Fragment Condition    

    # Logp + Sascore
    # output=$(python sample.py --num_samples 1000 --ckpt_path "out/llama2-M-Full-RSS.pt" --max_new_tokens 256 --cmp_dataset_path="data/OrganiX13.parquet" --context_smi "$smi" --context_cols "logp" "sascore" )


    # Logp + Mol Weight
    # output=$(python sample.py --num_samples 1000 --ckpt_path "out/llama2-M-Full-RSS.pt" --max_new_tokens 256 --cmp_dataset_path="data/OrganiX13.parquet" --context_smi "$smi" --context_cols "logp" "mol_weight" )

    # Sascore + Mol Weight    
    # output=$(python sample.py --num_samples 1000 --ckpt_path "out/llama2-M-Full-RSS.pt" --max_new_tokens 256 --cmp_dataset_path="data/OrganiX13.parquet" --context_smi "$smi" --context_cols "sascore" "mol_weight" )

    # Logp +  Sascore + Mol Weight    
    output=$(python sample.py --num_samples 1000 --ckpt_path "out/llama2-M-Full-RSS.pt" --max_new_tokens 256 --cmp_dataset_path="data/OrganiX13.parquet" --context_smi "$smi" --context_cols "logp" "sascore" "mol_weight" )


    echo "SMI: $smi"
    echo "----------------------"
done