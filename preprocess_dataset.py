import argparse
import json
import os
import pickle
import random
from functools import partial

import pandas as pd
import numpy as np
import requests
import torch
import torch.distributed as dist
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool
from fragment_creator import BaseFragmentCreator, BricksFragmentCreator, Fragment
from tokenizer import SmilesTokenizer
from torch.utils.data.distributed import DistributedSampler
from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from tqdm.contrib.concurrent import process_map, thread_map
from typing import List
import swifter

DATA_CACHE_DIR = "data"


def _tokenize_smiles(
    smi: List[str],
    tokenizer: SmilesTokenizer = None,
    max_smiles_len=256,
    log_output=True,
):
    # try:
    tokens = tokenizer.encode(smi)
    if len(tokens) > max_smiles_len:
        if log_output:
            print(f"Removing to long {smi} with smiles len of {len(tokens)} ")
        return None

    return tokens

    # except Exception as e:
    #     print(e)
    #     return None


def _tokenize_scaffolds(smi: str, tokenizer=None, max_smiles_len=256, log_output=True):
    # try:

    smi = MurckoScaffoldSmiles(smi)
    tokens = tokenizer.encode(smi)
    tokens = tokens[1:-1]  # remove [SEP] and [CLS] tokens
    if len(tokens) > max_smiles_len:
        if log_output:
            print(f"Removing to long {smi} with smiles len of {len(tokens)} ")
        return None

    return tokens

    # except Exception as e:
    #     print(e)
    #     return None


def pad_batch(src, pad_idx):
    max_len = max([len(d) for d in src])
    # src = [d["src_input_ids"] for d in data]
    padded_src = np.ones([len(src), max_len]) * pad_idx

    for i, j in enumerate(src):
        padded_src[i][0 : len(j)] = j

    # try to predict the next token from the previouse tokens
    # essentially reconstructing the src sentence from the embeddings and the previous sentence
    padded_src = padded_src.T
    return padded_src


def pretokenize(
    data_file=os.path.join(
        DATA_CACHE_DIR, "FULL_combined_zinc_pubchemqc_qm9_pc9_reddb_chembl.parquet"
    ),
    tokenizer=SmilesTokenizer(),
    limit=None,
    context=["logp", "sascore", "mol_weight"],
    out_name: str = "processed_dataset",
    remove_nan_context_rows: bool = False,
):
    df = pd.read_parquet(data_file)

    if limit is not None:
        # smiles_list = df.smiles[:limit]
        df = df.sample(n=limit)  # df[:limit]
        # NOTE: Set here if necessary, but for memory efficiency not duplicating millions of smiles
        # smiles_list = df.smiles
    else:
        # shuffle the rows
        df = df.sample(frac=1.0)

    cpu_count = (
        multiprocessing.cpu_count()
    )  # min(int(multiprocessing.cpu_count() * 0.8), 8)
    print(f"Running on {cpu_count} CPUs ")

    tqdm.pandas()

    df["scaffolds"] = df["smiles"].progress_map(lambda s: None if "." in s else s)
    df["smiles"] = df["scaffolds"].copy()
    orig_len = len(df)
    if context is not None:
        if df.get("origin") is not None:
            origins = df["origin"].unique()
            origin_dics = {}
            for i, o in enumerate(origins):
                df.loc[df["origin"] == o, "origin"] = i
                origin_dics[o] = i
            df["origin"] = df["origin"].astype(float)
            with open(
                os.path.join(
                    DATA_CACHE_DIR, os.path.basename(data_file) + "_origins.json"
                ),
                "w",
            ) as f:
                json.dump(origin_dics, f)

        mask = (
            ~df["smiles"].isna()
            & (
                (~df[context].isna()).all(axis=1)
                if remove_nan_context_rows
                else np.ones(len(df["smiles"]), dtype=bool)
            )
            & ~df["scaffolds"].isna()
        )
    else:
        mask = ~df["smiles"].isna()
    error_count = np.count_nonzero(~mask)
    df = df[mask]
    # print("HELLO")
    # print("***"*10)

    # tokenizer.batch_encode_plus()

    # df["scaffolds"] = df["scaffolds"].swifter.apply(
    #     partial(_tokenize_scaffolds, tokenizer=tokenizer, log_output=False)
    # )
    # df["scaffolds"] = df["scaffolds"].swifter.apply(
    #     partial(_tokenize_scaffolds, tokenizer=tokenizer, log_output=False)
    # )
    df["tokens"] = df["smiles"].swifter.apply(
        partial(_tokenize_smiles, tokenizer=tokenizer, log_output=False)
    )
    df["scaffolds"] = df["tokens"].copy()

    mask = ~df["tokens"].isna() & ~df["scaffolds"].isna()
    df = df[mask]
    error_count += np.count_nonzero(~mask)

    # Shuffle the data
    df = df.sample(frac=1).reset_index(drop=True)
    # with Pool(cpu_count) as p:
    #     df["scaffolds"] = list(

    #             p.map(partial( _tokenize_scaffolds ,tokenizer=tokenizer, log_output=False), tqdm(df.smiles.to_numpy(),total=len(df)), chunksize=1000),

    #     )

    #     df["smiles"] = list(
    #             p.map(partial( _tokenize_smiles ,tokenizer=tokenizer, log_output=False), tqdm(df.smiles.to_numpy(),total=len(df)), chunksize=1000),
    #     )

    if context is not None:
        context_list = df[context].to_numpy()
        context_dict = {k: context_list[:, i] for i, k in enumerate(context)}
    else:
        context_dict = {}

    print(f"Error count: {error_count} / {orig_len} = {error_count/orig_len}")

    cache_path = os.path.join(os.path.dirname(__file__), ".cache")
    os.makedirs(cache_path, exist_ok=True)
    out_path = os.path.join(cache_path, f"{out_name}_{limit}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(
            {
                "tokens": df["tokens"].tolist(),
                "smiles": df["smiles"].tolist(),
                "scaf": df["scaffolds"].tolist(),
                **context_dict,
            },
            f,
        )
    print(f"Saved to {out_path}")
    print("Done.")


class PretokDataset(torch.utils.data.Dataset):
    """Loads pretokenized example from disk and returns them as PyTorch tensors."""

    def __init__(self, split, pad_token_id, dataset="processed_dataset.pkl"):
        super().__init__()
        self.split = split
        self.dataset = dataset
        self.pad_token_id = pad_token_id
        cache_path = os.path.join(os.path.dirname(__file__), ".cache")
        with open(os.path.join(cache_path, self.dataset), "rb") as f:
            self.data_dict = pickle.load(f)

        # split out 10% of the data for validation
        split_ix = int(len(self.data_dict["tokens"]) * 0.9)
        if self.split == "train":
            self.data_dict = {k: self.data_dict[k][:split_ix] for k in self.data_dict}
        elif self.split == "val":
            self.data_dict = {k: self.data_dict[k][split_ix:] for k in self.data_dict}
        else:
            raise RuntimeError(f"Could not find split for: self.split={self.split}")

    def __len__(self):
        return len(self.data_dict["tokens"])

    def __getitem__(self, idx):
        m = self.data_dict

        start = idx
        end = idx + 1

        # calling .astype will copy the data into a new numpy array, now in RAM
        padded_tokens = pad_batch(m["tokens"][start:end], self.pad_token_id)
        chunk = torch.from_numpy((padded_tokens).astype(np.int64))

        padded_scaffolds = torch.from_numpy(
            pad_batch(m["scaf"][start:end], self.pad_token_id).astype(np.int64)
        )

        item = {
            "seq": chunk,
            "scaf": padded_scaffolds,
            "smiles": m["smiles"][start:end],
            **{
                k: torch.tensor(m[k][start:end], dtype=torch.float32)
                for k in m
                if k != "scaf" and k != "tokens" and k != "smiles"
            },
        }

        return item


def padding_collate_fn(
    data, tokenizer: SmilesTokenizer, fragment_creator: BaseFragmentCreator
):
    # data = list of dicts
    pad_idx = tokenizer.pad_token_id

    src = [d["seq"] for d in data]

    max_len = max([len(d) for d in src])
    padded_src = np.ones([len(src), max_len]) * pad_idx
    for i, j in enumerate(src):
        padded_src[i][0 : len(j)] = j.ravel()

    if fragment_creator is None:
        smiles_context = [d["scaf"] for d in data]
    else:
        # Remove start and end token after tokenization with [1:-1  ]
        smiles_context = []
        for d in data:
            s = d["smiles"][0]
            tokens = d["seq"]
            frag = fragment_creator.create_fragment(Fragment(smiles=s, tokens=tokens))
            if frag.tokens is not None:
                smiles_context.append(frag.tokens)
            else:
                smiles_context.append(
                    torch.tensor(
                        tokenizer.encode(frag.smiles)[1:-1],
                        dtype=torch.long,
                        device=tokens.device,
                    )
                )

    max_len_ctx = max([len(d) for d in smiles_context])
    padded_smiles_context = np.ones([len(smiles_context), max_len_ctx]) * pad_idx
    for i, j in enumerate(smiles_context):
        padded_smiles_context[i][0 : len(j)] = j.ravel()
    # try to predict the next token from the previouse tokens
    # essentially reconstructing the src sentence from the embeddings and the previous sentence
    padded_src = padded_src.T

    original_context_keys = [
        k for k in data[0].keys() if k != "seq" and k != "scaf" and k != "smiles"
    ]
    context_out_dict = {k: [] for k in original_context_keys}

    for k in original_context_keys:
        val_list = []
        for d in data:
            val_list.append(d[k])

        context_out_dict[k] = torch.concat(val_list, dim=0)

    return {
        "src": torch.tensor(padded_src, dtype=torch.long),  # for (seq_len, batch_size)
        "fragment": torch.tensor(padded_smiles_context.T, dtype=torch.long),
        "context": context_out_dict,
    }


class SmilesTask:
    @staticmethod
    def iter_batches(
        split,
        batch_size,
        device,
        context_keys: List[str],
        num_workers=0,
        dataset="processed_dataset.pkl",
        fragment_creator: BaseFragmentCreator = BricksFragmentCreator(),
    ):
        tokenizer = SmilesTokenizer()
        ds = PretokDataset(split, tokenizer.pad_token_id, dataset=dataset)
        is_ddp = int(os.environ.get("RANK", -1)) != -1
        dl = torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            pin_memory=True,
            num_workers=num_workers,
            shuffle=False,
            sampler=DistributedSampler(ds) if is_ddp else None,
            collate_fn=lambda batch: padding_collate_fn(
                batch, tokenizer, fragment_creator
            ),
        )

        for data in dl:
            data["src"] = data["src"].to(device, non_blocking=True)
            data["tgt"] = data["src"].to(device, non_blocking=True)

            data["src"] = data["src"][:-1, :].T  # batch_size, seq_len
            data["tgt"] = data["tgt"][1:, :].T  # batch_size, seq_len

            data["fragment"] = (
                data["fragment"].to(device, non_blocking=True).T
            )  # batch_size, seq_len
            keys = list(data["context"].keys())
            for d in keys:
                if d not in context_keys:
                    del data["context"][d]
                else:
                    data["context"][d] = data["context"][d].to(
                        device, non_blocking=True
                    )

            yield data


if __name__ == "__main__":
    
    pretokenize(
        data_file=os.path.join(
            DATA_CACHE_DIR,
            "OrganiX13.parquet",
        ),
        limit=None,  # Set how many molecules should be processed, if None all molecules will be processed,
        context=["logp", "sascore", "mol_weight"],
        out_name="processed_dataset",
        remove_nan_context_rows=False,
    )

