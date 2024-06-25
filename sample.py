import os
from contextlib import nullcontext
import sys
import time
import pandas as pd
import torch
from tqdm.auto import tqdm

# from tqdm.notebook import tqdm
from model import Transformer
from plot_utils import (
    check_metrics,
    plot_1D_condition,
    plot_2D_condition,
    plot_3D_condition,
    plot_unconditional,
)
from tokenizer import SmilesTokenizer
import numpy as np
from typing import Dict, List, Tuple, Union
import re

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols

import logging

logger = logging.getLogger(__name__)


class Sampler:
    def __init__(
        self,
        load_path: str,
        device: str = "cpu",
        seed: int = 1337,
        dtype: str = "float16",
        compile: bool = True,
        quantize: bool = False,
        sample_range : str = "UNC_1D"
    ) -> None:
        self.sample_range = sample_range.upper()
        self.load_path = load_path
        self.device = device
        self.dtype = dtype
        self.compile = compile
        self.quantize = quantize
        self.seed = seed
        self._init_model()

    def _init_model(self):
        np.random.seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
        self.device_type = (
            "cuda" if "cuda" in self.device else "cpu"
        )  # for later use in torch.autocast
        ptdtype = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[self.dtype]
        self.ptdtype = ptdtype

        self.ctx = self._autocast()
        # init from a model saved in a specific directory
        # ckpt_path = os.path.join(out_dir, "ckpt_full_dim=256.pt")
        self.model = Transformer.load(self.load_path, device=self.device)

        self.model.eval()
        if self.quantize:
            raise NotImplementedError("Not properly implemented for CPU / GPU")
            self.model = torch.ao.quantization.quantize_dynamic(
                self.model,  # the original model
                {torch.nn.Linear},  # a set of layers to dynamically quantize
                dtype=torch.qint8,
            )

        if self.compile:
            logger.info("Compiling the model...")
            self.model = torch.compile(self.model)  # requires PyTorch 2.0 (optional)

        self.model = self.model.to(self.device)
        # load the tokenizer
        self.tokenizer = SmilesTokenizer()

    def get_context(
        self,
        context_col: List[str],
        context_smi: str,
        num_examples: int = 50,
    ):
        """
        Returns a dictionary in the form of
        {
        "fragment": torch.tensor,
        "context": {
            "logp": torch.tensor,
            "sascore": torch.tensor,
            "mol_weight": torch.tensor
        }
        }


        When context_smi is set to a string, then the "fragment" field is populated.
        All of the properties listed in the context_col list is set to the keys and the values are set to a resonable range for each property.

        num_examples indicates how many values are sampled for each property.
        """
        output_dict = {"context": {}, "fragment": None}

        if context_smi is not None:
            logger.debug(
                f"context_smiles: {context_smi}",
            )
            # NOTE: Remove beginning [CLS] and end token [SEP]
            incorporate_selfie = self.tokenizer.encode(context_smi)[1:-1]

            context = torch.tensor(
                [incorporate_selfie] * num_examples,
                dtype=torch.long,
                device=self.device,
            )

            output_dict["fragment"] = context

        if context_col is None:
            return output_dict

            
        if "logp" in context_col:
            if self.sample_range == "UNC" or self.sample_range == "1D":
                context = torch.randint(
                    -2, 7, (num_examples,), device=self.device, dtype=torch.float
                )
            elif self.sample_range == "2D":
                context = torch.tensor(
                    np.random.choice([2, 4, 6], (num_examples,)),
                    device=self.device,
                    dtype=self.ptdtype,
                )
            elif self.sample_range == "3D" or self.sample_range == "TOK":
                context = torch.tensor(
                    np.random.choice([-2, 0, 2], (num_examples,)),
                    device=self.device,
                    dtype=self.ptdtype,
                )
            else:
                raise RuntimeError(f"Do not have sample range {self.sample_range}")
            
            output_dict["context"]["logp"] = context

        if "energy" in context_col:
            context = 0.1 * torch.randint(
                -15, 15, (num_examples,), device=self.device, dtype=torch.float
            )
            # context = -2.0*torch.ones((num_examples,2),device=device,dtype=torch.float)
            context, _ = torch.sort(context, 0)
            output_dict["context"]["energy"] = context

        if "sascore" in context_col:
            
            if self.sample_range == "UNC" or self.sample_range == "1D":
                context = torch.randint(
                    1, 10, (num_examples, ), device=self.device, dtype=torch.float
                )
            elif self.sample_range == "2D" or self.sample_range == "TOK":
                context = torch.tensor(
                    np.random.choice([2, 3, 4], (num_examples,)),
                    device=self.device,
                    dtype=self.ptdtype,
                )
            elif self.sample_range == "3D" :
                context = torch.tensor(
                    np.random.choice([2, 3], (num_examples,)),
                    device=self.device,
                    dtype=self.ptdtype,
                )
            else:
                raise RuntimeError(f"Do not have sample range {self.sample_range}")
            

            # context, _ = torch.sort(context, 0)
            output_dict["context"]["sascore"] = context

        if "mol_weight" in context_col:
            if self.sample_range == "UNC" or self.sample_range == "1D":
                context = torch.randint(
                    1, 10, (num_examples, ), device=self.device, dtype=torch.float
                )
            elif self.sample_range == "2D" or self.sample_range == "TOK":
                context = torch.tensor(
                    np.random.choice([2, 3, 4], (num_examples,)),
                    device=self.device,
                    dtype=self.ptdtype,
                )
            elif self.sample_range == "3D" :
                context = torch.tensor(
                    np.random.choice([3, 4], (num_examples,)),
                    device=self.device,
                    dtype=self.ptdtype,
                )
            else:
                raise RuntimeError(f"Do not have sample range {self.sample_range}")
            
            # context, _ = torch.sort(context, 0)
            output_dict["context"]["mol_weight"] = context
        # logger.info(f"get_context: {output_dict}")
        return output_dict

    def _autocast(self):
        if "cuda" in self.device:
            if self.dtype == "bfloat16" and torch.cuda.is_bf16_supported():
                return torch.cuda.amp.autocast(dtype=torch.bfloat16)
            elif self.dtype == "float16":
                return torch.cuda.amp.autocast(dtype=torch.float16)
            else:
                return torch.cuda.amp.autocast(dtype=torch.float32)
        else:  # cpu
            return nullcontext()

    @torch.no_grad()
    def generate(
        self,
        context_cols: Union[List[str], None, Dict[str, torch.Tensor]] = None,
        context_smi: Union[str, None] = None,
        start_smiles: Union[str, None] = None,
        num_samples: int = 50,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_k: Union[int, None] = None,
        return_context: bool = False,
        total_gen_steps: int = 1,
        use_kv_cache: bool = False,
    ) -> Union[List[str], Tuple[List[str], List[float]]]:
        """
        Generates a list of SMILES. With the default options it would generate them unconditionally.
        Params:
            - context_cols : When a list the context is randomly sampled from the get_context method, when given a dictionary the
                             context values are taken from the dictionary instead.
            - context_smi : Further conditioning by the usage of a molecular fragment
            . start_smiles : Can be used to start the SMILES with a specific string, the model then generates the next tokens including that start sequence.
            - num_samples : Controlls how many SMILES in total will be generated be the model.
            - max_new_tokens : Controlls the maximum length of each SMILES (in tokens) that is generated.
            - temperature: Controlls the randomness of the model. A temperature = 1.0 means it is the trained distribution. A temperature < 1 is more deterministic and temperature > 1 is more random
            - top_k : Clamps the probability distribution to the top k tokens. From these the next token is then sampled from.
            - return_context : Whether the context that was given to the model should be returned.
            - total_gen_steps : In how many sub steps the generation should be split up to. Useful when generation 10k + SMILES and wanting to chunk these into for example 10 * 1k generations with total_gen_steps = 10.
            - use_kv_cache: Runs the generation using kv-caching. It is faster, but takes more memory.
        """

        with self.ctx:
            gens_per_step = num_samples // total_gen_steps

            logger.debug(f"Gens per Step: {gens_per_step}")
            context = None  # {"context": None, "fragment" : None}
            out_smiles = []
            with tqdm(total=total_gen_steps, desc="Batch") as pbar:
                for i in range(total_gen_steps):
                    if isinstance(context_cols, dict):
                        # TODO: Test if same length
                        cd = {
                            c: context_cols[c][
                                i * gens_per_step : (i + 1) * gens_per_step
                            ]
                            for c in context_cols.keys()
                        }

                        context_dict = {"context": cd, "fragment": None}
                        if context_smi is not None:
                            logger.debug(
                                f"context_smiles: {context_smi}",
                            )
                            # NOTE: Remove beginning [CLS] and end token [SEP]
                            incorporate_selfie = self.tokenizer.encode(context_smi)[
                                1:-1
                            ]

                            context_tensor = torch.tensor(
                                [incorporate_selfie] * gens_per_step,
                                dtype=torch.long,
                                device=self.device,
                            )

                            context_dict["fragment"] = context_tensor
                        context_cols = list(context_cols.keys())

                    else:
                        context_dict = self.get_context(
                            context_cols, context_smi, num_examples=gens_per_step
                        )

                    # for k in range(num_samples):
                    y = self.model.generate(
                        self.tokenizer,
                        context=context_dict["context"],
                        fragments=context_dict["fragment"],
                        start_smiles=start_smiles,
                        num_gen=gens_per_step,
                        temperature=temperature,
                        top_k=top_k,
                        max_length=max_new_tokens,
                        device=self.device,
                        cache_kv=use_kv_cache,
                    )

                    new_context = {k: [] for k in context_dict["context"]}
                    for i, sample in enumerate(y):
                        # print(sample)
                        mol = Chem.MolFromSmiles(sample)

                        if mol is not None:
                            can_smiles = Chem.MolToSmiles(mol,isomericSmiles=True, canonical=True)
                            # print("SMILES", sample, "vs CAN SMILES", can_smiles)
                            out_smiles.append(can_smiles)
                            for k in new_context:
                                new_context[k].append(
                                    context_dict["context"][k][i].unsqueeze(-1)
                                )

                    for k in new_context:
                        new_context[k] = torch.concat(new_context[k], dim=0)

                    if context is None:
                        context = new_context
                    else:
                        for k in context:
                            context[k] = torch.concat(
                                [context[k], new_context[k]], dim=0
                            )

                    pbar.update(1)

            logger.info(
                f"Number valid generated: {len(out_smiles) / num_samples * 100} %"
            )
            logger.info("---------------")

            if return_context:
                return (out_smiles, context)

            else:
                return out_smiles

    @torch.no_grad()
    def generate_with_evaluation(
        self,
        context_cols: Union[List[str], None] = None,
        context_smi: Union[str, None] = None,
        start_smiles: Union[str, None] = None,
        num_samples: int = 50,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_k: Union[int, None] = None,
        cmp_context_dict: Union[Dict[str, torch.Tensor], None] = None,
        total_gen_steps: int = 1,
        use_kv_cache: bool = False,
    ):
        out_smiles, new_context = self.generate(
            context_cols=context_cols,
            context_smi=context_smi,
            start_smiles=start_smiles,
            num_samples=num_samples,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            return_context=True,
            total_gen_steps=total_gen_steps,
            use_kv_cache=use_kv_cache,
        )

        out_dir = os.path.dirname(self.load_path)

        if context_cols is not None:
            if len(context_cols) == 1:
                plot_1D_condition(
                    context_cols,
                    os.path.join(out_dir, "plots"),
                    new_context,
                    out_smiles,
                    temperature,
                    cmp_context_dict,
                    context_scaler=None,
                )

            elif len(context_cols) == 2:
                plot_2D_condition(
                    context_cols,
                    os.path.join(out_dir, "plots"),
                    new_context,
                    out_smiles,
                    temperature,
                    label=context_smi,
                )

            elif len(context_cols) == 3:
                plot_3D_condition(
                    context_cols,
                    os.path.join(out_dir, "plots"),
                    new_context,
                    out_smiles,
                    temperature,
                )

            else:
                raise NotImplementedError(
                    "Currently not implemented for len(context_col) > 3"
                )

        else:
            # Unconditional Case
            plot_unconditional(
                out_path=os.path.join(out_dir, "plots"),
                smiles=out_smiles,
                temperature=temperature,
                cmp_context_dict=cmp_context_dict,
            )

        if context_smi is not None:
            pattern = r"\[\d+\*\]"
            # replace [14*] etc
            context_smi = re.sub(pattern, "", context_smi)

            context_mol = Chem.MolFromSmiles(context_smi)
            context_smarts = Chem.MolToSmarts(context_mol)

            pattern = r"(?<!\[)([:-=#])(?!\])(?![^\[]*?\])"

            context_smarts = re.sub(pattern, "~", context_smarts)
            logger.info(f"context_smarts {context_smarts}")
            out_mols = [Chem.MolFromSmiles(smi) for smi in out_smiles]

            context_fingerprint = FingerprintMols.FingerprintMol(context_mol)
            out_fingerprints = [FingerprintMols.FingerprintMol(fi) for fi in out_mols]
            all_sim = []
            all_sub = []
            for out_fing, out_mol in zip(out_fingerprints, out_mols):
                similarity = DataStructs.TanimotoSimilarity(
                    context_fingerprint, out_fing
                )

                has_sub = out_mol.HasSubstructMatch(Chem.MolFromSmarts(context_smarts))
                all_sub.append(has_sub)
                all_sim.append(similarity)

                # print(similarity,has_sub)
            logger.info(f"Mean sim {np.mean(all_sim)}")
            logger.info(
                f"Has Sub: {np.count_nonzero(all_sub)} or {round(np.count_nonzero(all_sub) / len(all_sub) * 100, 4)} %"
            )

        return out_smiles, new_context


if __name__ == "__main__":
    import argparse
    import rdkit.rdBase as rkrb
    import rdkit.RDLogger as rkl

    logger = rkl.logger()
    logger.setLevel(rkl.ERROR)
    rkrb.DisableLog("rdApp.error")

    torch.set_num_threads(8)
    # logging.basicConfig(level=logging.INFO)
    logging.basicConfig(level=logging.INFO, filename="new_gen_log_FULL_can_canmodel_tokensseq.out", filemode="a",
                    format='%(asctime)s - %(levelname)s - %(message)s')    
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description="Generate SMILES strings using a trained model."
    )
    # parser.add_argument('--context_cols', type=str, nargs='+', default=None)
    parser.add_argument(
        "--context_cols",
        type=str,
        nargs="+",
        default=None,
        help="The given conditions are sampled from a fixed interval and given to the modeĺ.",
    )
    parser.add_argument(
        "--context_smi",
        type=str,
        default=None,
        help="This SMILES is given as context to the model and should be integrated in the generated molecules.",
    )
    parser.add_argument(
        "--start_smiles",
        type=str,
        default=None,
        help="This SMILES is placed at the front of each sample, from which on the generation continues.",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "out", "llama2-M-Full-RSS.pt"),
        help="Which model should be used in the generation",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=50,
        help="Controls how many samples should be generated",
    )
    parser.add_argument(
        "--num_samples_per_step",
        type=int,
        default=1000,
        help="Works in conjunction with num_samples, by splitting the total into num_samples_per_step jobs. When num_samples > num_samples_per_step then it is split up into multiple seperate generation steps.",
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Sets how many tokens should be generated from the model. We only trained with a max size of 256, but it is possible to generate longer molecules. However, these might be worse in quality.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sets the randomness of the generation - A temperature of 0 would be deterministic and a temperature of > 1 is more random.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="The top_k of the sampling. Per default it is None, but can be set to an integer to have a more focused generation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random number generator seed, to make sampling consistent.",
    )
    parser.add_argument(
        "--cmp_dataset_path",
        type=str,
        default=None,
        help="A dataset in parquet or csv format to be used in the sample plots and to compute the metrics such as the novelty.",
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    parser.add_argument(
        "--device",
        type=str,
        default=device,
        help="Change the device the model and generation is run on",
    )

    if "cuda" in device:
        # dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
        dtype = "float16" if torch.cuda.is_available() else "float32"
    else:
        dtype = "float32"

    parser.add_argument(
        "--dtype",
        type=str,
        default=dtype,
        help="Change the datatype of the computation. Per default it is float32 on CPU and float16 on GPU",
    )
    parser.add_argument(
        "--compile",
        type=bool,
        default=True,
        help="Use torch.compile to compile the model. Only works on torch>=2.0, but should make the inference faster.",
    )
    parser.add_argument(
        "--quantize",
        type=bool,
        default=False,
        help="(CURRENTLY NOT WORKING) Enable quantization to in8.",
    )
    parser.add_argument(
        "--kv_caching",
        action="store_true",
        default=False,
        help="Makes the attention mechanism linear, because the old keys and values are cached. The drawback is higher memory consumption.",
    )
    parser.add_argument(
        "--sample_range",
        type=str,
        default="UNC",
        help="Is either Unc,1D,2D,3D or TOK determines the range of logp,sascore and mol_weight",
    )
    args = parser.parse_args()

    logger.info("Sampling with the following parameters:")
    logger.info(f"Checkpoint: {args.ckpt_path}")
    logger.info(f"Context columns: {args.context_cols}")
    logger.info(f"Context SMILES: {args.context_smi}")
    logger.info(f"Start SMILES: {args.start_smiles}")
    logger.info(f"Number of samples: {args.num_samples}")
    logger.info(f"Max new tokens: {args.max_new_tokens}")
    logger.info(f"Temperature: {args.temperature}")
    logger.info(f"Top k: {args.top_k}")
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Data type: {args.dtype}")
    logger.info(f"Compile: {args.compile}")
    logger.info(f"Comparison dataset path: {args.cmp_dataset_path}")
    logger.info(f"Quantize: {args.quantize}")
    logger.info(f"Key Value Caching Enabled: {args.kv_caching}")
    logger.info(f"Sample range: {args.sample_range}")
    sampler = Sampler(
        load_path=os.path.join(os.path.dirname(__file__), args.ckpt_path),
        device=args.device,
        seed=args.seed,
        dtype=args.dtype,
        compile=args.compile,
        quantize=args.quantize,
        sample_range=args.sample_range,
    )

    comp_context_dict = None
    comp_smiles = None
    if args.cmp_dataset_path is not None:
        df_comp = pd.read_parquet(args.cmp_dataset_path)
        # df_comp = df_comp.sample(n=2_500_000)
        comp_context_dict = {
            c: df_comp[c].to_numpy() for c in ["logp", "sascore", "mol_weight"]
        }
        comp_smiles = df_comp["smiles"]

    # Canonicalize input smiles
    context_smi = args.context_smi
    if args.context_smi is not None:
        mol = Chem.MolFromSmiles(args.context_smi)
        if mol is None:
            raise RuntimeError("Context Smiles is not a valid molecule!")
    
        context_smi = Chem.MolToSmiles(mol,isomericSmiles=True, canonical=True)
    # context_smi = args.context_smi
    measure_time = True
    start_time = time.time()
    smiles, context = sampler.generate_with_evaluation(
        context_cols=args.context_cols,
        context_smi=context_smi,
        start_smiles=args.start_smiles,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        cmp_context_dict=comp_context_dict,
        total_gen_steps=int(np.ceil(args.num_samples / args.num_samples_per_step)),
        use_kv_cache=args.kv_caching,
    )
    end_time = time.time()
    if measure_time:
        logger.info(f"Generation took: {end_time - start_time} sec")
   
    if comp_smiles is not None:
        res_metrics = check_metrics(smiles, comp_smiles)
        logger.info(f"Metrics: {res_metrics}")

    logger.info("Generated Molecules:\n\n")
    for s in smiles[:20]:
        print(s)
