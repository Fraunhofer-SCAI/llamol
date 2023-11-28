from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List, Union
from fragment_creator import fragment_creator_factory

from model import ContextArgs, ModelArgs
from tqdm import tqdm
import math
import os
import time
from contextlib import nullcontext
from datetime import datetime
from functools import partial

import torch
import numpy as np
from model import ContextArgs, Transformer, ModelArgs
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from preprocess_dataset import SmilesTask
from tokenizer import SmilesTokenizer

import logging

logger = logging.getLogger(__name__)


@dataclass
class IOConfig:
    # I/O
    out_dir: str = "out"
    eval_interval: int = 500
    log_interval: int = 10
    eval_iters: int = 25
    eval_only: bool = False  # if True, script exits right after the first eval
    always_save_checkpoint: bool = (
        False  # if True, always save a checkpoint after each eval
    )
    init_from: str = "scratch"  # 'scratch' or 'resume'
    resume_when_snapshot_available: bool = True


@dataclass
class LoaderConfig:
    # data
    batch_size: int = (
        384  # if gradient_accumulation_steps > 1, this is the micro-batch size
    )
    max_seq_len: int = 768
    dataset: str = "smiles"
    processed_dataset_ckpt: str = "processed_dataset_None.pkl"
    fragment_creator: Union[str, None] = None


# dim = 256
# n_layers = 8
# n_heads = 8
# multiple_of = 128
# dropout = 0.1


@dataclass
class OptimizerConfig:
    # adamw optimizer
    gradient_accumulation_steps: int = 4  # used to simulate larger batch sizes
    learning_rate: float = 1e-4  # max learning rate
    max_iters: int = 100000  # total number of training iterations
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0  # clip gradients at this value, or disable if == 0.0
    # learning rate decay settings
    decay_lr: bool = True  # whether to decay the learning rate
    warmup_iters: int = 1000  # how many steps to warm up for

    lr_decay_iters: int = 100000  # should be ~= max_iters per Chinchilla
    min_lr: float = (
        0.0  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
    )


@dataclass
class TrainerArgs:
    # Input / Output
    io_conf: IOConfig

    # Loader Configs
    loader_conf: LoaderConfig

    # Transformer Args
    model_conf: ModelArgs
    context_conf: ContextArgs

    # Optimizer
    optimizer_conf: OptimizerConfig

    run_name: str


class Trainer:
    def __init__(
        self, train_args: TrainerArgs, dtype: str = "float16", compile: bool = False
    ) -> None:
        self.train_conf = train_args
        self.dtype = dtype
        self.compile = compile
        # system
        self.run_name = train_args.run_name
        self.device = (
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )  # "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks

        self.CKPT_PT = f"{self.run_name}.pt"
        self.SNAPSHOT_PT = f"snapshot_{self.run_name}.pt"

    def _init_ddp_if_possible(self):
        # various inits, derived attributes, I/O setup
        self.ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
        if self.ddp:
            logger.info(f"Using ddp!")
            init_process_group(backend="nccl")
            self.ddp_rank = int(os.environ["RANK"])
            self.ddp_local_rank = int(os.environ["LOCAL_RANK"])
            self.ddp_world_size = int(os.environ["WORLD_SIZE"])
            logger.info(f"{self.ddp_rank}, {self.ddp_local_rank},{self.ddp_world_size}")

            self.device = f"cuda:{self.ddp_local_rank}"
            torch.cuda.set_device(self.device)
            self.master_process = (
                self.ddp_rank == 0
            )  # this process will do logging, checkpointing etc.

            logger.info(f"Is master process {self.device}? {self.master_process}")
            self.seed_offset = self.ddp_rank  # each process gets a different seed
            # world_size number of processes will be training simultaneously, so we can scale
            # down the desired gradient accumulation iterations per process proportionally
            assert (
                self.train_conf.optimizer_conf.gradient_accumulation_steps
                % self.ddp_world_size
                == 0
            )
            self.train_conf.optimizer_conf.gradient_accumulation_steps //= (
                self.ddp_world_size
            )
        else:
            # if not ddp, we are running on a single gpu, and one process
            self.master_process = True
            self.seed_offset = 0
            self.ddp_world_size = 1

    def _init_train(self):
        self.tokens_per_iter = (
            self.train_conf.optimizer_conf.gradient_accumulation_steps
            * self.ddp_world_size
            * self.train_conf.loader_conf.batch_size
            * self.train_conf.loader_conf.max_seq_len
        )
        if self.master_process:
            logger.info(f"tokens per iteration will be: {self.tokens_per_iter:,}")
            logger.info(
                f"breaks down as: {self.train_conf.optimizer_conf.gradient_accumulation_steps} grad accum steps * {self.ddp_world_size} processes * {self.train_conf.loader_conf.batch_size} batch size * {self.train_conf.loader_conf.max_seq_len } max seq len"
            )

        if self.master_process:
            os.makedirs(self.train_conf.io_conf.out_dir, exist_ok=True)

        torch.manual_seed(1337 + self.seed_offset)
        np.random.seed(1337 + self.seed_offset)
        torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
        self.device_type = (
            "cuda" if "cuda" in self.device else "cpu"
        )  # for later use in torch.autocast
        # note: float16 data type will automatically use a GradScaler
        ptdtype = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[self.dtype]
        self.ctx = (
            nullcontext()
            if self.device_type == "cpu"
            else torch.amp.autocast(device_type=self.device_type, dtype=ptdtype)
        )
        # task-specific setup
        task = {"smiles": SmilesTask}[self.train_conf.loader_conf.dataset]
        self.iter_batches = partial(
            task.iter_batches,
            batch_size=self.train_conf.loader_conf.batch_size,
            device=self.device,
            context_keys=self.train_conf.context_conf.context_keys,
            num_workers=0,
            dataset=self.train_conf.loader_conf.processed_dataset_ckpt,
            fragment_creator=fragment_creator_factory(
                self.train_conf.loader_conf.fragment_creator
            ),
        )
        # init these up here, can override if init_from='resume' (i.e. from a checkpoint)
        self.iter_num = 0
        self.best_val_loss = 1e9
        self.epoch = 1

        self.tokenizer = SmilesTokenizer()

        has_resumed = False
        if (
            self.train_conf.io_conf.init_from == "resume"
            or self.train_conf.io_conf.resume_when_snapshot_available
        ):
            snapshot_path = os.path.join(
                self.train_conf.io_conf.out_dir, self.SNAPSHOT_PT
            )
            if os.path.exists(snapshot_path):
                has_resumed = True
                logger.info(f"Resuming training from {self.train_conf.io_conf.out_dir}")
                # resume training from a checkpoint.
                ckpt_path = os.path.join(self.train_conf.io_conf.out_dir, self.CKPT_PT)
                self.model = Transformer.load(ckpt_path, device=self.device)
                snapshot = torch.load(snapshot_path, map_location=self.device)
                self.iter_num = snapshot["iter_num"]
                self.best_val_loss = snapshot["best_val_loss"]
                self.epoch = snapshot["epoch"]

        if self.train_conf.io_conf.init_from == "scratch" and not has_resumed:
            # init a new model from scratch
            logger.info("Initializing a new model from scratch")
            logger.info(self.device)

            model_conf = self.train_conf.model_conf
            model_conf.vocab_size = self.tokenizer.vocab_size

            self.model = Transformer(model_conf, self.train_conf.context_conf).to(
                self.device
            )
            logger.info(
                f"Number of params: {self.model.getNumberParams()} Number Trainable Params: {self.model.getNumberTrainableParams()}"
            )

        # else:
        #     raise ValueError(
        #         f"Could not find option: {self.train_conf.io_conf.init_from}. Use either 'scratch' or 'resume'"
        #     )

        self.model = self.model.to(self.device)

        # initialize a GradScaler. If enabled=False scaler is a no-op
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.dtype == "float16"))

        # optimizer
        self.optimizer = self.model.configure_optimizers(
            self.train_conf.optimizer_conf.weight_decay,
            self.train_conf.optimizer_conf.learning_rate,
            (
                self.train_conf.optimizer_conf.beta1,
                self.train_conf.optimizer_conf.beta2,
            ),
            self.device_type,
        )

        if (
            self.train_conf.io_conf.init_from == "resume"
            and "optimizer_state" in snapshot
        ):
            logger.info("Loading optimizer state from snapshot")
            self.optimizer.load_state_dict(snapshot["optimizer_state"])
        snapshot = None  # free up memory

        # compile the model
        if self.compile:
            logger.info("compiling the model... (takes a ~minute)")
            self.unoptimized_model = self.model
            # NOTE: This is REALLY REALLY slow in our case, as the shapes are different in each epoch.
            # So it recompiles every batch ._.
            self.model = torch.compile(
                self.model, dynamic=False
            )  # requires PyTorch 2.0

        # wrap model into DDP container
        if self.ddp:
            # Ignore the `freqs_cis` buffer so that DDP does not broadcast it at
            # construction time since NCCL does not support `ComplexFloat`
            prefix = "_orig_mod." if compile else ""
            self.model._ddp_params_and_buffers_to_ignore = {prefix + "freqs_cis"}
            self.model = DDP(self.model, device_ids=[self.ddp_local_rank])

    # helps estimate an arbitrarily accurate loss over either split using many batches
    @torch.no_grad()
    def estimate_loss(self):
        out = {}
        self.model.eval()
        for split in ["train", "val"]:
            batch_iter = self.iter_batches(split)
            losses = torch.zeros(self.train_conf.io_conf.eval_iters)  # keep on CPU
            for k in tqdm(
                range(self.train_conf.io_conf.eval_iters),
                total=self.train_conf.io_conf.eval_iters,
                desc="Eval",
            ):
                try:
                    X = next(batch_iter)
                    with self.ctx:
                        # logger.info(model)
                        # logger.info(X["src"].device)

                        logits = self.model(
                            X["src"],
                            targets=X["tgt"],
                            context=X["context"],
                            fragment=X["fragment"],
                        )

                        loss = self.raw_model.last_loss
                    losses[k] = loss.item()
                except StopIteration:
                    logger.info("Early Eval Stop")

            out[split] = losses.mean()
        self.model.train()
        return out

    # learning rate decay scheduler (cosine with warmup)
    def get_lr(self, it: int):
        warmup_iters = self.train_conf.optimizer_conf.warmup_iters
        learning_rate = self.train_conf.optimizer_conf.learning_rate
        lr_decay_iters = self.train_conf.optimizer_conf.lr_decay_iters
        min_lr = self.train_conf.optimizer_conf.min_lr

        # 1) linear warmup for warmup_iters steps
        if it < warmup_iters:
            return learning_rate * it / warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > lr_decay_iters:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return min_lr + coeff * (learning_rate - min_lr)

    def train(self):
        self._init_ddp_if_possible()
        self._init_train()

        # training loop
        train_batch_iter = self.iter_batches("train")
        X = next(train_batch_iter)  # fetch the very first batch
        t0 = time.time()
        local_iter_num = 0  # number of iterations in the lifetime of this process
        self.raw_model = (
            self.model.module if self.ddp else self.model
        )  # unwrap DDP container if needed
        running_mfu = -1.0

        gradient_accumulation_steps = (
            self.train_conf.optimizer_conf.gradient_accumulation_steps
        )
        while True:
            # determine and set the learning rate for this iteration
            lr = (
                self.get_lr(self.iter_num)
                if self.train_conf.optimizer_conf.decay_lr
                else self.train_conf.optimizer_conf.learning_rate
            )
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            # evaluate the loss on train/val sets and write checkpoints
            if (
                self.iter_num % self.train_conf.io_conf.eval_interval == 0
                and self.master_process
                and self.iter_num != 0
            ):
                logger.info(
                    f"Estimating loss for master_process({self.master_process}) on iter {self.iter_num}"
                )
                losses = self.estimate_loss()
                logger.info(
                    f"step {self.iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
                )
                log_dict = {
                    "iter": self.iter_num,
                    "tokens": self.iter_num * self.tokens_per_iter,
                    "loss/train": losses["train"],
                    "loss/val": losses["val"],
                    "lr": lr,
                    "mfu": running_mfu * 100,  # convert to percentage
                }
                logger.info(f"{log_dict}")

                if (
                    losses["val"] < self.best_val_loss
                    or self.train_conf.io_conf.always_save_checkpoint
                ):
                    self.best_val_loss = losses["val"]
                    if self.iter_num > 0:
                        logger.info(
                            f"saving checkpoint to {self.train_conf.io_conf.out_dir}"
                        )
                        self.raw_model.save(
                            os.path.join(self.train_conf.io_conf.out_dir, self.CKPT_PT)
                        )

                        torch.save(
                            {
                                "iter_num": self.iter_num,
                                "epoch": self.epoch,
                                "best_val_loss": self.best_val_loss,
                                "optimizer_state": self.optimizer.state_dict(),
                            },
                            os.path.join(
                                self.train_conf.io_conf.out_dir, self.SNAPSHOT_PT
                            ),
                        )

            if self.iter_num == 0 and self.train_conf.io_conf.eval_only:
                break

            # forward backward update, with optional gradient accumulation to simulate larger batch size
            # and using the GradScaler if data type is float16
            for micro_step in range(gradient_accumulation_steps):
                if self.ddp:
                    # in DDP training we only need to sync gradients at the last micro step.
                    # the official way to do this is with model.no_sync() context manager, but
                    # I really dislike that this bloats the code and forces us to repeat code
                    # looking at the source of that context manager, it just toggles this variable
                    self.model.require_backward_grad_sync = (
                        micro_step == gradient_accumulation_steps - 1
                    )
                with self.ctx:
                    context = X["context"]

                    fragment = X["fragment"]

                    # SCL (Stochastic context learning) algorithm
                    if np.random.random() < 0.15 or fragment is None:
                        fragment = None

                    # NOTE: random delete one context or more context columns
                    current_context_keys = list(context.keys())
                    for k in current_context_keys:
                        if np.random.random() < 0.15:
                            del context[k]

                    logits = self.model(
                        X["src"], targets=X["tgt"], context=context, fragment=fragment
                    )
                    loss = self.raw_model.last_loss
                    loss = loss / gradient_accumulation_steps
                # immediately async prefetch next batch while model is doing the forward pass on the GPU
                try:
                    X = next(train_batch_iter)

                except StopIteration:
                    # StopIteration is thrown if dataset ends
                    # reinitialize data loader
                    logger.info(f"Done Epoch {self.epoch}")
                    train_batch_iter = self.iter_batches("train")
                    X = next(train_batch_iter)
                    self.epoch += 1

                # backward pass, with gradient scaling if training in fp16
                self.scaler.scale(loss).backward()
                # logger.info(loss)
            # clip the gradient
            if self.train_conf.optimizer_conf.grad_clip != 0.0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.train_conf.optimizer_conf.grad_clip
                )
            # step the optimizer and scaler if training in fp16
            self.scaler.step(self.optimizer)
            self.scaler.update()
            # flush the gradients as soon as we can, no need for this memory anymore
            self.optimizer.zero_grad(set_to_none=True)

            # timing and logging
            t1 = time.time()
            dt = t1 - t0
            t0 = t1

            if (
                self.iter_num % self.train_conf.io_conf.log_interval == 0
                and self.master_process
            ):
                # get loss as float, scale up due to the divide above. note: this is a CPU-GPU sync point
                lossf = loss.item() * gradient_accumulation_steps
                if local_iter_num >= 5:  # let the training loop settle a bit
                    mfu = self.raw_model.estimate_mfu(
                        self.train_conf.loader_conf.batch_size
                        * gradient_accumulation_steps,
                        dt,
                    )
                    running_mfu = (
                        mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
                    )
                logger.info(
                    f"{self.iter_num} | loss {lossf:.4f} | lr {lr:e} | {dt*1000:.2f}ms | mfu {running_mfu*100:.2f}%"
                )
            self.iter_num += 1
            local_iter_num += 1

            # termination conditions

            if self.iter_num > self.train_conf.optimizer_conf.max_iters:
                logger.info("Done with training iters!")
                break

        if self.ddp:
            destroy_process_group()


if __name__ == "__main__":
    pass
