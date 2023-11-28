from trainer import (
    IOConfig,
    LoaderConfig,
    Trainer,
    TrainerArgs,
    ModelArgs,
    ContextArgs,
    OptimizerConfig,
)
from torch.distributed.elastic.multiprocessing.errors import record

import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import sys
import os
import torch


def setup_logger(run_name: str, log_path: str):
    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    if ddp:
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])

        formatter = logging.Formatter(
            f"[%(levelname)s] DDP[{ddp_rank},{ddp_local_rank},{ddp_world_size}] %(asctime)s - [%(filename)s:%(lineno)d]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    else:
        formatter = logging.Formatter(
            r"[%(levelname)s] %(asctime)s - [%(filename)s:%(lineno)d]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    os.makedirs(log_path, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(log_path, f"train_{run_name}.log"))
    file_handler.setFormatter(formatter)

    logging.basicConfig(level=logging.INFO, handlers=[stream_handler, file_handler])

    return logging.getLogger()


@record
@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    logger = setup_logger(
        cfg.get("run_name", "default"), cfg.get("io", {"out_dir": "out"})["out_dir"]
    )

    logger.info("Using config")
    logger.info(cfg)

    cfg = cfg["train"]
    io_conf = IOConfig(**cfg.get("io", {}))
    loader_conf = LoaderConfig(**cfg.get("loader", {}))
    model_args = ModelArgs(**cfg.get("model", {}))
    ctx_args = ContextArgs(**cfg.get("context", {}))
    optmizer_conf = OptimizerConfig(**cfg.get("optimizer", {}))
    train_args = TrainerArgs(
        io_conf=io_conf,
        loader_conf=loader_conf,
        model_conf=model_args,
        context_conf=ctx_args,
        optimizer_conf=optmizer_conf,
        run_name=cfg.get("label", "train_run"),
    )

    # When training on cpu / testing to not max out all cpu cores
    torch.set_num_threads(8)

    trainer = Trainer(
        train_args=train_args,
        dtype=cfg.get("dtype", "float16"),
        compile=cfg.get("compile", False),
    )
    should_profile = cfg.get("profile", False)

    if should_profile:
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as p:
            trainer.train()

        print(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))

    else:
        trainer.train()


if __name__ == "__main__":
    # python train.py train=llama2-M-Full train.model.dim=1024
    main()
