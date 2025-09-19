"""
MNIST trainer — DDP-ready, production-grade baseline

Properties
- Single-file. No persistence.
- Deterministic by default. Seeded per-rank.
- Native PyTorch DistributedDataParallel via torchrun.
- AMP on CUDA, grad clipping, cosine LR.
- Config via CLI. Clean rank-0 logging. Optional torch.compile.

Run (multi-GPU example)
    torchrun --standalone --nnodes=1 --nproc_per_node=4 mnist_trainer_pro.py \
        --epochs 5 --batch-size 128 --lr 1e-3

Notes
    - Use NCCL backend on Linux with CUDA. Set env var NVIDIA_VISIBLE_DEVICES as needed.
    - Exits non-zero on NaN loss.
"""
from __future__ import annotations

import argparse
import dataclasses
import logging
import math
import os
import random
import sys
from dataclasses import dataclass
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import wrap  # Optional for fine-grained control
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload, BackwardPrefetch, ShardingStrategy
from torch.distributed.fsdp import MixedPrecision
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms

# ---------------------------
# Config
# ---------------------------
@dataclass
class Config:
    data_dir: str = "../ddplearning/data"
    batch_size: int = 128
    test_batch_size: int = 1024
    epochs: int = 5
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip_norm: float | None = 1.0
    seed: int = 42
    workers: int = max(1, os.cpu_count() // 2)
    deterministic: bool = True
    amp: bool = True
    compile: bool = False  # requires PyTorch 2.x
    log_interval: int = 100
    dist_backend: str = "nccl"  # or "gloo" for CPU
    scale_lr_by_world_size: bool = True


# ---------------------------
# Distributed helpers
# ---------------------------

def is_dist() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1 and dist.is_available()


def get_rank() -> int:
    if is_dist() and dist.is_initialized():
        return dist.get_rank()
    return int(os.environ.get("RANK", "0"))


def get_world_size() -> int:
    if is_dist() and dist.is_initialized():
        return dist.get_world_size()
    return int(os.environ.get("WORLD_SIZE", "1"))


def setup_distributed(cfg: Config) -> Optional[int]:
    """Initialize process group. Returns local_rank or None if not distributed."""
    if not is_dist():
        return None

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    dist.init_process_group(backend=cfg.dist_backend)

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    return local_rank


def cleanup_distributed() -> None:
    if is_dist() and dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------
# Logging
# ---------------------------

def setup_logging() -> None:
    level = logging.INFO if get_rank() == 0 else logging.WARN
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


# ---------------------------
# Reproducibility
# ---------------------------

def set_seed(seed: int, deterministic: bool = True) -> None:
    # Make per-rank seed for stochastic ops but keep global reproducibility
    rank = get_rank()
    final_seed = seed + rank

    random.seed(final_seed)
    os.environ["PYTHONHASHSEED"] = str(final_seed)
    torch.manual_seed(final_seed)
    torch.cuda.manual_seed_all(final_seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


# ---------------------------
# Model
# ---------------------------
class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B,1,28,28] -> [B,10]
        return self.net(x)

# ---------------------------
# Data
# ---------------------------

def build_dataloaders(cfg: Config) -> Tuple[DataLoader, DataLoader, Optional[DistributedSampler]]:
    # Standard MNIST mean/std
    mean, std = (0.1307,), (0.3081,)

    train_tfms = transforms.Compose([
        transforms.RandomRotation(10),  # light augmentation
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_tfms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = datasets.MNIST(cfg.data_dir, train=True, download=False, transform=train_tfms)
    test_ds = datasets.MNIST(cfg.data_dir, train=False, download=False, transform=test_tfms)

    sampler = None
    if is_dist():
        sampler = DistributedSampler(train_ds, num_replicas=get_world_size(), rank=get_rank(), shuffle=True, drop_last=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=cfg.workers,
        pin_memory=True,
        persistent_workers=cfg.workers > 0,
        drop_last=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.test_batch_size,
        shuffle=False,
        num_workers=cfg.workers,
        pin_memory=True,
        persistent_workers=cfg.workers > 0,
    )

    return train_loader, test_loader, sampler


# ---------------------------
# Train / Eval
# ---------------------------

def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,   # <— pass scheduler in
    device: torch.device,
    scaler: torch.amp.GradScaler | None,
    cfg: Config,
    epoch: int,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    running_acc = 0.0

    for step, (images, targets) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if cfg.amp and torch.cuda.is_available():
            with torch.amp.autocast(device_type="cuda"):
                logits = model(images)
                loss = criterion(logits, targets)
        else:
            logits = model(images)
            loss = criterion(logits, targets)

        is_finite = torch.isfinite(loss.detach())
        abort_flag = torch.tensor(
            [0 if is_finite else 1],
            device=device,
            dtype=torch.int32
        )
        if is_dist():
            dist.all_reduce(abort_flag, op=dist.ReduceOp.SUM)
            
        if abort_flag.item() > 0:
            logging.error("Non-finite loss detected on at least one rank. Aborting cleanly.")
            raise RuntimeError("non-finite loss")

        if scaler is not None:
            scaler.scale(loss).backward()
            if cfg.grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if cfg.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optimizer.step()

        # step scheduler here, once per batch
        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item()
        running_acc += accuracy(logits, targets)

        if step % cfg.log_interval == 0 or step == 1:
            if get_rank() == 0:
                avg_loss = running_loss / step
                avg_acc = running_acc / step
                logging.info(
                    f"epoch {epoch} | step {step}/{len(loader)} "
                    f"| loss {avg_loss:.4f} | acc {avg_acc:.4f} "
                    f"| lr {optimizer.param_groups[0]['lr']:.2e}"
                )

    return running_loss / len(loader), running_acc / len(loader)

def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    cfg: Config,
) -> Tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    acc_sum = 0.0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if cfg.amp and torch.cuda.is_available():
                with torch.amp.autocast(device_type = "cuda"):
                    logits = model(images)
                    loss = criterion(logits, targets)
            else:
                logits = model(images)
                loss = criterion(logits, targets)

            loss_sum += loss.item()
            acc_sum += accuracy(logits, targets)

    return loss_sum / len(loader), acc_sum / len(loader)


# ---------------------------
# Orchestration
# ---------------------------

def train(cfg: Config) -> None:
    local_rank = setup_distributed(cfg)
    setup_logging()
    set_seed(cfg.seed, cfg.deterministic)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}" if local_rank is not None else "cuda")
    else:
        device = torch.device("cpu")
    if get_rank() == 0:
        logging.info(f"device: {device} | world_size: {get_world_size()}")

    train_loader, test_loader, train_sampler = build_dataloaders(cfg)

    model = MLP().to(device)

    if cfg.compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)  # type: ignore[attr-defined]
            if get_rank() == 0:
                logging.info("torch.compile enabled")
        except Exception as e:
            if get_rank() == 0:
                logging.warning(f"torch.compile unavailable: {e}")

    if is_dist():
        fsdp_config = dict(
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            cpu_offload=CPUOffload(offload_params=False),
            auto_wrap_policy=None,  # Can use `transformer_auto_wrap_policy` or custom
            mixed_precision=MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16,
            ) if cfg.amp else None,
            device_id=device,
        )
        model = FSDP(model, **fsdp_config)


    criterion = nn.CrossEntropyLoss()

    world_size = get_world_size()
    base_lr = cfg.lr * world_size if cfg.scale_lr_by_world_size else cfg.lr
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=cfg.weight_decay)

    total_steps = len(train_loader) * cfg.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, total_steps))

    scaler = None
    if cfg.amp and torch.cuda.is_available():
        scaler = torch.amp.GradScaler(device = "cuda", enabled=cfg.amp and torch.cuda.is_available())

    for epoch in range(1, cfg.epochs + 1):
        if is_dist() and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, scaler, cfg, epoch
        )
        val_loss, val_acc = evaluate(model, test_loader, criterion, device, cfg)

        if get_rank() == 0:
            print(
                f"epoch {epoch} done | train_loss {train_loss:.4f} | train_acc {train_acc:.4f} | val_loss {val_loss:.4f} | val_acc {val_acc:.4f}"
            )

    cleanup_distributed()


# ---------------------------
# CLI
# ---------------------------

def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="MNIST trainer (no persistence), DDP-ready")
    parser.add_argument("--data-dir", type=str, default=Config.data_dir)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--test-batch-size", type=int, default=Config.test_batch_size)
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--lr", type=float, default=Config.lr)
    parser.add_argument("--weight-decay", type=float, default=Config.weight_decay)
    parser.add_argument("--grad-clip-norm", type=float, default=Config.grad_clip_norm)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--workers", type=int, default=Config.workers)
    parser.add_argument("--no-deterministic", action="store_true", help="disable deterministic mode")
    parser.add_argument("--no-amp", action="store_true", help="disable mixed precision")
    parser.add_argument("--compile", action="store_true", help="enable torch.compile if available")
    parser.add_argument("--log-interval", type=int, default=Config.log_interval)
    parser.add_argument("--dist-backend", type=str, default=Config.dist_backend, choices=["nccl", "gloo", "mpi"])
    parser.add_argument("--no-scale-lr", action="store_true", help="do not scale LR by world size")

    args = parser.parse_args()

    cfg = Config(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip_norm=None if math.isclose(args.grad_clip_norm, 0.0) else args.grad_clip_norm,
        seed=args.seed,
        workers=args.workers,
        deterministic=not args.no_deterministic,
        amp=not args.no_amp,
        compile=args.compile,
        log_interval=args.log_interval,
        dist_backend=args.dist_backend,
        scale_lr_by_world_size=not args.no_scale_lr,
    )
    return cfg


def main() -> None:
    cfg = parse_args()
    try:
        train(cfg)
    finally:
        # we never strand process group even if an exception is raised
        cleanup_distributed()

if __name__ == "__main__":
    main()
