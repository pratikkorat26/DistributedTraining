from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import random
import sys
from dataclasses import dataclass
from functools import partial
from typing import Optional, Tuple, Type

# Silence HF tokenizers fork warning in multi-proc
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from datasets import load_dataset, load_from_disk
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, DistributedSampler
from torch.distributed.checkpoint.stateful import Stateful
# --- FSDP ---
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    ShardingStrategy,
)

from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

# New checkpoint state-dict helpers (future-proof)
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict, StateDictOptions

# --- HF ---
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding


# =========================
# Config
# =========================
@dataclass
class Config:
    # Model & Data
    model_name: str = "distilbert-base-uncased" # Changed default for faster example
    data_dir: str = ".data/imdb"
    # Train
    batch_size: int = 32
    test_batch_size: int = 64
    epochs: int = 3
    lr: float = 3e-5
    weight_decay: float = 0.01
    grad_clip_norm: Optional[float] = 1.0
    # System
    seed: int = 42
    workers: int = max(1, os.cpu_count() // 2)
    deterministic: bool = True
    amp: bool = True
    compile: bool = False
    dist_backend: str = "nccl"
    # Prod
    output_dir: str = "./output"
    run_name: str = "fsdp-run1"
    log_interval: int = 20
    resume_from_checkpoint: Optional[str] = None

# =========================
# Model Agnostic Helper
# =========================
def get_transformer_layer_class(model: nn.Module) -> Type[nn.Module]:
    """
    Inspects a given Hugging Face model to find its core transformer layer class.
    This is used for FSDP's auto_wrap_policy.
    """
    
    found_class = None
    
    # Heuristics to find the transformer layer class
    # Common names for transformer blocks in Hugging Face models
    layer_class_names = ["BertLayer", "GPT2Block", "RobertaLayer", "DecoderLayer", "TransformerBlock"]
    
    for name, module in model.named_modules():
        module_class_name = module.__class__.__name__
        if module_class_name in layer_class_names:
            found_class = module.__class__
            break
            
    if found_class is None:
        raise ValueError(
            f"Could not find a known transformer layer class in the model. "
            f"Please manually inspect the model architecture and add the layer class name "
            f"to the `layer_class_names` list in `get_transformer_layer_class`."
        )
        
    if get_rank() == 0:
        logging.info(f"Identified Transformer Layer Class for FSDP wrapping: {found_class.__name__}")
        
    return found_class

# Everything from here down is the same as before, no changes needed.

#########################################################
#### SAVE/LOAD UTILS
#########################################################
class TrainingState(Stateful):
    def __init__(self, model, optimizer, scheduler, epoch=0, best_acc=0.0):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epoch = epoch
        self.best_acc = best_acc
        self.state_dict_options = StateDictOptions(full_state_dict = True)

    def state_dict(self):
        model_state, optim_state = get_state_dict(self.model, self.optimizer, options = self.state_dict_options)
        return {
            "model": model_state,
            "optimizer": optim_state,
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "epoch": self.epoch,
            "best_acc": self.best_acc,
        }

    def load_state_dict(self, state_dict):
        set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optimizer"],
            options=self.state_dict_options,
        )
        if self.scheduler and state_dict.get("scheduler") is not None:
            self.scheduler.load_state_dict(state_dict["scheduler"])
        self.epoch = state_dict["epoch"]
        self.best_acc = state_dict["best_acc"]

# =========================
# Dist / Logging / Seed
# =========================
def is_dist() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1 and dist.is_available()

def get_rank() -> int:
    return dist.get_rank() if is_dist() and dist.is_initialized() else int(os.environ.get("RANK", "0"))

def get_world_size() -> int:
    return dist.get_world_size() if is_dist() and dist.is_initialized() else int(os.environ.get("WORLD_SIZE", "1"))

def setup_distributed(cfg: Config) -> Optional[int]:
    if not is_dist():
        return None
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=cfg.dist_backend)
    return local_rank

def cleanup_distributed() -> None:
    if is_dist() and dist.is_initialized():
        dist.destroy_process_group()

def setup_logging() -> None:
    level = logging.INFO if get_rank() == 0 else logging.WARN
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

def set_seed(seed: int, deterministic: bool = True) -> None:
    rank = get_rank()
    final = seed + rank
    random.seed(final)
    os.environ["PYTHONHASHSEED"] = str(final)
    torch.manual_seed(final)
    torch.cuda.manual_seed_all(final)
    torch.backends.cudnn.benchmark = not deterministic


# =========================
# Data
# =========================
def build_dataloaders(cfg: Config, local_rank: Optional[int]) -> Tuple[DataLoader, DataLoader, Optional[DistributedSampler]]:
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    # Pad token for open-ended models like GPT2
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def preprocess(ex):
        return tokenizer(ex["text"], truncation=True, max_length=512)

    proc_path = os.path.join(cfg.data_dir, f"imdb_tokenized_{cfg.model_name.split('/')[-1]}")
    if get_rank() == 0 and not os.path.exists(proc_path):
        os.makedirs(cfg.data_dir, exist_ok=True)
        raw = load_dataset("imdb", cache_dir=cfg.data_dir)
        tok = raw.map(preprocess, batched=True)
        tok.save_to_disk(proc_path)

    if is_dist():
        dist.barrier(device_ids=[local_rank])

    tok = load_from_disk(proc_path)
    train_ds = tok["train"].remove_columns(["text"]).with_format("torch")
    test_ds  = tok["test"].remove_columns(["text"]).with_format("torch")

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_sampler = DistributedSampler(train_ds) if is_dist() else None
    use_cuda = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=cfg.workers,
        pin_memory=use_cuda,
        collate_fn=collator,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.test_batch_size,
        shuffle=False,
        num_workers=cfg.workers,
        pin_memory=use_cuda,
        collate_fn=collator,
    )
    return train_loader, test_loader, train_sampler


# =========================
# Train / Eval
# =========================
def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return (logits.argmax(dim=-1) == targets).float().mean().item()

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler,
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler],
    cfg: Config,
    epoch: int,
    amp_dtype: torch.dtype,
    use_cuda: bool,
) -> Tuple[float, float]:
    model.train()
    loss_sum = 0.0
    acc_sum = 0.0
    for step, batch in enumerate(loader, 1):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=(cfg.amp and use_cuda)):
            out = model(**batch)
            loss = out.loss
        if scaler and scaler.is_enabled():
            scaler.scale(loss).backward()
            if cfg.grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if cfg.grad_clip_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optimizer.step()
        if scheduler:
            scheduler.step()
        loss_sum += loss.item()
        acc_sum  += accuracy(out.logits, batch["labels"])
        if (step % cfg.log_interval == 0 or step == 1) and get_rank() == 0:
            logging.info(
                f"epoch {epoch} | step {step}/{len(loader)} | loss {loss_sum/step:.4f} | "
                f"acc {acc_sum/step:.4f} | lr {optimizer.param_groups[0]['lr']:.2e}"
            )
    return loss_sum / len(loader), acc_sum / len(loader)

def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg: Config,
    amp_dtype: torch.dtype,
    use_cuda: bool,
) -> Tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    acc_sum = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=(cfg.amp and use_cuda)):
                out = model(**batch)
            loss_sum += out.loss.item()
            acc_sum  += accuracy(out.logits, batch["labels"])
    return loss_sum / len(loader), acc_sum / len(loader)


# =========================
# Checkpointing (best only) — new APIs
# =========================
def save_checkpoint(
    model: FSDP,
    optimizer: optim.Optimizer,
    scheduler,
    epoch: int,
    cfg: Config,
    local_rank: Optional[int],
    *,
    is_best: bool,
    best_acc: float
) -> None:
    """Saves a sharded, resumable checkpoint using torch.distributed.checkpoint."""
    if not is_best:
        return

    ckpt_dir = os.path.join(cfg.output_dir, cfg.run_name, "checkpoints")
    epoch_ckpt_dir = os.path.join(ckpt_dir, f"epoch_{epoch}")

    app_state = TrainingState(model, optimizer, scheduler, epoch, best_acc)

    dcp.save(
        state_dict={"app": app_state},
        checkpoint_id=epoch_ckpt_dir,
    )

    if get_rank() == 0:
        symlink_name = f"best.pt"
        best_symlink = os.path.join(ckpt_dir, symlink_name)
        if os.path.lexists(best_symlink):
            os.remove(best_symlink)
        os.symlink(os.path.basename(epoch_ckpt_dir), best_symlink)
        logging.info(f"✅ New best saved: {best_symlink} -> {os.path.basename(epoch_ckpt_dir)} (val_acc={best_acc:.4f})")

    if is_dist():
        dist.barrier(device_ids=[local_rank])


def load_checkpoint(model: FSDP, optimizer: optim.Optimizer, scheduler, ckpt_path: str) -> tuple[int, float]:
    """Loads a sharded checkpoint using torch.distributed.checkpoint."""
    app_state = TrainingState(model, optimizer, scheduler)
    dcp.load(
        state_dict={"app": app_state},
        checkpoint_id=ckpt_path,
    )
    start_epoch = app_state.epoch + 1
    best_acc = app_state.best_acc
    return start_epoch, best_acc


# =========================
# Orchestration
# =========================
def train(cfg: Config) -> None:
    local_rank = setup_distributed(cfg)
    setup_logging()
    set_seed(cfg.seed, cfg.deterministic)
    use_cuda = torch.cuda.is_available()

    if not is_dist() and not use_cuda:
        device = torch.device("cpu")
    else:
        gpu_id = local_rank if is_dist() else 0
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(device)

    if get_rank() == 0:
        logging.info(f"Device: {device} | World Size: {get_world_size()}")

    # data
    train_loader, test_loader, train_sampler = build_dataloaders(cfg, local_rank)

    # model init
    model_config = AutoConfig.from_pretrained(cfg.model_name, num_labels=2)

    if cfg.resume_from_checkpoint:
        if get_rank() == 0:
            logging.info("Resuming from checkpoint: initializing model from config for faster loading.")
        model = AutoModelForSequenceClassification.from_config(model_config)
    else:
        if get_rank() == 0:
            logging.info(f"Starting new run: initializing model with pre-trained weights from '{cfg.model_name}'.")
        model = AutoModelForSequenceClassification.from_pretrained(cfg.model_name, config=model_config)

    # --- ✅ MODEL-AGNOSTIC FSDP WRAPPING ---
    # Find the correct transformer layer class dynamically
    transformer_layer_cls = get_transformer_layer_class(model)
    auto_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={transformer_layer_cls})
    # --- END CHANGE ---
    
    # optional compile BEFORE wrapping
    if cfg.compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
        except Exception as e:
            logging.warning(f"torch.compile disabled: {e}")

    # AMP dtype selection
    use_bf16 = cfg.amp and use_cuda and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if cfg.amp and use_cuda else torch.float32)
    if get_rank() == 0 and cfg.amp:
        logging.info(f"AMP dtype: {'bf16' if amp_dtype is torch.bfloat16 else ('fp16' if amp_dtype is torch.float16 else 'fp32')}")

    mp = MixedPrecision(param_dtype=amp_dtype, reduce_dtype=amp_dtype, buffer_dtype=amp_dtype)

    if is_dist():
        model = FSDP(
            model,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            cpu_offload=CPUOffload(offload_params=False),
            mixed_precision=mp,
            device_id=local_rank,
            use_orig_params=True,
            auto_wrap_policy=auto_wrap_policy, # Use the dynamic policy
        )
    else:
        model.to(device)

    # opt/sched/scaler
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = len(train_loader) * cfg.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, total_steps))
    scaler = torch.amp.GradScaler(enabled=(cfg.amp and use_cuda and amp_dtype is torch.float16))

    # Resume logic
    best_acc = -1.0
    start_epoch = 1
    if cfg.resume_from_checkpoint:
        start_epoch, best_acc = load_checkpoint(model, optimizer, scheduler, cfg.resume_from_checkpoint)
        if get_rank() == 0:
            logging.info(f"Resumed from {cfg.resume_from_checkpoint} at epoch {start_epoch}")
            logging.info(f"Restored best validation accuracy to {best_acc:.4f}")

    # Training loop
    for epoch in range(start_epoch, cfg.epochs + 1):
        if train_sampler is not None and is_dist():
            train_sampler.set_epoch(epoch)

        tr_loss, tr_acc = run_epoch(model, train_loader, optimizer, scheduler, device, scaler, cfg, epoch, amp_dtype, use_cuda)
        va_loss, va_acc = evaluate(model, test_loader, device, cfg, amp_dtype, use_cuda)

        if get_rank() == 0:
            logging.info(
                f"--- epoch {epoch} done --- | "
                f"train_loss {tr_loss:.4f} | train_acc {tr_acc:.4f} | "
                f"val_loss {va_loss:.4f} | val_acc {va_acc:.4f}"
            )

        # Save best only
        is_best = va_acc > best_acc
        if is_best:
            best_acc = va_acc

        save_checkpoint(model, optimizer, scheduler, epoch, cfg, local_rank, is_best=is_best, best_acc=best_acc)

    cleanup_distributed()


# =========================
# CLI
# =========================
def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="IMDb FSDP Model-Agnostic Trainer")
    for field in dataclasses.fields(Config):
        name = f"--{field.name.replace('_', '-')}"
        default = field.default
        if isinstance(default, bool):
            parser.add_argument(name, action="store_true", default=default)
        else:
            if default is None:
                parser.add_argument(name, type=str, default=None)
            else:
                try:
                    arg_type = field.type if callable(field.type) else type(default)
                except Exception:
                    arg_type = type(default)
                parser.add_argument(name, type=arg_type, default=default)
    return Config(**vars(parser.parse_args()))


if __name__ == "__main__":
    cfg = parse_args()
    try:
        train(cfg)
    finally:
        cleanup_distributed()
