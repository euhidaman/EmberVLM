"""
Training Utilities for EmberVLM

Includes distributed training setup, optimizer/scheduler creation,
checkpointing, and other training utilities.
"""

import os
import math
import json
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
from typing import Optional, Dict, Any, Tuple, List, Union
from dataclasses import dataclass, field
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration."""

    # Basic
    seed: int = 42
    output_dir: str = "./outputs"

    # Distributed
    distributed: bool = True
    backend: str = "nccl"
    find_unused_parameters: bool = False

    # Precision
    mixed_precision: str = "bf16"  # "fp32", "fp16", "bf16"
    gradient_checkpointing: bool = True

    # Optimizer
    optimizer: str = "adamw"
    learning_rate: float = 2e-4
    min_learning_rate: float = 2e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8

    # Scheduler
    scheduler: str = "cosine"
    warmup_steps: int = 500
    num_training_steps: int = 10000

    # Batch
    batch_size: int = 128
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0

    # Checkpointing
    save_steps: int = 500
    max_checkpoints: int = 1
    save_optimizer: bool = True

    # Logging
    log_steps: int = 50
    eval_steps: int = 500

    # HuggingFace
    push_to_hub: bool = False
    hub_model_id: str = "embervlm"
    hub_token: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})


def setup_distributed() -> Tuple[int, int, int]:
    """
    Setup distributed training environment.

    Returns:
        Tuple of (rank, local_rank, world_size)
    """
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    elif "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        local_rank = rank % torch.cuda.device_count()
        world_size = int(os.environ["SLURM_NTASKS"])
    else:
        rank = 0
        local_rank = 0
        world_size = 1

    if world_size > 1:
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
        )
        torch.cuda.set_device(local_rank)

    return rank, local_rank, world_size


def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    """Check if this is the main process."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_rank() -> int:
    """Get current process rank."""
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    """Get world size."""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def barrier():
    """Synchronize all processes."""
    if dist.is_initialized():
        dist.barrier()


def reduce_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Reduce tensor across all processes."""
    if not dist.is_initialized():
        return tensor

    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= get_world_size()
    return rt


def set_seed(seed: int, rank: int = 0):
    """Set random seed for reproducibility."""
    import random
    import numpy as np

    seed = seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For deterministic operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_optimizer(
    model: nn.Module,
    config: TrainingConfig,
) -> torch.optim.Optimizer:
    """
    Create optimizer for model.

    Args:
        model: Model to optimize
        config: Training configuration

    Returns:
        Configured optimizer
    """
    # Separate parameters for weight decay
    decay_params = []
    no_decay_params = []

    no_decay_names = ['bias', 'LayerNorm', 'layer_norm', 'ln_']

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if any(nd in name for nd in no_decay_names):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer_groups = [
        {'params': decay_params, 'weight_decay': config.weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ]

    if config.optimizer.lower() == 'adamw':
        optimizer = torch.optim.AdamW(
            optimizer_groups,
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
        )
    elif config.optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam(
            optimizer_groups,
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")

    return optimizer


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Create learning rate scheduler.

    Args:
        optimizer: Optimizer to schedule
        config: Training configuration

    Returns:
        Configured scheduler
    """
    if config.scheduler.lower() == 'cosine':
        # Cosine schedule with warmup
        def lr_lambda(current_step):
            if current_step < config.warmup_steps:
                return float(current_step) / float(max(1, config.warmup_steps))

            progress = float(current_step - config.warmup_steps) / float(
                max(1, config.num_training_steps - config.warmup_steps)
            )

            min_lr_ratio = config.min_learning_rate / config.learning_rate
            return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    elif config.scheduler.lower() == 'linear':
        # Linear schedule with warmup
        def lr_lambda(current_step):
            if current_step < config.warmup_steps:
                return float(current_step) / float(max(1, config.warmup_steps))

            return max(
                config.min_learning_rate / config.learning_rate,
                float(config.num_training_steps - current_step) /
                float(max(1, config.num_training_steps - config.warmup_steps))
            )

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    elif config.scheduler.lower() == 'constant':
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

    else:
        raise ValueError(f"Unknown scheduler: {config.scheduler}")

    return scheduler


def get_grad_scaler(config: TrainingConfig) -> Optional[GradScaler]:
    """
    Create gradient scaler for mixed precision.

    Args:
        config: Training configuration

    Returns:
        GradScaler or None
    """
    if config.mixed_precision == 'fp16':
        return GradScaler()
    return None


def get_autocast_context(config: TrainingConfig):
    """
    Get autocast context for mixed precision.

    Args:
        config: Training configuration

    Returns:
        Autocast context
    """
    if config.mixed_precision == 'bf16':
        return autocast(dtype=torch.bfloat16)
    elif config.mixed_precision == 'fp16':
        return autocast(dtype=torch.float16)
    else:
        return torch.cuda.amp.autocast(enabled=False)


def wrap_model_ddp(
    model: nn.Module,
    config: TrainingConfig,
    device: torch.device,
) -> nn.Module:
    """
    Wrap model with DistributedDataParallel.

    Args:
        model: Model to wrap
        config: Training configuration
        device: Device to use

    Returns:
        Wrapped model
    """
    model = model.to(device)

    if config.distributed and get_world_size() > 1:
        model = DDP(
            model,
            device_ids=[device.index],
            output_device=device.index,
            find_unused_parameters=config.find_unused_parameters,
        )

    return model


def unwrap_model(model: nn.Module) -> nn.Module:
    """Unwrap model from DDP wrapper."""
    if isinstance(model, DDP):
        return model.module
    return model


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    config: TrainingConfig,
    step: int,
    metrics: Dict[str, float],
    output_dir: str,
    scaler: Optional[GradScaler] = None,
):
    """
    Save training checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: Scheduler state
        config: Training configuration
        step: Current training step
        metrics: Current metrics
        output_dir: Directory to save to
        scaler: Optional gradient scaler
    """
    if not is_main_process():
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Unwrap model
    model_to_save = unwrap_model(model)

    # Save model
    model_path = output_dir / 'pytorch_model.bin'
    torch.save(model_to_save.state_dict(), model_path)

    # Save config
    if hasattr(model_to_save, 'config'):
        config_path = output_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(model_to_save.config.to_dict(), f, indent=2)

    # Save training state
    training_state = {
        'step': step,
        'metrics': metrics,
        'training_config': config.to_dict(),
    }

    if config.save_optimizer:
        training_state['optimizer'] = optimizer.state_dict()
        training_state['scheduler'] = scheduler.state_dict()
        if scaler is not None:
            training_state['scaler'] = scaler.state_dict()

    state_path = output_dir / 'training_state.pt'
    torch.save(training_state, state_path)

    logger.info(f"Saved checkpoint at step {step} to {output_dir}")


def load_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    checkpoint_dir: str,
    scaler: Optional[GradScaler] = None,
) -> int:
    """
    Load training checkpoint.

    Args:
        model: Model to load into
        optimizer: Optional optimizer to restore
        scheduler: Optional scheduler to restore
        checkpoint_dir: Directory to load from
        scaler: Optional gradient scaler

    Returns:
        Training step from checkpoint
    """
    checkpoint_dir = Path(checkpoint_dir)

    # Load model weights
    model_path = checkpoint_dir / 'pytorch_model.bin'
    if model_path.exists():
        model_to_load = unwrap_model(model)
        state_dict = torch.load(model_path, map_location='cpu')
        model_to_load.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded model weights from {model_path}")

    # Load training state
    state_path = checkpoint_dir / 'training_state.pt'
    step = 0

    if state_path.exists():
        training_state = torch.load(state_path, map_location='cpu')
        step = training_state.get('step', 0)

        if optimizer is not None and 'optimizer' in training_state:
            optimizer.load_state_dict(training_state['optimizer'])
            logger.info("Loaded optimizer state")

        if scheduler is not None and 'scheduler' in training_state:
            scheduler.load_state_dict(training_state['scheduler'])
            logger.info("Loaded scheduler state")

        if scaler is not None and 'scaler' in training_state:
            scaler.load_state_dict(training_state['scaler'])
            logger.info("Loaded scaler state")

    return step


class MetricTracker:
    """Track and aggregate training metrics."""

    def __init__(self):
        self.metrics = {}
        self.counts = {}

    def update(self, metrics: Dict[str, float]):
        """Update with new metrics."""
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = 0.0
                self.counts[key] = 0

            if isinstance(value, torch.Tensor):
                value = value.item()

            self.metrics[key] += value
            self.counts[key] += 1

    def get_average(self) -> Dict[str, float]:
        """Get averaged metrics."""
        return {
            key: self.metrics[key] / max(1, self.counts[key])
            for key in self.metrics
        }

    def reset(self):
        """Reset all metrics."""
        self.metrics = {}
        self.counts = {}


class EarlyStopping:
    """Early stopping callback."""

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = 'min',
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.

        Args:
            score: Current metric score

        Returns:
            True if should stop
        """
        if self.mode == 'min':
            score = -score

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

        return self.early_stop


def count_trainable_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_parameters(model: nn.Module) -> int:
    """Count total parameters in model."""
    return sum(p.numel() for p in model.parameters())


def print_trainable_parameters(model: nn.Module):
    """Print trainable parameter info."""
    trainable = count_trainable_parameters(model)
    total = count_total_parameters(model)

    print(f"Trainable parameters: {trainable:,} ({100 * trainable / total:.2f}%)")
    print(f"Total parameters: {total:,}")


def get_parameter_groups(model: nn.Module) -> Dict[str, List[str]]:
    """Get parameter groups by trainability."""
    trainable = []
    frozen = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable.append(name)
        else:
            frozen.append(name)

    return {'trainable': trainable, 'frozen': frozen}


def enable_gradient_checkpointing(model: nn.Module):
    """Enable gradient checkpointing for memory efficiency."""
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    elif hasattr(model, 'enable_gradient_checkpointing'):
        model.enable_gradient_checkpointing()
    else:
        # Manual gradient checkpointing
        for module in model.modules():
            if hasattr(module, 'gradient_checkpointing'):
                module.gradient_checkpointing = True


def compute_effective_batch_size(
    batch_size: int,
    gradient_accumulation_steps: int,
    world_size: int,
) -> int:
    """Compute effective batch size across all processes."""
    return batch_size * gradient_accumulation_steps * world_size

