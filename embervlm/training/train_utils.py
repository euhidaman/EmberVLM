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
from torch.cuda.amp import GradScaler
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

    if world_size > 1 and not dist.is_initialized():
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
        return torch.amp.autocast('cuda', dtype=torch.bfloat16)
    elif config.mixed_precision == 'fp16':
        return torch.amp.autocast('cuda', dtype=torch.float16)
    else:
        return torch.amp.autocast('cuda', enabled=False)


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
    # Model should already be on correct device, but ensure it
    if not next(model.parameters()).is_cuda:
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


def force_rebuild_embeddings(
    model: nn.Module,
    new_vocab_size: int,
    device: torch.device = None,
    logger: logging.Logger = None,
) -> nn.Module:
    """
    Force complete reconstruction of embedding layers to fix CUDA memory issues.

    This function creates brand new embedding and lm_head layers with fresh CUDA
    memory allocation, rather than relying on resize_token_embeddings which may
    leave stale CUDA state.

    Args:
        model: The EmberVLM model (unwrapped from DDP)
        new_vocab_size: Target vocabulary size
        device: Target device (if None, uses CPU for safe reconstruction)
        logger: Logger for status messages

    Returns:
        Model with reconstructed embedding layers
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Move model to CPU for safe memory operations
    original_device = next(model.parameters()).device
    model = model.cpu()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    logger.info(f"🔧 Force rebuilding embeddings to size {new_vocab_size}...")

    # Find the language model and its embedding layer
    lang_model = None
    embed_layer = None
    embed_attr_path = None

    if hasattr(model, 'language_model'):
        lang_model = model.language_model

        # Check for HuggingFace-style model (PretrainedTinyLLMBackbone)
        if hasattr(lang_model, 'model'):
            inner_model = lang_model.model

            # GPT-2 style: transformer.wte
            if hasattr(inner_model, 'transformer') and hasattr(inner_model.transformer, 'wte'):
                embed_layer = inner_model.transformer.wte
                embed_attr_path = 'language_model.model.transformer.wte'
            # Generic: get_input_embeddings
            elif hasattr(inner_model, 'get_input_embeddings'):
                embed_layer = inner_model.get_input_embeddings()
                embed_attr_path = 'language_model.model (via get_input_embeddings)'

        # Check for custom TinyLLMBackbone
        elif hasattr(lang_model, 'model') and hasattr(lang_model.model, 'transformer'):
            if hasattr(lang_model.model.transformer, 'wte'):
                embed_layer = lang_model.model.transformer.wte
                embed_attr_path = 'language_model.model.transformer.wte'

        # Direct embedding access
        elif hasattr(lang_model, 'get_input_embeddings'):
            embed_layer = lang_model.get_input_embeddings()
            embed_attr_path = 'language_model (via get_input_embeddings)'

    if embed_layer is None:
        raise RuntimeError("Could not find embedding layer in model")

    logger.info(f"  Found embedding layer at: {embed_attr_path}")

    # Get current embedding properties
    old_vocab_size = embed_layer.weight.shape[0]
    embed_dim = embed_layer.weight.shape[1]
    old_weights = embed_layer.weight.data.clone()

    logger.info(f"  Current embedding: vocab_size={old_vocab_size}, embed_dim={embed_dim}")
    logger.info(f"  Target embedding: vocab_size={new_vocab_size}, embed_dim={embed_dim}")

    # Create brand new embedding layer
    new_embed = nn.Embedding(new_vocab_size, embed_dim)

    # Initialize with small random values
    nn.init.normal_(new_embed.weight, mean=0.0, std=0.02)

    # Copy old weights
    copy_size = min(old_vocab_size, new_vocab_size)
    with torch.no_grad():
        new_embed.weight[:copy_size] = old_weights[:copy_size]

    logger.info(f"  ✓ Created new embedding layer, copied {copy_size} token embeddings")

    # Replace the embedding layer
    if hasattr(model.language_model, 'model'):
        inner_model = model.language_model.model

        if hasattr(inner_model, 'transformer') and hasattr(inner_model.transformer, 'wte'):
            inner_model.transformer.wte = new_embed
            logger.info(f"  ✓ Replaced transformer.wte")

        if hasattr(inner_model, 'set_input_embeddings'):
            inner_model.set_input_embeddings(new_embed)
            logger.info(f"  ✓ Called set_input_embeddings()")

        # Also rebuild lm_head if it exists and is tied
        if hasattr(inner_model, 'lm_head'):
            old_lm_head = inner_model.lm_head
            new_lm_head = nn.Linear(old_lm_head.in_features, new_vocab_size, bias=old_lm_head.bias is not None)

            # Initialize
            nn.init.normal_(new_lm_head.weight, mean=0.0, std=0.02)
            if new_lm_head.bias is not None:
                nn.init.zeros_(new_lm_head.bias)

            # Copy old weights
            with torch.no_grad():
                copy_size_lm = min(old_lm_head.weight.shape[0], new_vocab_size)
                new_lm_head.weight[:copy_size_lm] = old_lm_head.weight[:copy_size_lm]
                if old_lm_head.bias is not None and new_lm_head.bias is not None:
                    new_lm_head.bias[:copy_size_lm] = old_lm_head.bias[:copy_size_lm]

            inner_model.lm_head = new_lm_head
            logger.info(f"  ✓ Rebuilt lm_head to size {new_vocab_size}")

        # Update config
        if hasattr(inner_model, 'config'):
            inner_model.config.vocab_size = new_vocab_size
            logger.info(f"  ✓ Updated inner model config.vocab_size")

    elif hasattr(model.language_model, 'set_input_embeddings'):
        model.language_model.set_input_embeddings(new_embed)
        logger.info(f"  ✓ Called language_model.set_input_embeddings()")

    # Update language_model config
    if hasattr(model.language_model, 'config'):
        model.language_model.config.vocab_size = new_vocab_size
        logger.info(f"  ✓ Updated language_model.config.vocab_size")

    if hasattr(model.language_model, 'hf_config'):
        model.language_model.hf_config.vocab_size = new_vocab_size
        logger.info(f"  ✓ Updated language_model.hf_config.vocab_size")

    # Update main model config
    if hasattr(model, 'config'):
        model.config.language_vocab_size = new_vocab_size
        logger.info(f"  ✓ Updated model.config.language_vocab_size")

    # Move to target device
    target_device = device if device is not None else original_device
    model = model.to(target_device)

    # Force CUDA synchronization
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    # Verify the rebuild worked
    verify_embed = None
    if hasattr(model.language_model, 'get_input_embeddings'):
        verify_embed = model.language_model.get_input_embeddings()
    elif hasattr(model.language_model, 'model'):
        if hasattr(model.language_model.model, 'get_input_embeddings'):
            verify_embed = model.language_model.model.get_input_embeddings()

    if verify_embed is not None:
        actual_size = verify_embed.weight.shape[0]
        if actual_size != new_vocab_size:
            raise RuntimeError(
                f"Embedding rebuild verification failed! "
                f"Expected {new_vocab_size}, got {actual_size}"
            )
        logger.info(f"  ✓ Verified embedding size: {actual_size}")

    logger.info(f"✅ Embedding reconstruction complete on device {target_device}")

    return model


def validate_tensor_bounds(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    vocab_size: int,
    logger_instance: logging.Logger = None,
    clamp: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Validate and optionally clamp input_ids and labels to be within vocabulary bounds.

    This is a failsafe utility to prevent CUDA index out-of-bounds errors.
    Should be called before any embedding lookup operation.

    Args:
        input_ids: Token IDs tensor
        labels: Labels tensor (may contain -100 for ignore)
        vocab_size: Maximum valid token ID + 1
        logger_instance: Logger for warnings
        clamp: Whether to clamp out-of-bounds values (True) or raise error (False)

    Returns:
        Tuple of (validated_input_ids, validated_labels)
    """
    if logger_instance is None:
        logger_instance = logger

    modified = False

    # Validate input_ids
    if input_ids is not None:
        max_token_id = input_ids.max().item()
        min_token_id = input_ids.min().item()

        if max_token_id >= vocab_size:
            msg = f"input_ids max={max_token_id} >= vocab_size={vocab_size}"
            if clamp:
                logger_instance.warning(f"⚠️ {msg} - Clamping to valid range")
                input_ids = torch.clamp(input_ids, max=vocab_size - 1)
                modified = True
            else:
                raise ValueError(f"❌ CRITICAL: {msg}")

        if min_token_id < 0:
            msg = f"input_ids min={min_token_id} < 0"
            if clamp:
                logger_instance.warning(f"⚠️ {msg} - Clamping to valid range")
                input_ids = torch.clamp(input_ids, min=0)
                modified = True
            else:
                raise ValueError(f"❌ CRITICAL: {msg}")

    # Validate labels (preserve -100 ignore index)
    if labels is not None:
        valid_labels_mask = labels != -100
        if valid_labels_mask.any():
            valid_labels = labels[valid_labels_mask]
            max_label = valid_labels.max().item()
            min_label = valid_labels.min().item()

            if max_label >= vocab_size or min_label < 0:
                if clamp:
                    if max_label >= vocab_size:
                        logger_instance.warning(
                            f"⚠️ labels max={max_label} >= vocab_size={vocab_size} - Clamping"
                        )
                    if min_label < 0:
                        logger_instance.warning(
                            f"⚠️ labels min={min_label} < 0 - Clamping"
                        )
                    labels = torch.where(
                        valid_labels_mask,
                        torch.clamp(labels, min=0, max=vocab_size - 1),
                        labels
                    )
                    modified = True
                else:
                    raise ValueError(
                        f"❌ CRITICAL: labels out of bounds "
                        f"(min={min_label}, max={max_label}, vocab_size={vocab_size})"
                    )

    if modified:
        logger_instance.debug(f"Tensors validated and clamped to range [0, {vocab_size - 1}]")

    return input_ids, labels


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

