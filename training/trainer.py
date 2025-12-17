"""
EmberVLM Base Trainer
Distributed training infrastructure with DDP for 2×A100 GPUs.
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
import yaml
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Basic settings
    output_dir: str = "outputs"
    experiment_name: str = "embervlm"
    seed: int = 42

    # Training parameters
    num_epochs: int = 1
    batch_size_per_gpu: int = 32
    gradient_accumulation_steps: int = 4
    max_steps: int = -1  # -1 means use epochs

    # Optimizer
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0

    # Scheduler
    scheduler_type: str = "cosine"  # cosine, linear
    warmup_ratio: float = 0.05
    min_lr_ratio: float = 0.1

    # Mixed precision
    mixed_precision: str = "bf16"  # bf16, fp16, fp32

    # Distributed training
    local_rank: int = -1
    world_size: int = 1

    # Checkpointing
    save_steps: int = 500
    save_total_limit: int = 2
    resume_from_checkpoint: Optional[str] = None

    # Logging
    logging_steps: int = 10
    eval_steps: int = 1000

    # HuggingFace Hub
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
    hub_token: Optional[str] = None

    @classmethod
    def from_yaml(cls, path: str) -> 'TrainingConfig':
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        return cls(**config.get('training', config))

    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items()}


class DistributedTrainer:
    """
    Base trainer with DDP support for multi-GPU training.
    Designed for 2×A100 80GB configuration.
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        callbacks: Optional[List[Callable]] = None
    ):
        self.config = config
        self.callbacks = callbacks or []

        # Setup distributed training
        self._setup_distributed()

        # Move model to device and wrap with DDP
        self.device = torch.device(f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        if self.world_size > 1:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False
            )

        # Setup dataloaders
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader

        # Setup optimizer
        self.optimizer = optimizer or self._create_optimizer()

        # Setup scheduler
        self.scheduler = scheduler or self._create_scheduler()

        # Setup mixed precision
        self.scaler = GradScaler() if config.mixed_precision in ["fp16", "bf16"] else None
        self.autocast_dtype = self._get_autocast_dtype()

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_metric = None

        # Output directory
        self.output_dir = Path(config.output_dir) / config.experiment_name
        if self.is_main_process:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # Logging
        self._setup_logging()

    def _setup_distributed(self):
        """Initialize distributed training."""
        self.local_rank = int(os.environ.get("LOCAL_RANK", self.config.local_rank))
        self.world_size = int(os.environ.get("WORLD_SIZE", self.config.world_size))

        if self.world_size > 1 and not dist.is_initialized():
            dist.init_process_group(backend="nccl")
            torch.cuda.set_device(self.local_rank)

        self.is_main_process = self.local_rank in [-1, 0]

        if self.is_main_process:
            logger.info(f"Distributed training: world_size={self.world_size}, local_rank={self.local_rank}")

    def _setup_logging(self):
        """Setup logging for training."""
        if self.is_main_process:
            log_file = self.output_dir / "training.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create AdamW optimizer with weight decay."""
        # Separate parameters that should have weight decay
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "norm" in name or "layernorm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0}
        ]

        return AdamW(
            optimizer_groups,
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_epsilon
        )

    def _create_scheduler(self) -> Any:
        """Create learning rate scheduler."""
        num_training_steps = self._get_num_training_steps()
        num_warmup_steps = int(num_training_steps * self.config.warmup_ratio)

        if self.config.scheduler_type == "cosine":
            # Warmup + cosine decay
            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=num_warmup_steps
            )
            main_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=num_training_steps - num_warmup_steps,
                eta_min=self.config.learning_rate * self.config.min_lr_ratio
            )
            return SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[num_warmup_steps]
            )
        else:
            # Linear decay
            return LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=self.config.min_lr_ratio,
                total_iters=num_training_steps
            )

    def _get_num_training_steps(self) -> int:
        """Calculate total training steps."""
        if self.config.max_steps > 0:
            return self.config.max_steps

        num_batches = len(self.train_dataloader)
        steps_per_epoch = num_batches // self.config.gradient_accumulation_steps
        return steps_per_epoch * self.config.num_epochs

    def _get_autocast_dtype(self) -> torch.dtype:
        """Get dtype for autocast based on config."""
        if self.config.mixed_precision == "bf16":
            return torch.bfloat16
        elif self.config.mixed_precision == "fp16":
            return torch.float16
        return torch.float32

    def train(self) -> Dict[str, float]:
        """Main training loop."""
        logger.info("Starting training...")
        logger.info(f"  Num epochs = {self.config.num_epochs}")
        logger.info(f"  Batch size per GPU = {self.config.batch_size_per_gpu}")
        logger.info(f"  Gradient accumulation steps = {self.config.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {self._get_num_training_steps()}")

        # Resume from checkpoint if specified
        if self.config.resume_from_checkpoint:
            self._load_checkpoint(self.config.resume_from_checkpoint)

        # Training loop
        self.model.train()
        train_loss = 0.0
        step_loss = 0.0

        for epoch in range(self.epoch, self.config.num_epochs):
            self.epoch = epoch

            if hasattr(self.train_dataloader.sampler, 'set_epoch'):
                self.train_dataloader.sampler.set_epoch(epoch)

            epoch_start_time = time.time()

            for step, batch in enumerate(self.train_dataloader):
                # Move batch to device
                batch = self._prepare_batch(batch)

                # Forward pass with mixed precision
                with autocast(device_type='cuda', dtype=self.autocast_dtype):
                    outputs = self.model(**batch)
                    loss = outputs['loss'] if isinstance(outputs, dict) else outputs[0]
                    loss = loss / self.config.gradient_accumulation_steps

                # Backward pass
                if self.scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                step_loss += loss.item()

                # Optimizer step
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    if self.scaler:
                        self.scaler.unscale_(self.optimizer)

                    # Gradient clipping
                    if self.config.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.max_grad_norm
                        )

                    if self.scaler:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    self.global_step += 1
                    train_loss += step_loss

                    # Logging
                    if self.global_step % self.config.logging_steps == 0:
                        self._log_metrics({
                            'loss': step_loss,
                            'learning_rate': self.scheduler.get_last_lr()[0],
                            'epoch': epoch,
                            'step': self.global_step
                        })

                    step_loss = 0.0

                    # Evaluation
                    if self.eval_dataloader and self.global_step % self.config.eval_steps == 0:
                        eval_metrics = self.evaluate()
                        self._log_metrics(eval_metrics, prefix='eval')
                        self.model.train()

                    # Checkpointing
                    if self.global_step % self.config.save_steps == 0:
                        self._save_checkpoint()

                    # Callbacks
                    for callback in self.callbacks:
                        callback(self)

                    # Check max steps
                    if self.config.max_steps > 0 and self.global_step >= self.config.max_steps:
                        break

            epoch_time = time.time() - epoch_start_time
            logger.info(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s")

            if self.config.max_steps > 0 and self.global_step >= self.config.max_steps:
                break

        # Final save
        self._save_checkpoint(final=True)

        avg_train_loss = train_loss / self.global_step if self.global_step > 0 else 0
        return {'train_loss': avg_train_loss}

    def evaluate(self) -> Dict[str, float]:
        """Evaluation loop."""
        if not self.eval_dataloader:
            return {}

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.eval_dataloader:
                batch = self._prepare_batch(batch)

                with autocast(device_type='cuda', dtype=self.autocast_dtype):
                    outputs = self.model(**batch)
                    loss = outputs['loss'] if isinstance(outputs, dict) else outputs[0]

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0

        # Synchronize across processes
        if self.world_size > 1:
            loss_tensor = torch.tensor([avg_loss], device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            avg_loss = loss_tensor.item()

        return {'eval_loss': avg_loss}

    def _prepare_batch(self, batch: Dict) -> Dict:
        """Move batch tensors to device."""
        prepared = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                prepared[key] = value.to(self.device)
            else:
                prepared[key] = value
        return prepared

    def _log_metrics(self, metrics: Dict[str, float], prefix: str = 'train'):
        """Log metrics."""
        if not self.is_main_process:
            return

        log_str = f"[{prefix}] Step {self.global_step}: "
        log_str += ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        logger.info(log_str)

    def _save_checkpoint(self, final: bool = False):
        """Save model checkpoint."""
        if not self.is_main_process:
            return

        checkpoint_name = "final" if final else f"checkpoint-{self.global_step}"
        checkpoint_dir = self.output_dir / checkpoint_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Get unwrapped model
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model

        # Save model
        torch.save(model_to_save.state_dict(), checkpoint_dir / "pytorch_model.bin")

        # Save training state
        training_state = {
            'global_step': self.global_step,
            'epoch': self.epoch,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'config': self.config.to_dict()
        }
        torch.save(training_state, checkpoint_dir / "training_state.pt")

        # Save config
        with open(checkpoint_dir / "config.yaml", 'w') as f:
            yaml.dump(self.config.to_dict(), f)

        logger.info(f"Checkpoint saved to {checkpoint_dir}")

        # Cleanup old checkpoints
        self._cleanup_checkpoints()

    def _load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint for resuming training."""
        checkpoint_dir = Path(checkpoint_path)

        # Load model
        model_path = checkpoint_dir / "pytorch_model.bin"
        if model_path.exists():
            state_dict = torch.load(model_path, map_location=self.device)
            model_to_load = self.model.module if hasattr(self.model, 'module') else self.model
            model_to_load.load_state_dict(state_dict)

        # Load training state
        state_path = checkpoint_dir / "training_state.pt"
        if state_path.exists():
            training_state = torch.load(state_path, map_location=self.device)
            self.global_step = training_state['global_step']
            self.epoch = training_state['epoch']
            self.optimizer.load_state_dict(training_state['optimizer_state_dict'])
            if self.scheduler and training_state['scheduler_state_dict']:
                self.scheduler.load_state_dict(training_state['scheduler_state_dict'])

        logger.info(f"Resumed from checkpoint {checkpoint_path} at step {self.global_step}")

    def _cleanup_checkpoints(self):
        """Remove old checkpoints keeping only the most recent."""
        if self.config.save_total_limit <= 0:
            return

        checkpoints = sorted(
            [d for d in self.output_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
            key=lambda x: int(x.name.split("-")[1])
        )

        while len(checkpoints) > self.config.save_total_limit:
            old_checkpoint = checkpoints.pop(0)
            import shutil
            shutil.rmtree(old_checkpoint)
            logger.info(f"Removed old checkpoint: {old_checkpoint}")


def setup_training(
    model: nn.Module,
    train_dataset: Any,
    eval_dataset: Optional[Any] = None,
    config: Optional[TrainingConfig] = None,
    **kwargs
) -> DistributedTrainer:
    """
    Setup training with distributed configuration.

    Args:
        model: Model to train
        train_dataset: Training dataset
        eval_dataset: Optional evaluation dataset
        config: Training configuration

    Returns:
        Configured trainer
    """
    if config is None:
        config = TrainingConfig(**kwargs)

    # Create distributed sampler if needed
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", -1))

    if world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=local_rank,
            shuffle=True
        )
        eval_sampler = DistributedSampler(
            eval_dataset,
            num_replicas=world_size,
            rank=local_rank,
            shuffle=False
        ) if eval_dataset else None
    else:
        train_sampler = None
        eval_sampler = None

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size_per_gpu,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=config.batch_size_per_gpu,
        sampler=eval_sampler,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    ) if eval_dataset else None

    return DistributedTrainer(
        model=model,
        config=config,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader
    )


if __name__ == "__main__":
    # Test trainer setup
    print("Testing Distributed Trainer...")

    # Create dummy model
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(768, 768)

        def forward(self, input_ids, attention_mask=None, labels=None):
            x = self.linear(torch.randn(input_ids.shape[0], 768, device=input_ids.device))
            loss = x.mean()
            return {'loss': loss, 'logits': x}

    model = DummyModel()
    config = TrainingConfig(
        output_dir="test_outputs",
        num_epochs=1,
        batch_size_per_gpu=4,
        learning_rate=1e-4
    )

    print(f"Config: {config.to_dict()}")
    print("Trainer setup complete!")

