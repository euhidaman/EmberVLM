"""
Stage 2: Multimodal Instruction Tuning

Teaches task-following capabilities with teacher distillation
from larger VLM models.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from embervlm.models import EmberVLM
from embervlm.training.train_utils import (
    TrainingConfig,
    setup_distributed,
    cleanup_distributed,
    is_main_process,
    get_rank,
    barrier,
    set_seed,
    get_optimizer,
    get_scheduler,
    get_grad_scaler,
    get_autocast_context,
    wrap_model_ddp,
    save_checkpoint,
    MetricTracker,
    print_trainable_parameters,
)
from embervlm.data.loaders import get_instruction_dataloader
from embervlm.monitoring.wandb_logger import WandbLogger
from embervlm.monitoring.carbon_tracker import CarbonTracker

logger = logging.getLogger(__name__)


class DistillationLoss(nn.Module):
    """Knowledge distillation loss for teacher-student training."""

    def __init__(self, temperature: float = 2.0, alpha: float = 0.3):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute distillation loss.

        Args:
            student_logits: Student model logits [B, seq_len, vocab]
            teacher_logits: Teacher model logits [B, seq_len, vocab]
            labels: Target labels [B, seq_len]

        Returns:
            Dictionary with loss components
        """
        # Soft loss (distillation)
        student_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft_loss = self.kl_loss(student_soft, teacher_soft) * (self.temperature ** 2)

        # Hard loss (cross-entropy with labels)
        hard_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )

        # Combined loss
        total_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss

        return {
            'loss': total_loss,
            'soft_loss': soft_loss,
            'hard_loss': hard_loss,
        }


class HiddenStateAlignmentLoss(nn.Module):
    """Loss for aligning hidden states with teacher."""

    def __init__(self, student_dim: int, teacher_dim: int):
        super().__init__()
        if student_dim != teacher_dim:
            self.proj = nn.Linear(student_dim, teacher_dim)
        else:
            self.proj = nn.Identity()

    def forward(
        self,
        student_hidden: torch.Tensor,
        teacher_hidden: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute hidden state alignment loss.

        Args:
            student_hidden: Student hidden states [B, seq_len, student_dim]
            teacher_hidden: Teacher hidden states [B, seq_len, teacher_dim]
        """
        student_proj = self.proj(student_hidden)

        # L2 loss on normalized features
        student_norm = F.normalize(student_proj, dim=-1)
        teacher_norm = F.normalize(teacher_hidden, dim=-1)

        return F.mse_loss(student_norm, teacher_norm)


class Stage2Trainer:
    """Trainer for Stage 2: Multimodal Instruction Tuning."""

    def __init__(
        self,
        model: EmberVLM,
        config: TrainingConfig,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        tokenizer: Any = None,
        teacher_model: Optional[nn.Module] = None,
        distillation_config: Optional[Dict[str, Any]] = None,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.teacher_model = teacher_model

        # Setup distributed
        self.rank, self.local_rank, self.world_size = setup_distributed()
        self.device = torch.device(f'cuda:{self.local_rank}')

        # Set seed
        set_seed(config.seed, self.rank)

        # Ensure model is on correct device before DDP
        try:
            model = model.to(self.device)
            torch.cuda.synchronize()  # Ensure CUDA operations complete
        except RuntimeError as e:
            logger.error(f"[Rank {self.rank}] Failed to move model to device: {e}")
            raise

        # Synchronize to ensure all ranks have model loaded
        if torch.distributed.is_initialized() and self.world_size > 1:
            try:
                torch.distributed.barrier()
            except Exception as e:
                logger.error(f"[Rank {self.rank}] Barrier failed: {e}")
                raise

        # Model
        self.model = wrap_model_ddp(model, config, self.device)

        # Teacher model (frozen)
        if self.teacher_model is not None:
            self.teacher_model = self.teacher_model.to(self.device)
            self.teacher_model.eval()
            for param in self.teacher_model.parameters():
                param.requires_grad = False

        # Data
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        # Optimizer and scheduler
        self.optimizer = get_optimizer(self.model, config)
        self.scheduler = get_scheduler(self.optimizer, config)
        self.scaler = get_grad_scaler(config)

        # Distillation
        distillation_config = distillation_config or {}
        self.use_distillation = self.teacher_model is not None

        if self.use_distillation:
            self.distillation_loss = DistillationLoss(
                temperature=distillation_config.get('temperature', 2.0),
                alpha=distillation_config.get('alpha', 0.3),
            )

        # Loss weights
        self.sft_weight = distillation_config.get('sft_weight', 0.7)
        self.distill_weight = distillation_config.get('distill_weight', 0.3)

        # Logging
        self.wandb_logger = None
        self.carbon_tracker = None

        if is_main_process():
            from embervlm.monitoring.wandb_logger import EnhancedWandbLogger
            self.wandb_logger = EnhancedWandbLogger(
                project="embervlm",
                name="stage2_instruct",
                config=config.to_dict(),
                output_dir=config.output_dir,
            )
            self.carbon_tracker = CarbonTracker(output_dir=config.output_dir)

        # Metrics
        self.metric_tracker = MetricTracker()
        self.global_step = 0

        if is_main_process():
            print_trainable_parameters(self.model)

    def compute_teacher_outputs(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Get outputs from teacher model."""
        if self.teacher_model is None:
            return {}

        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        return {
            'logits': teacher_outputs.get('logits'),
            'hidden_states': teacher_outputs.get('hidden_states'),
        }

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        # Move to device
        pixel_values = batch['pixel_values'].to(self.device)
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)

        # Forward pass
        with get_autocast_context(self.config):
            # Student forward
            outputs = self.model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                labels=labels,
                output_hidden_states=self.use_distillation,
            )

            # SFT loss
            sft_loss = outputs['loss']

            # Distillation loss
            if self.use_distillation:
                teacher_outputs = self.compute_teacher_outputs(
                    pixel_values, input_ids, attention_mask
                )

                if teacher_outputs.get('logits') is not None:
                    distill_losses = self.distillation_loss(
                        outputs['logits'],
                        teacher_outputs['logits'],
                        labels,
                    )
                    distill_loss = distill_losses['loss']

                    # Combined loss
                    loss = self.sft_weight * sft_loss + self.distill_weight * distill_loss
                else:
                    loss = sft_loss
                    distill_loss = torch.tensor(0.0)
            else:
                loss = sft_loss
                distill_loss = torch.tensor(0.0)

        metrics = {
            'loss': loss.item(),
            'sft_loss': sft_loss.item(),
            'distill_loss': distill_loss.item() if isinstance(distill_loss, torch.Tensor) else distill_loss,
        }

        return loss, metrics

    def train_epoch(self, epoch: int):
        """Train for one epoch."""
        self.model.train()
        self.metric_tracker.reset()

        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {epoch}",
            disable=not is_main_process(),
        )

        for batch_idx, batch in enumerate(progress_bar):
            loss, metrics = self.train_step(batch)

            # Backward
            if self.config.gradient_accumulation_steps > 1:
                loss = loss / self.config.gradient_accumulation_steps

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

                # Add lr to metrics
                metrics['lr'] = self.scheduler.get_last_lr()[0]
                self.metric_tracker.update(metrics)

                # Logging
                if self.global_step % self.config.log_steps == 0:
                    avg_metrics = self.metric_tracker.get_average()

                    if is_main_process():
                        self.wandb_logger.log_with_visualization(
                            avg_metrics,
                            step=self.global_step,
                            stage_name="stage2",
                        )
                        progress_bar.set_postfix({
                            'loss': f"{avg_metrics['loss']:.4f}",
                        })

                        # Log gradient distribution periodically
                        if self.global_step % 500 == 0:
                            gradients = {}
                            for name, param in self.model.named_parameters():
                                if param.grad is not None:
                                    gradients[name] = param.grad
                            if gradients:
                                self.wandb_logger.log_gradient_distribution(
                                    gradients, self.global_step, "stage2"
                                )

                    self.metric_tracker.reset()

                # Save checkpoint
                if self.global_step % self.config.save_steps == 0:
                    barrier()  # Sync before checkpoint
                    self.save_checkpoint()
                    barrier()  # Sync after checkpoint

        # Sync at end of epoch before evaluation
        barrier()

    @torch.no_grad()
    def evaluate(self, eval_step: Optional[int] = None):
        """Evaluate on validation set."""
        if eval_step is None:
            eval_step = self.global_step

        self.model.eval()
        eval_metrics = MetricTracker()

        # Track for visualization (sample 3 random indices)
        num_batches = len(self.val_dataloader)
        vis_indices = set(np.random.choice(num_batches, min(3, num_batches), replace=False)) if is_main_process() else set()

        for batch_idx, batch in enumerate(tqdm(
            self.val_dataloader,
            desc="Evaluating",
            disable=not is_main_process(),
        )):
            pixel_values = batch['pixel_values'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            with get_autocast_context(self.config):
                outputs = self.model(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    attention_mask=attention_mask,
                    labels=labels,
                    return_dict=True,
                )

                loss = outputs['loss']

                # Compute accuracy (token-level)
                logits = outputs['logits']
                predictions = logits.argmax(dim=-1)

                # Align predictions with labels length (handle visual token offset)
                if predictions.size(1) != labels.size(1):
                    min_len = min(predictions.size(1), labels.size(1))
                    # Truncate both to same length to avoid mismatch
                    predictions = predictions[:, :min_len]
                    labels_eval = labels[:, :min_len]
                else:
                    labels_eval = labels

                mask = labels_eval != -100
                correct = ((predictions == labels_eval) & mask).sum()
                total = mask.sum()
                accuracy = correct.float() / total.float() if total > 0 else torch.tensor(0.0)

                # Visualize attention for random samples
                if batch_idx in vis_indices and is_main_process():
                    try:
                        # Get attention weights if available
                        if 'attentions' in outputs or hasattr(outputs, 'attentions'):
                            attentions = outputs.get('attentions') or outputs.attentions
                            if attentions is not None and len(attentions) > 0:
                                # Use last layer attention
                                attn = attentions[-1][0]  # [num_heads, seq_len, seq_len]

                                # Decode tokens for caption
                                text_tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0][:20].tolist())

                                self.wandb_logger.log_attention_visualization(
                                    image=pixel_values[0],
                                    attention_map=attn,
                                    text_tokens=text_tokens,
                                    step=eval_step,
                                    stage_name="stage2",
                                )
                    except Exception as e:
                        logger.debug(f"Failed to visualize attention: {e}")

            eval_metrics.update({
                'loss': loss.item(),
                'accuracy': accuracy.item(),
            })

        avg_metrics = eval_metrics.get_average()
        avg_metrics = {f'val_{k}': v for k, v in avg_metrics.items()}

        if is_main_process():
            self.wandb_logger.log(avg_metrics, step=eval_step)
            logger.info(f"Validation: {avg_metrics}")

        self.model.train()
        return avg_metrics

    def save_checkpoint(self):
        """Save checkpoint."""
        output_dir = Path(self.config.output_dir) / f'checkpoint-{self.global_step}'

        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            config=self.config,
            step=self.global_step,
            metrics=self.metric_tracker.get_average(),
            output_dir=str(output_dir),
            scaler=self.scaler,
        )

        # Clean old checkpoints
        if is_main_process():
            checkpoints = sorted(
                Path(self.config.output_dir).glob('checkpoint-*'),
                key=lambda x: int(x.name.split('-')[1])
            )

            while len(checkpoints) > self.config.max_checkpoints:
                oldest = checkpoints.pop(0)
                import shutil
                shutil.rmtree(oldest)

    def train(self, num_epochs: int):
        """Run full training."""
        if is_main_process() and self.carbon_tracker is not None:
            self.carbon_tracker.start()

        try:
            for epoch in range(num_epochs):
                if hasattr(self.train_dataloader.sampler, 'set_epoch'):
                    self.train_dataloader.sampler.set_epoch(epoch)

                self.train_epoch(epoch)

                if self.val_dataloader is not None:
                    self.evaluate()

                self.save_checkpoint()
                barrier()

        except Exception as e:
            # Log error to WandB before crashing
            error_msg = f"Training failed at epoch {epoch}, step {self.global_step}: {str(e)}"
            logger.error(error_msg)

            if is_main_process() and self.wandb_logger is not None:
                self.wandb_logger.log({
                    'error': error_msg,
                    'error_type': type(e).__name__,
                    'failed_at_step': self.global_step,
                }, step=self.global_step)

            raise  # Re-raise to preserve stack trace

        finally:
            if is_main_process():
                if self.carbon_tracker is not None:
                    emissions = self.carbon_tracker.stop()
                    logger.info(f"Total emissions: {emissions:.4f} kg CO2eq")

                if self.wandb_logger is not None:
                    self.wandb_logger.finish()

            cleanup_distributed()


def run_stage2_training(
    model: EmberVLM,
    config: TrainingConfig,
    data_dir: str,
    tokenizer: Any,
    num_epochs: int = 5,
    teacher_model: Optional[nn.Module] = None,
):
    """Run Stage 2 training."""
    train_dataloader = get_instruction_dataloader(
        data_dir=data_dir,
        tokenizer=tokenizer,
        batch_size=config.batch_size,
        split='train',
        distributed=config.distributed,
    )

    val_dataloader = get_instruction_dataloader(
        data_dir=data_dir,
        tokenizer=tokenizer,
        batch_size=config.batch_size,
        split='val',
        distributed=False,
    )

    trainer = Stage2Trainer(
        model=model,
        config=config,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        tokenizer=tokenizer,
        teacher_model=teacher_model,
        distillation_config={
            'temperature': 2.0,
            'alpha': 0.3,
            'sft_weight': 0.7,
            'distill_weight': 0.3,
        },
    )

    trainer.train(num_epochs)


if __name__ == "__main__":
    import argparse
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./outputs/stage2')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=2e-4)
    args = parser.parse_args()

    config = TrainingConfig(
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    model = EmberVLM()

    if args.checkpoint:
        model = EmberVLM.from_pretrained(args.checkpoint)

    run_stage2_training(
        model=model,
        config=config,
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        num_epochs=args.epochs,
    )

