"""
Stage 1: Visual-Language Alignment Training

Aligns RepViT vision features with TinyLLM text space using
contrastive learning and image captioning.
"""

import os
import math
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from embervlm.models import EmberVLM
from embervlm.training.train_utils import (
    TrainingConfig,
    setup_distributed,
    cleanup_distributed,
    is_main_process,
    get_rank,
    get_world_size,
    barrier,
    set_seed,
    get_optimizer,
    get_scheduler,
    get_grad_scaler,
    get_autocast_context,
    wrap_model_ddp,
    save_checkpoint,
    load_checkpoint,
    MetricTracker,
    print_trainable_parameters,
)
from embervlm.data.loaders import get_alignment_dataloader
from embervlm.monitoring.wandb_logger import WandbLogger
from embervlm.monitoring.carbon_tracker import CarbonTracker

logger = logging.getLogger(__name__)


class ContrastiveLoss(nn.Module):
    """Image-text contrastive loss (CLIP-style)."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1 / temperature)))

    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute contrastive loss.

        Args:
            image_features: [B, D] normalized image features
            text_features: [B, D] normalized text features

        Returns:
            Loss and metrics dictionary
        """
        # Normalize features
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        # Compute logits
        logit_scale = self.logit_scale.exp().clamp(max=100)
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logits_per_image.T

        # Labels (diagonal is positive)
        batch_size = image_features.size(0)
        labels = torch.arange(batch_size, device=image_features.device)

        # Cross entropy loss
        loss_i2t = F.cross_entropy(logits_per_image, labels)
        loss_t2i = F.cross_entropy(logits_per_text, labels)
        loss = (loss_i2t + loss_t2i) / 2

        # Compute accuracy
        with torch.no_grad():
            pred_i2t = logits_per_image.argmax(dim=-1)
            pred_t2i = logits_per_text.argmax(dim=-1)
            acc_i2t = (pred_i2t == labels).float().mean()
            acc_t2i = (pred_t2i == labels).float().mean()

        metrics = {
            'contrastive_loss': loss.item(),
            'loss_i2t': loss_i2t.item(),
            'loss_t2i': loss_t2i.item(),
            'acc_i2t': acc_i2t.item(),
            'acc_t2i': acc_t2i.item(),
            'logit_scale': logit_scale.item(),
        }

        return loss, metrics


class Stage1Trainer:
    """Trainer for Stage 1: Visual-Language Alignment."""

    def __init__(
        self,
        model: EmberVLM,
        config: TrainingConfig,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        tokenizer: Any = None,
    ):
        self.config = config
        self.tokenizer = tokenizer

        # Setup distributed
        self.rank, self.local_rank, self.world_size = setup_distributed()
        self.device = torch.device(f'cuda:{self.local_rank}')

        # Set seed
        set_seed(config.seed, self.rank)

        # Ensure model is on correct device before DDP
        model = model.to(self.device)

        # Synchronize to ensure all ranks have model loaded
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        # Model
        self.model = wrap_model_ddp(model, config, self.device)

        # Data
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        # Optimizer and scheduler
        self.optimizer = get_optimizer(self.model, config)
        self.scheduler = get_scheduler(self.optimizer, config)
        self.scaler = get_grad_scaler(config)

        # Losses
        self.contrastive_loss = ContrastiveLoss(temperature=0.07).to(self.device)

        # Logging
        self.wandb_logger = None
        self.carbon_tracker = None

        if is_main_process():
            self.wandb_logger = WandbLogger(
                project="embervlm",
                name="stage1_alignment",
                config=config.to_dict(),
            )
            self.carbon_tracker = CarbonTracker(output_dir=config.output_dir)

        # Metrics
        self.metric_tracker = MetricTracker()
        self.global_step = 0

        # Print info
        if is_main_process():
            print_trainable_parameters(self.model)

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
            # Move to device
            pixel_values = batch['pixel_values'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch.get('labels')
            if labels is not None:
                labels = labels.to(self.device)

            # Forward pass
            with get_autocast_context(self.config):
                # Get visual features
                vision_output = self.model.module.encode_image(pixel_values) \
                    if hasattr(self.model, 'module') else self.model.encode_image(pixel_values)
                visual_tokens = vision_output['visual_tokens']

                # Fuse to language space
                fused_visual = self.model.module.fuse_features(visual_tokens) \
                    if hasattr(self.model, 'module') else self.model.fuse_features(visual_tokens)

                # Get text embeddings
                model_ref = self.model.module if hasattr(self.model, 'module') else self.model
                text_embeds = model_ref.language_model.embed_tokens(input_ids)

                # Pool features for contrastive loss
                image_pooled = fused_visual.mean(dim=1)
                text_pooled = (text_embeds * attention_mask.unsqueeze(-1)).sum(dim=1) / \
                              attention_mask.sum(dim=1, keepdim=True).clamp(min=1)

                # Contrastive loss
                contrastive_loss, contrastive_metrics = self.contrastive_loss(
                    image_pooled, text_pooled
                )

                # Captioning loss (language modeling)
                inputs_embeds, _ = model_ref.prepare_inputs_embeds(
                    input_ids, pixel_values
                )

                # Adjust labels to match inputs_embeds length (account for visual tokens)
                # Note: The language model will internally shift labels for causal LM
                if labels.size(1) != inputs_embeds.size(1):
                    batch_size = labels.size(0)
                    num_visual = inputs_embeds.size(1) - labels.size(1)

                    # Create adjusted labels with -100 at visual token positions
                    # Visual tokens are inserted at the beginning (position 0)
                    adjusted_labels = torch.full(
                        (batch_size, inputs_embeds.size(1)),
                        -100,
                        dtype=labels.dtype,
                        device=labels.device
                    )

                    # Visual tokens get -100 (positions 0 to num_visual-1)
                    # Copy original labels after visual tokens
                    adjusted_labels[:, num_visual:] = labels
                else:
                    adjusted_labels = labels

                lm_outputs = model_ref.language_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=torch.ones(
                        inputs_embeds.size(0), inputs_embeds.size(1),
                        device=self.device, dtype=torch.long
                    ),
                    labels=adjusted_labels,
                )

                captioning_loss = lm_outputs.get('loss', torch.tensor(0.0, device=self.device))

                # Total loss
                loss = 0.5 * contrastive_loss + 0.5 * captioning_loss

            # Backward pass
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

                # Update metrics
                metrics = {
                    'loss': loss.item() * self.config.gradient_accumulation_steps,
                    'contrastive_loss': contrastive_metrics['contrastive_loss'],
                    'captioning_loss': captioning_loss.item() if isinstance(captioning_loss, torch.Tensor) else 0.0,
                    'lr': self.scheduler.get_last_lr()[0],
                    **contrastive_metrics,
                }
                self.metric_tracker.update(metrics)

                # Logging
                if self.global_step % self.config.log_steps == 0:
                    avg_metrics = self.metric_tracker.get_average()

                    if is_main_process():
                        self.wandb_logger.log(avg_metrics, step=self.global_step)
                        progress_bar.set_postfix({
                            'loss': f"{avg_metrics['loss']:.4f}",
                            'acc': f"{avg_metrics.get('acc_i2t', 0):.3f}",
                        })

                    self.metric_tracker.reset()

                # Checkpointing
                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint()

                # Evaluation
                if self.val_dataloader is not None and \
                   self.global_step % self.config.eval_steps == 0:
                    self.evaluate(eval_step=self.global_step)

    @torch.no_grad()
    def evaluate(self, eval_step: Optional[int] = None):
        """Evaluate on validation set."""
        if eval_step is None:
            eval_step = self.global_step

        self.model.eval()

        eval_metrics = MetricTracker()

        for batch in tqdm(
            self.val_dataloader,
            desc="Evaluating",
            disable=not is_main_process(),
        ):
            pixel_values = batch['pixel_values'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)

            with get_autocast_context(self.config):
                # Get features
                model_ref = self.model.module if hasattr(self.model, 'module') else self.model
                vision_output = model_ref.encode_image(pixel_values)
                visual_tokens = vision_output['visual_tokens']
                fused_visual = model_ref.fuse_features(visual_tokens)
                text_embeds = model_ref.language_model.embed_tokens(input_ids)

                # Pool
                image_pooled = fused_visual.mean(dim=1)
                text_pooled = (text_embeds * attention_mask.unsqueeze(-1)).sum(dim=1) / \
                              attention_mask.sum(dim=1, keepdim=True).clamp(min=1)

                # Contrastive metrics
                _, metrics = self.contrastive_loss(image_pooled, text_pooled)
                eval_metrics.update(metrics)

        avg_metrics = eval_metrics.get_average()
        avg_metrics = {f'val_{k}': v for k, v in avg_metrics.items()}

        if is_main_process():
            self.wandb_logger.log(avg_metrics, step=eval_step)
            logger.info(f"Validation metrics: {avg_metrics}")

        self.model.train()
        return avg_metrics

    def save_checkpoint(self):
        """Save training checkpoint."""
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

                # Validation at end of epoch
                if self.val_dataloader is not None:
                    self.evaluate()

                # Save at end of epoch
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


def run_stage1_training(
    model: EmberVLM,
    config: TrainingConfig,
    data_dir: str,
    tokenizer: Any,
    num_epochs: int = 3,
):
    """
    Run Stage 1 training.

    Args:
        model: EmberVLM model
        config: Training configuration
        data_dir: Directory with training data
        tokenizer: Tokenizer
        num_epochs: Number of training epochs
    """
    # Create data loaders
    train_dataloader = get_alignment_dataloader(
        data_dir=data_dir,
        tokenizer=tokenizer,
        batch_size=config.batch_size,
        split='train',
        distributed=config.distributed,
    )

    val_dataloader = get_alignment_dataloader(
        data_dir=data_dir,
        tokenizer=tokenizer,
        batch_size=config.batch_size,
        split='val',
        distributed=False,
    )

    # Create trainer
    trainer = Stage1Trainer(
        model=model,
        config=config,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        tokenizer=tokenizer,
    )

    # Train
    trainer.train(num_epochs)


if __name__ == "__main__":
    import argparse
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./outputs/stage1')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=2e-4)
    args = parser.parse_args()

    # Config
    config = TrainingConfig(
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Model
    model = EmberVLM()

    # Train
    run_stage1_training(
        model=model,
        config=config,
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        num_epochs=args.epochs,
    )

