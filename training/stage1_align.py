"""
EmberVLM Stage 1: Vision-Language Alignment
Connects RepViT vision features to TinyLLM text space.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Optional, Any, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

from ..models import EmberVLM, EmberVLMConfig
from .trainer import DistributedTrainer, TrainingConfig, setup_training

logger = logging.getLogger(__name__)


@dataclass
class Stage1Config(TrainingConfig):
    """Configuration for Stage 1: Vision-Language Alignment."""

    # Stage-specific settings
    num_epochs: int = 1
    samples: int = 500_000
    batch_size_per_gpu: int = 128
    gradient_accumulation_steps: int = 4
    learning_rate: float = 3e-4

    # Loss weights
    contrastive_weight: float = 0.5
    captioning_weight: float = 0.5

    # Contrastive learning settings
    temperature: float = 0.07

    def __post_init__(self):
        self.experiment_name = "stage1_alignment"


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for image-text alignment.
    Uses InfoNCE loss similar to CLIP.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / temperature)))

    def forward(
        self,
        image_features: torch.Tensor,  # [B, D]
        text_features: torch.Tensor,   # [B, D]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute contrastive loss between image and text features.

        Args:
            image_features: Image embeddings [B, D]
            text_features: Text embeddings [B, D]

        Returns:
            loss: Scalar loss
            metrics: Dict with accuracy metrics
        """
        # Normalize features
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        # Compute similarity
        logit_scale = self.logit_scale.exp().clamp(max=100)
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logits_per_image.T

        # Labels are just indices (diagonal should match)
        batch_size = image_features.shape[0]
        labels = torch.arange(batch_size, device=image_features.device)

        # Compute cross-entropy loss both ways
        loss_i2t = F.cross_entropy(logits_per_image, labels)
        loss_t2i = F.cross_entropy(logits_per_text, labels)
        loss = (loss_i2t + loss_t2i) / 2

        # Compute accuracy metrics
        with torch.no_grad():
            i2t_acc = (logits_per_image.argmax(dim=-1) == labels).float().mean()
            t2i_acc = (logits_per_text.argmax(dim=-1) == labels).float().mean()

        metrics = {
            'i2t_acc': i2t_acc.item(),
            't2i_acc': t2i_acc.item(),
            'logit_scale': logit_scale.item()
        }

        return loss, metrics


class Stage1AlignmentTrainer(DistributedTrainer):
    """
    Trainer for Stage 1: Vision-Language Alignment.

    Objectives:
    - Align vision features with text embeddings
    - Learn to project RepViT features to TinyLLM space
    - Basic captioning capability
    """

    def __init__(
        self,
        model: EmberVLM,
        config: Stage1Config,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        **kwargs
    ):
        super().__init__(
            model=model,
            config=config,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            **kwargs
        )

        self.stage_config = config

        # Contrastive loss
        self.contrastive_loss = ContrastiveLoss(config.temperature).to(self.device)

        # Feature projectors for contrastive learning
        hidden_size = model.config.hidden_size
        self.image_projector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 256)
        ).to(self.device)

        self.text_projector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 256)
        ).to(self.device)

        # Add projectors to optimizer
        self._add_projectors_to_optimizer()

        # Metrics tracking
        self.clip_scores = []

    def _add_projectors_to_optimizer(self):
        """Add projection layers to optimizer."""
        proj_params = list(self.image_projector.parameters()) + \
                      list(self.text_projector.parameters()) + \
                      list(self.contrastive_loss.parameters())

        self.optimizer.add_param_group({
            'params': proj_params,
            'lr': self.config.learning_rate
        })

    def compute_loss(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss for Stage 1.

        Args:
            batch: Batch containing pixel_values, input_ids, attention_mask, labels

        Returns:
            loss: Combined loss
            metrics: Loss components and metrics
        """
        metrics = {}
        total_loss = torch.tensor(0.0, device=self.device)

        # Get model outputs
        outputs = self.model(
            pixel_values=batch.get('pixel_values'),
            input_ids=batch.get('input_ids'),
            attention_mask=batch.get('attention_mask'),
            labels=batch.get('labels'),
            output_hidden_states=True
        )

        # 1. Captioning loss (language modeling)
        if 'loss' in outputs:
            captioning_loss = outputs['loss']
            total_loss = total_loss + self.stage_config.captioning_weight * captioning_loss
            metrics['captioning_loss'] = captioning_loss.item()

        # 2. Contrastive loss
        if batch.get('pixel_values') is not None:
            # Get vision and text features
            vision_features = outputs.get('vision_features')  # [B, n_tokens, D]

            if vision_features is not None:
                # Pool vision features
                image_features = vision_features.mean(dim=1)  # [B, D]
                image_features = self.image_projector(image_features)

                # Get text features from last hidden state
                hidden_states = outputs.get('hidden_states', [])
                if hidden_states:
                    text_hidden = hidden_states[-1]  # Last layer
                    # Use mean pooling or [CLS] token
                    text_features = text_hidden.mean(dim=1)  # [B, D]
                    text_features = self.text_projector(text_features)

                    # Compute contrastive loss
                    contrastive_loss, contrast_metrics = self.contrastive_loss(
                        image_features, text_features
                    )

                    total_loss = total_loss + self.stage_config.contrastive_weight * contrastive_loss
                    metrics['contrastive_loss'] = contrastive_loss.item()
                    metrics.update(contrast_metrics)

        metrics['total_loss'] = total_loss.item()

        return total_loss, metrics

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        self.model.train()

        # Forward pass with mixed precision
        with autocast(device_type='cuda', dtype=self.autocast_dtype):
            loss, metrics = self.compute_loss(batch)
            loss = loss / self.config.gradient_accumulation_steps

        # Backward pass
        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return metrics

    def evaluate(self) -> Dict[str, float]:
        """Evaluation with CLIP score and perplexity."""
        if not self.eval_dataloader:
            return {}

        self.model.eval()
        total_metrics = {}
        num_batches = 0

        with torch.no_grad():
            for batch in self.eval_dataloader:
                batch = self._prepare_batch(batch)

                with autocast(device_type='cuda', dtype=self.autocast_dtype):
                    _, metrics = self.compute_loss(batch)

                for key, value in metrics.items():
                    total_metrics[key] = total_metrics.get(key, 0) + value
                num_batches += 1

        # Average metrics
        avg_metrics = {f'eval_{k}': v / num_batches for k, v in total_metrics.items()}

        # Compute perplexity from captioning loss
        if 'eval_captioning_loss' in avg_metrics:
            avg_metrics['eval_perplexity'] = torch.exp(
                torch.tensor(avg_metrics['eval_captioning_loss'])
            ).item()

        return avg_metrics

    def save_checkpoint(self, final: bool = False):
        """Save checkpoint with projectors."""
        super()._save_checkpoint(final)

        if self.is_main_process:
            checkpoint_name = "final" if final else f"checkpoint-{self.global_step}"
            checkpoint_dir = self.output_dir / checkpoint_name

            # Save projectors
            torch.save({
                'image_projector': self.image_projector.state_dict(),
                'text_projector': self.text_projector.state_dict(),
                'contrastive_loss': self.contrastive_loss.state_dict()
            }, checkpoint_dir / "projectors.pt")


def run_stage1_alignment(
    model: EmberVLM,
    train_dataset: Any,
    eval_dataset: Optional[Any] = None,
    config: Optional[Stage1Config] = None,
    **kwargs
) -> Dict[str, float]:
    """
    Run Stage 1: Vision-Language Alignment training.

    Args:
        model: EmberVLM model
        train_dataset: Training dataset with image-text pairs
        eval_dataset: Optional evaluation dataset
        config: Stage 1 configuration

    Returns:
        Training metrics
    """
    if config is None:
        config = Stage1Config(**kwargs)

    logger.info("=" * 50)
    logger.info("Stage 1: Vision-Language Alignment")
    logger.info("=" * 50)
    logger.info(f"Training samples: {config.samples:,}")
    logger.info(f"Batch size: {config.batch_size_per_gpu} × {config.gradient_accumulation_steps} accum")
    logger.info(f"Learning rate: {config.learning_rate}")
    logger.info(f"Loss weights: contrastive={config.contrastive_weight}, captioning={config.captioning_weight}")

    # Setup trainer
    trainer = setup_training(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        config=config
    )

    # Convert to Stage1 trainer
    stage1_trainer = Stage1AlignmentTrainer(
        model=model,
        config=config,
        train_dataloader=trainer.train_dataloader,
        eval_dataloader=trainer.eval_dataloader
    )

    # Train
    results = stage1_trainer.train()

    logger.info("Stage 1 complete!")
    logger.info(f"Final results: {results}")

    return results


if __name__ == "__main__":
    # Test Stage 1 training
    print("Testing Stage 1: Vision-Language Alignment...")

    config = Stage1Config(
        output_dir="test_outputs",
        num_epochs=1,
        batch_size_per_gpu=4,
        gradient_accumulation_steps=1,
        learning_rate=1e-4
    )

    print(f"Stage 1 Config: {config}")

    # Test contrastive loss
    contrastive = ContrastiveLoss(temperature=0.07)
    image_feat = torch.randn(8, 256)
    text_feat = torch.randn(8, 256)
    loss, metrics = contrastive(image_feat, text_feat)
    print(f"Contrastive loss: {loss.item():.4f}")
    print(f"Metrics: {metrics}")

    print("Stage 1 tests complete!")

