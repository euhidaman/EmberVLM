"""
EmberVLM Stage 2: Multimodal Instruction Tuning
Instruction following with knowledge distillation from larger VLM.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Optional, Any, Tuple, List
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

from ..models import EmberVLM, EmberVLMConfig
from .trainer import DistributedTrainer, TrainingConfig

logger = logging.getLogger(__name__)


@dataclass
class Stage2Config(TrainingConfig):
    """Configuration for Stage 2: Multimodal Instruction Tuning."""

    # Stage-specific settings
    num_epochs: int = 2
    samples: int = 300_000
    batch_size_per_gpu: int = 64
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.1

    # Knowledge Distillation settings
    distillation_enabled: bool = True
    teacher_model_name: str = "Qwen/Qwen-VL-Chat"
    temperature: float = 2.0

    # Loss weights
    task_loss_weight: float = 0.4
    logit_distillation_weight: float = 0.3
    hidden_distillation_weight: float = 0.2
    attention_distillation_weight: float = 0.1

    def __post_init__(self):
        self.experiment_name = "stage2_instruction"


class DistillationLoss(nn.Module):
    """
    Knowledge distillation losses for multimodal instruction tuning.
    """

    def __init__(self, temperature: float = 2.0):
        super().__init__()
        self.temperature = temperature

    def logit_distillation(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        KL-divergence loss on output logits.

        Args:
            student_logits: [B, L, V] student model logits
            teacher_logits: [B, L, V] teacher model logits
            mask: Optional [B, L] mask for valid positions

        Returns:
            Scalar loss
        """
        # Apply temperature
        student_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)

        # KL divergence
        kl_loss = F.kl_div(student_probs, teacher_probs, reduction='none')
        kl_loss = kl_loss.sum(dim=-1)  # Sum over vocab

        # Apply mask if provided
        if mask is not None:
            kl_loss = kl_loss * mask
            kl_loss = kl_loss.sum() / mask.sum().clamp(min=1)
        else:
            kl_loss = kl_loss.mean()

        # Scale by temperature squared (standard KD practice)
        return kl_loss * (self.temperature ** 2)

    def hidden_state_distillation(
        self,
        student_hidden: torch.Tensor,
        teacher_hidden: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        MSE loss on hidden states (typically [CLS] or mean-pooled).

        Args:
            student_hidden: [B, D_student] student hidden states
            teacher_hidden: [B, D_teacher] teacher hidden states
            mask: Optional mask

        Returns:
            Scalar loss
        """
        # Project if dimensions don't match
        if student_hidden.shape[-1] != teacher_hidden.shape[-1]:
            # This should be handled by the caller with a projection layer
            raise ValueError(
                f"Hidden dimensions don't match: {student_hidden.shape[-1]} vs {teacher_hidden.shape[-1]}"
            )

        # Normalize for stable distillation
        student_hidden = F.normalize(student_hidden, dim=-1)
        teacher_hidden = F.normalize(teacher_hidden, dim=-1)

        # MSE loss
        loss = F.mse_loss(student_hidden, teacher_hidden, reduction='mean')

        return loss

    def attention_distillation(
        self,
        student_attention: torch.Tensor,
        teacher_attention: torch.Tensor
    ) -> torch.Tensor:
        """
        Cosine similarity loss on attention patterns.

        Args:
            student_attention: [B, H, L, L] student attention weights
            teacher_attention: [B, H, L, L] teacher attention weights

        Returns:
            Scalar loss
        """
        # Average over heads if needed
        if student_attention.shape[1] != teacher_attention.shape[1]:
            # Average teacher heads to match student
            n_student_heads = student_attention.shape[1]
            n_teacher_heads = teacher_attention.shape[1]
            ratio = n_teacher_heads // n_student_heads

            teacher_attention = teacher_attention.view(
                teacher_attention.shape[0],
                n_student_heads,
                ratio,
                *teacher_attention.shape[2:]
            ).mean(dim=2)

        # Flatten attention for cosine similarity
        B, H, L, _ = student_attention.shape
        student_flat = student_attention.view(B * H, -1)
        teacher_flat = teacher_attention.view(B * H, -1)

        # Cosine similarity (want to maximize, so loss = 1 - similarity)
        similarity = F.cosine_similarity(student_flat, teacher_flat, dim=-1)
        loss = 1 - similarity.mean()

        return loss


class TeacherWrapper:
    """
    Wrapper for teacher model (Qwen-VL or similar).
    Handles loading and inference without gradients.
    """

    def __init__(self, model_name: str, device: torch.device):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.processor = None
        self._loaded = False

    def load(self):
        """Load teacher model (lazy loading to save memory)."""
        if self._loaded:
            return

        try:
            from transformers import AutoModelForCausalLM, AutoProcessor

            logger.info(f"Loading teacher model: {self.model_name}")

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            self.model.eval()

            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )

            self._loaded = True
            logger.info("Teacher model loaded successfully")

        except Exception as e:
            logger.warning(f"Could not load teacher model: {e}")
            logger.warning("Distillation will be disabled")

    def unload(self):
        """Unload teacher model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        self._loaded = False
        torch.cuda.empty_cache()

    @torch.no_grad()
    def get_teacher_outputs(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Get teacher model outputs for distillation.

        Returns:
            Dict with logits, hidden_states, attentions (if available)
        """
        if not self._loaded or self.model is None:
            return None

        try:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                output_attentions=True,
                return_dict=True
            )

            return {
                'logits': outputs.logits,
                'hidden_states': outputs.hidden_states[-1] if outputs.hidden_states else None,
                'attentions': outputs.attentions[-1] if outputs.attentions else None
            }

        except Exception as e:
            logger.warning(f"Teacher forward pass failed: {e}")
            return None


class Stage2InstructionTrainer(DistributedTrainer):
    """
    Trainer for Stage 2: Multimodal Instruction Tuning with Knowledge Distillation.
    """

    def __init__(
        self,
        model: EmberVLM,
        config: Stage2Config,
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

        # Distillation components
        self.distillation_loss = DistillationLoss(config.temperature)

        # Teacher model (lazy loaded)
        self.teacher = None
        if config.distillation_enabled:
            self.teacher = TeacherWrapper(config.teacher_model_name, self.device)

        # Hidden state projection (if needed for distillation)
        # Teacher: 7B model ~= 4096 hidden, Student: 768 hidden
        self.hidden_projector = nn.Linear(768, 768).to(self.device)  # Identity for now

        # Metrics tracking
        self.distillation_metrics = []

    def _load_teacher_if_needed(self):
        """Load teacher model if not already loaded."""
        if self.teacher and not self.teacher._loaded:
            self.teacher.load()

    def compute_loss(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss with knowledge distillation.
        """
        metrics = {}
        total_loss = torch.tensor(0.0, device=self.device)

        # Student forward pass
        student_outputs = self.model(
            pixel_values=batch.get('pixel_values'),
            input_ids=batch.get('input_ids'),
            attention_mask=batch.get('attention_mask'),
            labels=batch.get('labels'),
            output_hidden_states=True,
            output_attentions=True
        )

        # 1. Task loss (language modeling)
        if 'loss' in student_outputs:
            task_loss = student_outputs['loss']
            total_loss = total_loss + self.stage_config.task_loss_weight * task_loss
            metrics['task_loss'] = task_loss.item()

        # 2. Knowledge distillation losses
        if self.teacher and self.stage_config.distillation_enabled:
            self._load_teacher_if_needed()

            teacher_outputs = self.teacher.get_teacher_outputs(
                pixel_values=batch.get('pixel_values'),
                input_ids=batch.get('input_ids'),
                attention_mask=batch.get('attention_mask')
            )

            if teacher_outputs is not None:
                # 2a. Logit distillation
                if teacher_outputs.get('logits') is not None:
                    # Truncate to match student vocab size if needed
                    teacher_logits = teacher_outputs['logits']
                    student_logits = student_outputs['logits']

                    min_vocab = min(teacher_logits.shape[-1], student_logits.shape[-1])
                    teacher_logits = teacher_logits[..., :min_vocab]
                    student_logits = student_logits[..., :min_vocab]

                    logit_loss = self.distillation_loss.logit_distillation(
                        student_logits=student_logits,
                        teacher_logits=teacher_logits,
                        mask=batch.get('attention_mask')
                    )
                    total_loss = total_loss + self.stage_config.logit_distillation_weight * logit_loss
                    metrics['logit_distill_loss'] = logit_loss.item()

                # 2b. Hidden state distillation
                if teacher_outputs.get('hidden_states') is not None:
                    student_hidden = student_outputs.get('hidden_states', [])
                    if student_hidden:
                        # Use [CLS] token or mean pooling
                        student_h = student_hidden[-1].mean(dim=1)  # [B, D]
                        teacher_h = teacher_outputs['hidden_states'].mean(dim=1)

                        # Project teacher to student dimension
                        if student_h.shape[-1] != teacher_h.shape[-1]:
                            # Use a simple learned projection
                            teacher_h = F.adaptive_avg_pool1d(
                                teacher_h.unsqueeze(1),
                                student_h.shape[-1]
                            ).squeeze(1)

                        hidden_loss = self.distillation_loss.hidden_state_distillation(
                            student_hidden=student_h,
                            teacher_hidden=teacher_h
                        )
                        total_loss = total_loss + self.stage_config.hidden_distillation_weight * hidden_loss
                        metrics['hidden_distill_loss'] = hidden_loss.item()

                # 2c. Attention distillation
                if teacher_outputs.get('attentions') is not None:
                    student_attn = student_outputs.get('attentions', [])
                    if student_attn:
                        # Use last layer attention
                        student_a = student_attn[-1]
                        teacher_a = teacher_outputs['attentions']

                        # Resize if needed
                        if student_a.shape[-1] != teacher_a.shape[-1]:
                            # Interpolate teacher attention to student size
                            teacher_a = F.interpolate(
                                teacher_a,
                                size=student_a.shape[-2:],
                                mode='bilinear',
                                align_corners=False
                            )

                        attn_loss = self.distillation_loss.attention_distillation(
                            student_attention=student_a,
                            teacher_attention=teacher_a
                        )
                        total_loss = total_loss + self.stage_config.attention_distillation_weight * attn_loss
                        metrics['attn_distill_loss'] = attn_loss.item()

        metrics['total_loss'] = total_loss.item()

        return total_loss, metrics

    def evaluate(self) -> Dict[str, float]:
        """Evaluation for instruction following."""
        if not self.eval_dataloader:
            return {}

        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for batch in self.eval_dataloader:
                batch = self._prepare_batch(batch)

                with autocast(device_type='cuda', dtype=self.autocast_dtype):
                    outputs = self.model(
                        pixel_values=batch.get('pixel_values'),
                        input_ids=batch.get('input_ids'),
                        attention_mask=batch.get('attention_mask'),
                        labels=batch.get('labels')
                    )

                if 'loss' in outputs:
                    total_loss += outputs['loss'].item()

                # Compute token accuracy
                logits = outputs['logits']
                labels = batch.get('labels')
                if labels is not None:
                    predictions = logits.argmax(dim=-1)
                    mask = labels != -100
                    correct_predictions += ((predictions == labels) & mask).sum().item()
                    total_predictions += mask.sum().item()

        num_batches = len(self.eval_dataloader)
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

        return {
            'eval_loss': avg_loss,
            'eval_accuracy': accuracy,
            'eval_perplexity': torch.exp(torch.tensor(avg_loss)).item()
        }

    def cleanup(self):
        """Cleanup resources."""
        if self.teacher:
            self.teacher.unload()


def run_stage2_instruction(
    model: EmberVLM,
    train_dataset: Any,
    eval_dataset: Optional[Any] = None,
    config: Optional[Stage2Config] = None,
    **kwargs
) -> Dict[str, float]:
    """
    Run Stage 2: Multimodal Instruction Tuning.

    Args:
        model: EmberVLM model
        train_dataset: Training dataset with instruction-response pairs
        eval_dataset: Optional evaluation dataset
        config: Stage 2 configuration

    Returns:
        Training metrics
    """
    if config is None:
        config = Stage2Config(**kwargs)

    logger.info("=" * 50)
    logger.info("Stage 2: Multimodal Instruction Tuning")
    logger.info("=" * 50)
    logger.info(f"Training samples: {config.samples:,}")
    logger.info(f"Distillation enabled: {config.distillation_enabled}")
    if config.distillation_enabled:
        logger.info(f"Teacher model: {config.teacher_model_name}")
        logger.info(f"Temperature: {config.temperature}")
    logger.info(f"Loss weights:")
    logger.info(f"  Task: {config.task_loss_weight}")
    logger.info(f"  Logit distillation: {config.logit_distillation_weight}")
    logger.info(f"  Hidden distillation: {config.hidden_distillation_weight}")
    logger.info(f"  Attention distillation: {config.attention_distillation_weight}")

    # Create dataloaders
    from torch.utils.data import DataLoader

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size_per_gpu,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=config.batch_size_per_gpu,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    ) if eval_dataset else None

    # Create trainer
    trainer = Stage2InstructionTrainer(
        model=model,
        config=config,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader
    )

    try:
        # Train
        results = trainer.train()

        logger.info("Stage 2 complete!")
        logger.info(f"Final results: {results}")

        return results

    finally:
        trainer.cleanup()


if __name__ == "__main__":
    # Test Stage 2 components
    print("Testing Stage 2: Multimodal Instruction Tuning...")

    config = Stage2Config(
        output_dir="test_outputs",
        num_epochs=1,
        batch_size_per_gpu=4,
        distillation_enabled=False  # Disable for testing
    )

    print(f"Stage 2 Config: {config}")

    # Test distillation loss
    distill_loss = DistillationLoss(temperature=2.0)

    student_logits = torch.randn(4, 32, 1000)
    teacher_logits = torch.randn(4, 32, 1000)

    kd_loss = distill_loss.logit_distillation(student_logits, teacher_logits)
    print(f"KD logit loss: {kd_loss.item():.4f}")

    student_hidden = torch.randn(4, 768)
    teacher_hidden = torch.randn(4, 768)

    hidden_loss = distill_loss.hidden_state_distillation(student_hidden, teacher_hidden)
    print(f"Hidden state loss: {hidden_loss.item():.4f}")

    student_attn = torch.randn(4, 12, 32, 32).softmax(dim=-1)
    teacher_attn = torch.randn(4, 12, 32, 32).softmax(dim=-1)

    attn_loss = distill_loss.attention_distillation(student_attn, teacher_attn)
    print(f"Attention loss: {attn_loss.item():.4f}")

    print("Stage 2 tests complete!")

