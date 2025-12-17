"""
EmberVLM Knowledge Distillation Module
Distillation utilities for teacher-student training.
"""

import logging
from typing import Dict, Optional, Any, Tuple, List
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class DistillationConfig:
    """Configuration for knowledge distillation."""
    temperature: float = 2.0
    alpha_ce: float = 0.5       # Weight for cross-entropy loss
    alpha_kd: float = 0.3       # Weight for KD loss
    alpha_hidden: float = 0.15  # Weight for hidden state matching
    alpha_attn: float = 0.05    # Weight for attention matching

    # Teacher settings
    teacher_model_name: str = "Qwen/Qwen-VL-Chat"
    teacher_device: str = "cuda"
    teacher_dtype: str = "float16"

    # Feature matching
    match_layers: List[int] = None  # Which layers to match (-1 = last)
    use_projectors: bool = True      # Project features to match dimensions

    def __post_init__(self):
        if self.match_layers is None:
            self.match_layers = [-1]  # Only last layer by default


class FeatureProjector(nn.Module):
    """Project features between different dimensions."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or max(in_dim, out_dim)

        self.projector = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(x)


class MultiLayerDistillation(nn.Module):
    """
    Multi-layer knowledge distillation.
    Matches intermediate representations across student and teacher.
    """

    def __init__(
        self,
        student_dims: List[int],
        teacher_dims: List[int],
        config: DistillationConfig
    ):
        super().__init__()
        self.config = config

        # Create projectors for each matched layer
        self.projectors = nn.ModuleList()
        for s_dim, t_dim in zip(student_dims, teacher_dims):
            if s_dim != t_dim and config.use_projectors:
                self.projectors.append(FeatureProjector(s_dim, t_dim))
            else:
                self.projectors.append(nn.Identity())

    def forward(
        self,
        student_hidden_states: List[torch.Tensor],
        teacher_hidden_states: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute multi-layer hidden state matching loss.
        """
        total_loss = 0.0
        num_layers = min(len(student_hidden_states), len(teacher_hidden_states))

        for i, (proj, layer_idx) in enumerate(zip(self.projectors, self.config.match_layers)):
            if abs(layer_idx) > num_layers:
                continue

            student_h = student_hidden_states[layer_idx]
            teacher_h = teacher_hidden_states[layer_idx]

            # Project student to teacher dimension
            student_h = proj(student_h)

            # Normalize for stable distillation
            student_h = F.normalize(student_h, dim=-1)
            teacher_h = F.normalize(teacher_h, dim=-1)

            # MSE loss
            loss = F.mse_loss(student_h, teacher_h)
            total_loss += loss

        return total_loss / max(1, len(self.config.match_layers))


class AttentionTransfer(nn.Module):
    """
    Attention transfer for distillation.
    Matches attention patterns between student and teacher.
    """

    def __init__(self, config: DistillationConfig):
        super().__init__()
        self.config = config

    def forward(
        self,
        student_attentions: List[torch.Tensor],
        teacher_attentions: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute attention transfer loss.

        Args:
            student_attentions: List of [B, H_s, L, L] attention weights
            teacher_attentions: List of [B, H_t, L, L] attention weights
        """
        total_loss = 0.0
        num_layers = min(len(student_attentions), len(teacher_attentions))

        for layer_idx in self.config.match_layers:
            if abs(layer_idx) > num_layers:
                continue

            student_attn = student_attentions[layer_idx]  # [B, H_s, L, L]
            teacher_attn = teacher_attentions[layer_idx]  # [B, H_t, L, L]

            # Handle different number of heads
            B, H_s, L_s, _ = student_attn.shape
            _, H_t, L_t, _ = teacher_attn.shape

            # Interpolate if sequence lengths differ
            if L_s != L_t:
                teacher_attn = F.interpolate(
                    teacher_attn.view(B * H_t, 1, L_t, L_t),
                    size=(L_s, L_s),
                    mode='bilinear',
                    align_corners=False
                ).view(B, H_t, L_s, L_s)

            # Average heads if counts differ
            if H_s != H_t:
                # Average teacher heads to match student
                if H_t > H_s:
                    ratio = H_t // H_s
                    teacher_attn = teacher_attn.view(B, H_s, ratio, L_s, L_s).mean(dim=2)
                else:
                    # Replicate student heads to match teacher
                    ratio = H_s // H_t
                    student_attn = student_attn.view(B, H_t, ratio, L_s, L_s).mean(dim=2)

            # KL divergence on attention distributions
            student_attn = student_attn.view(-1, L_s)
            teacher_attn = teacher_attn.view(-1, L_s)

            # Add small epsilon for numerical stability
            student_attn = student_attn + 1e-8
            teacher_attn = teacher_attn + 1e-8

            # Normalize
            student_attn = student_attn / student_attn.sum(dim=-1, keepdim=True)
            teacher_attn = teacher_attn / teacher_attn.sum(dim=-1, keepdim=True)

            loss = F.kl_div(
                student_attn.log(),
                teacher_attn,
                reduction='batchmean'
            )
            total_loss += loss

        return total_loss / max(1, len(self.config.match_layers))


class DistillationLossModule(nn.Module):
    """
    Complete distillation loss module combining all components.
    """

    def __init__(
        self,
        config: DistillationConfig,
        student_hidden_dim: int = 768,
        teacher_hidden_dim: int = 4096,
        student_vocab_size: int = 50257,
        teacher_vocab_size: int = 151936
    ):
        super().__init__()
        self.config = config
        self.temperature = config.temperature

        # Vocabulary projection (teacher -> student vocab)
        self.vocab_projector = nn.Linear(
            teacher_vocab_size,
            student_vocab_size,
            bias=False
        ) if teacher_vocab_size != student_vocab_size else nn.Identity()

        # Hidden state projector (student -> teacher dim for comparison)
        self.hidden_projector = FeatureProjector(
            student_hidden_dim,
            teacher_hidden_dim
        ) if student_hidden_dim != teacher_hidden_dim else nn.Identity()

        # Multi-layer distillation
        self.layer_distillation = MultiLayerDistillation(
            student_dims=[student_hidden_dim] * len(config.match_layers),
            teacher_dims=[teacher_hidden_dim] * len(config.match_layers),
            config=config
        )

        # Attention transfer
        self.attention_transfer = AttentionTransfer(config)

    def compute_kd_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute KL-divergence distillation loss on logits."""
        # Project teacher vocab to student vocab if needed
        if isinstance(self.vocab_projector, nn.Linear):
            teacher_logits = teacher_logits @ self.vocab_projector.weight.T

        # Temperature scaling
        student_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)

        # KL divergence
        kl_loss = F.kl_div(student_probs, teacher_probs, reduction='none')
        kl_loss = kl_loss.sum(dim=-1)

        # Apply mask
        if mask is not None:
            kl_loss = kl_loss * mask
            kl_loss = kl_loss.sum() / mask.sum().clamp(min=1)
        else:
            kl_loss = kl_loss.mean()

        return kl_loss * (self.temperature ** 2)

    def forward(
        self,
        student_outputs: Dict[str, torch.Tensor],
        teacher_outputs: Dict[str, torch.Tensor],
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute complete distillation loss.

        Args:
            student_outputs: Dict with logits, hidden_states, attentions
            teacher_outputs: Dict with logits, hidden_states, attentions
            labels: Ground truth labels for CE loss
            attention_mask: Mask for valid positions

        Returns:
            total_loss: Combined loss
            metrics: Dict with component losses
        """
        metrics = {}
        total_loss = torch.tensor(0.0, device=student_outputs['logits'].device)

        # 1. Cross-entropy loss with labels
        if labels is not None:
            ce_loss = F.cross_entropy(
                student_outputs['logits'].view(-1, student_outputs['logits'].shape[-1]),
                labels.view(-1),
                ignore_index=-100
            )
            total_loss += self.config.alpha_ce * ce_loss
            metrics['ce_loss'] = ce_loss.item()

        # 2. KD loss on logits
        if 'logits' in teacher_outputs:
            kd_loss = self.compute_kd_loss(
                student_outputs['logits'],
                teacher_outputs['logits'],
                attention_mask
            )
            total_loss += self.config.alpha_kd * kd_loss
            metrics['kd_loss'] = kd_loss.item()

        # 3. Hidden state matching
        if 'hidden_states' in student_outputs and 'hidden_states' in teacher_outputs:
            student_hidden = student_outputs['hidden_states']
            teacher_hidden = teacher_outputs['hidden_states']

            if student_hidden and teacher_hidden:
                hidden_loss = self.layer_distillation(student_hidden, teacher_hidden)
                total_loss += self.config.alpha_hidden * hidden_loss
                metrics['hidden_loss'] = hidden_loss.item()

        # 4. Attention transfer
        if 'attentions' in student_outputs and 'attentions' in teacher_outputs:
            student_attn = student_outputs['attentions']
            teacher_attn = teacher_outputs['attentions']

            if student_attn and teacher_attn:
                attn_loss = self.attention_transfer(student_attn, teacher_attn)
                total_loss += self.config.alpha_attn * attn_loss
                metrics['attn_loss'] = attn_loss.item()

        metrics['total_loss'] = total_loss.item()

        return total_loss, metrics


def create_distillation_module(
    config: Optional[DistillationConfig] = None,
    student_config: Optional[Dict] = None,
    teacher_config: Optional[Dict] = None
) -> DistillationLossModule:
    """
    Factory function to create distillation module.
    """
    if config is None:
        config = DistillationConfig()

    student_hidden = student_config.get('hidden_size', 768) if student_config else 768
    teacher_hidden = teacher_config.get('hidden_size', 4096) if teacher_config else 4096
    student_vocab = student_config.get('vocab_size', 50257) if student_config else 50257
    teacher_vocab = teacher_config.get('vocab_size', 151936) if teacher_config else 151936

    return DistillationLossModule(
        config=config,
        student_hidden_dim=student_hidden,
        teacher_hidden_dim=teacher_hidden,
        student_vocab_size=student_vocab,
        teacher_vocab_size=teacher_vocab
    )


if __name__ == "__main__":
    # Test distillation module
    print("Testing Knowledge Distillation Module...")

    config = DistillationConfig(
        temperature=2.0,
        alpha_ce=0.5,
        alpha_kd=0.3,
        alpha_hidden=0.15,
        alpha_attn=0.05
    )

    distill_module = create_distillation_module(
        config=config,
        student_config={'hidden_size': 768, 'vocab_size': 50257},
        teacher_config={'hidden_size': 4096, 'vocab_size': 151936}
    )

    # Create dummy outputs
    B, L, V_s, V_t = 4, 32, 50257, 151936
    H_s, H_t = 768, 4096

    student_outputs = {
        'logits': torch.randn(B, L, V_s),
        'hidden_states': [torch.randn(B, L, H_s) for _ in range(6)],
        'attentions': [torch.randn(B, 12, L, L).softmax(dim=-1) for _ in range(6)]
    }

    teacher_outputs = {
        'logits': torch.randn(B, L, V_t),
        'hidden_states': [torch.randn(B, L, H_t) for _ in range(32)],
        'attentions': [torch.randn(B, 32, L, L).softmax(dim=-1) for _ in range(32)]
    }

    labels = torch.randint(0, V_s, (B, L))
    attention_mask = torch.ones(B, L)

    loss, metrics = distill_module(
        student_outputs,
        teacher_outputs,
        labels,
        attention_mask
    )

    print(f"Total loss: {loss.item():.4f}")
    print(f"Metrics: {metrics}")

    print("Distillation module tests complete!")

