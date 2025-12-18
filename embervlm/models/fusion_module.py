"""
Fusion Module for EmberVLM

Bridges vision encoder features to language model space using
adapter blocks with bottleneck design for parameter efficiency.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple


class AdapterBlock(nn.Module):
    """
    Adapter block with bottleneck design.

    Inspired by TinyGPT-V but scaled down 8x for efficiency.
    """

    def __init__(
        self,
        input_dim: int,
        bottleneck_dim: int,
        output_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.down_proj = nn.Linear(input_dim, bottleneck_dim)
        self.up_proj = nn.Linear(bottleneck_dim, output_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Xavier initialization
        nn.init.xavier_uniform_(self.down_proj.weight)
        nn.init.xavier_uniform_(self.up_proj.weight)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through adapter.

        Args:
            x: Input tensor [B, seq_len, input_dim]

        Returns:
            Output tensor [B, seq_len, output_dim]
        """
        residual = x if x.size(-1) == self.up_proj.out_features else None

        x = self.down_proj(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.up_proj(x)
        x = self.layer_norm(x)

        if residual is not None:
            x = x + residual

        return x


class QKNormFusion(nn.Module):
    """
    Query-Key normalization layer for stable fusion.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim, eps=eps)
        self.norm_k = nn.LayerNorm(dim, eps=eps)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.norm_q(query), self.norm_k(key)


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention layer for vision-language fusion.
    """

    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_qk_norm: bool = True,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Projections
        self.q_proj = nn.Linear(query_dim, hidden_dim)
        self.k_proj = nn.Linear(key_dim, hidden_dim)
        self.v_proj = nn.Linear(key_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, query_dim)

        # QK normalization
        self.use_qk_norm = use_qk_norm
        if use_qk_norm:
            self.qk_norm = QKNormFusion(self.head_dim)

        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Cross-attention forward pass.

        Args:
            query: Query tensor [B, query_len, query_dim]
            key: Key tensor [B, key_len, key_dim]
            value: Value tensor (default: same as key)
            attention_mask: Optional attention mask

        Returns:
            Tuple of (output, attention_weights)
        """
        if value is None:
            value = key

        batch_size, query_len, _ = query.size()
        _, key_len, _ = key.size()

        # Project Q, K, V
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Reshape to multi-head
        q = q.view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply QK normalization
        if self.use_qk_norm:
            q, k = self.qk_norm(q, k)

        # Compute attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Apply attention to values
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, query_len, -1)
        output = self.out_proj(output)
        output = self.proj_dropout(output)

        return output, attn_weights


class FusionModule(nn.Module):
    """
    Fusion Module for EmberVLM.

    Maps vision encoder features to language model space using adapter blocks
    with QK-normalization for training stability.

    With tinyllm/30M-0.4 (hidden_size=384), vision and language dimensions match,
    simplifying the fusion architecture.

    Architecture:
        RepViT_Features(8×384) → Linear(384→384) → LayerNorm →
        AdapterBlock(bottleneck=48) → TinyLLM_Input(8×384)
    """

    def __init__(
        self,
        vision_dim: int = 384,
        language_dim: int = 384,  # Match tinyllm/30M-0.4 hidden size
        bottleneck_dim: int = 48,
        num_visual_tokens: int = 8,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
        use_qk_norm: bool = True,
        use_cross_attention: bool = False,
        num_cross_attention_heads: int = 6,  # Match tinyllm/30M-0.4 heads
    ):
        super().__init__()

        self.vision_dim = vision_dim
        self.language_dim = language_dim
        self.bottleneck_dim = bottleneck_dim
        self.num_visual_tokens = num_visual_tokens
        self.use_cross_attention = use_cross_attention

        # Initial projection
        self.vision_proj = nn.Linear(vision_dim, language_dim)

        # Layer normalization
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.pre_norm = nn.LayerNorm(language_dim)

        # Adapter block
        self.adapter = AdapterBlock(
            input_dim=language_dim,
            bottleneck_dim=bottleneck_dim,
            output_dim=language_dim,
            dropout=dropout,
        )

        # Optional cross-attention
        if use_cross_attention:
            self.cross_attention = CrossAttentionFusion(
                query_dim=language_dim,
                key_dim=language_dim,
                hidden_dim=language_dim,
                num_heads=num_cross_attention_heads,
                dropout=dropout,
                use_qk_norm=use_qk_norm,
            )

        # Output layer norm
        self.output_norm = nn.LayerNorm(language_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for projection layers."""
        nn.init.xavier_uniform_(self.vision_proj.weight)
        nn.init.zeros_(self.vision_proj.bias)

    def forward(
        self,
        visual_features: torch.Tensor,
        text_features: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through fusion module.

        Args:
            visual_features: Vision encoder output [B, num_visual_tokens, vision_dim]
            text_features: Optional text features for cross-attention [B, text_len, language_dim]
            attention_mask: Optional attention mask

        Returns:
            Dictionary containing:
                - fused_features: Fused visual features [B, num_visual_tokens, language_dim]
                - attention_weights: Optional attention weights
        """
        # Project vision features to language space
        fused = self.vision_proj(visual_features)

        # Apply pre-normalization
        if self.use_layer_norm:
            fused = self.pre_norm(fused)

        # Apply adapter
        fused = self.adapter(fused)

        # Optional cross-attention with text features
        attention_weights = None
        if self.use_cross_attention and text_features is not None:
            cross_out, attention_weights = self.cross_attention(
                query=fused,
                key=text_features,
                value=text_features,
                attention_mask=attention_mask,
            )
            fused = fused + cross_out

        # Output normalization
        fused = self.output_norm(fused)

        return {
            'fused_features': fused,
            'attention_weights': attention_weights,
        }

    def get_output_dim(self) -> int:
        """Return output dimension."""
        return self.language_dim

    def count_parameters(self) -> Dict[str, int]:
        """Count trainable and total parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'total': total,
            'trainable': trainable,
        }


class MultiScaleFusion(nn.Module):
    """
    Multi-scale fusion for handling different visual feature resolutions.
    """

    def __init__(
        self,
        vision_dims: list = [128, 256, 384],
        language_dim: int = 768,
        bottleneck_dim: int = 48,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, language_dim),
                nn.LayerNorm(language_dim),
                nn.GELU(),
            )
            for dim in vision_dims
        ])

        self.fusion = nn.Sequential(
            nn.Linear(language_dim * len(vision_dims), language_dim),
            nn.LayerNorm(language_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.adapter = AdapterBlock(
            input_dim=language_dim,
            bottleneck_dim=bottleneck_dim,
            output_dim=language_dim,
            dropout=dropout,
        )

    def forward(
        self,
        multi_scale_features: list,
    ) -> torch.Tensor:
        """
        Fuse multi-scale visual features.

        Args:
            multi_scale_features: List of features at different scales

        Returns:
            Fused features [B, seq_len, language_dim]
        """
        projected = []
        for feat, proj in zip(multi_scale_features, self.projections):
            projected.append(proj(feat))

        # Concatenate along feature dimension
        concat = torch.cat(projected, dim=-1)
        fused = self.fusion(concat)
        fused = self.adapter(fused)

        return fused


class VisualProjector(nn.Module):
    """
    Simple visual projector for baseline comparison.
    """

    def __init__(
        self,
        vision_dim: int = 384,
        language_dim: int = 768,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Linear(vision_dim, language_dim),
            nn.LayerNorm(language_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(language_dim, language_dim),
            nn.LayerNorm(language_dim),
        )

    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        return self.proj(visual_features)

