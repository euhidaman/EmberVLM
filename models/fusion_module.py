"""
EmberVLM Fusion Module: Lightweight cross-modal adapter
Connects vision features to language model embedding space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import math
import logging

logger = logging.getLogger(__name__)


class CrossAttentionLayer(nn.Module):
    """
    Cross-attention layer for vision-language fusion.
    Query from language, Key/Value from vision.
    """

    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        bottleneck_ratio: int = 16
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Bottleneck dimension for efficiency
        self.bottleneck_dim = query_dim // bottleneck_ratio

        # Down projection
        self.q_down = nn.Linear(query_dim, self.bottleneck_dim, bias=False)
        self.k_down = nn.Linear(key_dim, self.bottleneck_dim, bias=False)
        self.v_down = nn.Linear(key_dim, self.bottleneck_dim, bias=False)

        # Attention (in bottleneck space)
        self.bottleneck_heads = max(1, self.bottleneck_dim // 32)
        self.bottleneck_head_dim = self.bottleneck_dim // self.bottleneck_heads

        # Up projection
        self.out_proj = nn.Linear(self.bottleneck_dim, query_dim, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(query_dim)

        # Initialize with small values for stable training
        nn.init.normal_(self.out_proj.weight, std=0.01)

    def forward(
        self,
        query: torch.Tensor,  # [B, L_q, D_q]
        key_value: torch.Tensor,  # [B, L_kv, D_kv]
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Cross-attention from language tokens to vision tokens.

        Args:
            query: Language hidden states [B, L_text, hidden_size]
            key_value: Vision features [B, L_vision, vision_dim]
            attention_mask: Optional mask for vision tokens
            output_attentions: Whether to return attention weights

        Returns:
            output: Updated query tensor [B, L_text, hidden_size]
            attention_weights: Optional attention map
        """
        B, L_q, D = query.shape
        L_kv = key_value.shape[1]

        residual = query
        query = self.layer_norm(query)

        # Project to bottleneck
        q = self.q_down(query)  # [B, L_q, bottleneck]
        k = self.k_down(key_value)  # [B, L_kv, bottleneck]
        v = self.v_down(key_value)  # [B, L_kv, bottleneck]

        # Reshape for multi-head attention
        q = q.view(B, L_q, self.bottleneck_heads, self.bottleneck_head_dim).transpose(1, 2)
        k = k.view(B, L_kv, self.bottleneck_heads, self.bottleneck_head_dim).transpose(1, 2)
        v = v.view(B, L_kv, self.bottleneck_heads, self.bottleneck_head_dim).transpose(1, 2)

        # Compute attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * (self.bottleneck_head_dim ** -0.5)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # Compute output
        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L_q, -1)

        # Up project and residual
        output = residual + self.out_proj(attn_output)

        if output_attentions:
            return output, attn_probs
        return output, None


class VisionProjector(nn.Module):
    """
    Project vision features to language model hidden space.
    """

    def __init__(
        self,
        vision_dim: int,
        language_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()

        # Two-layer MLP projection
        self.proj = nn.Sequential(
            nn.Linear(vision_dim, language_dim),
            nn.LayerNorm(language_dim),
            nn.GELU(),
            nn.Linear(language_dim, language_dim),
            nn.LayerNorm(language_dim),
            nn.Dropout(dropout)
        )

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        Project vision features to language dimension.

        Args:
            vision_features: [B, num_vision_tokens, vision_dim]

        Returns:
            projected: [B, num_vision_tokens, language_dim]
        """
        return self.proj(vision_features)


class MultimodalFusionModule(nn.Module):
    """
    Lightweight multimodal fusion module (~500K parameters).

    Components:
    1. Vision projector: Linear(384 -> 768) + LayerNorm
    2. Cross-attention adapters: 2 layers with bottleneck ratio 1/16

    Design inspired by TinyGPT-V but scaled 10x smaller.
    """

    def __init__(
        self,
        vision_dim: int = 384,
        language_dim: int = 768,
        num_cross_attention_layers: int = 2,
        bottleneck_ratio: int = 16,
        dropout: float = 0.1,
        num_vision_tokens: int = 8
    ):
        super().__init__()

        self.vision_dim = vision_dim
        self.language_dim = language_dim
        self.num_vision_tokens = num_vision_tokens

        # Vision projection
        self.vision_projector = VisionProjector(
            vision_dim=vision_dim,
            language_dim=language_dim,
            dropout=dropout
        )

        # Cross-attention layers
        self.cross_attention_layers = nn.ModuleList([
            CrossAttentionLayer(
                query_dim=language_dim,
                key_dim=language_dim,  # After projection
                bottleneck_ratio=bottleneck_ratio,
                dropout=dropout
            )
            for _ in range(num_cross_attention_layers)
        ])

        # Learnable vision position embeddings
        self.vision_position_embedding = nn.Parameter(
            torch.randn(1, num_vision_tokens, language_dim) * 0.02
        )

        # Gating mechanism for gradual fusion
        self.gate = nn.Parameter(torch.zeros(1))

        logger.info(f"Initialized MultimodalFusionModule with {self.count_parameters()[0]:,} parameters")

    def forward(
        self,
        vision_features: torch.Tensor,
        language_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Fuse vision and language features.

        Args:
            vision_features: From vision encoder [B, num_vision_tokens, vision_dim]
            language_hidden_states: From language model [B, L_text, language_dim]
            attention_mask: Optional mask for vision tokens
            output_attentions: Whether to return attention maps

        Returns:
            Dictionary containing:
            - fused_hidden_states: Updated language hidden states
            - projected_vision: Vision features in language space
            - attention_weights: Optional attention maps for visualization
        """
        B = vision_features.shape[0]

        # Project vision features to language dimension
        projected_vision = self.vision_projector(vision_features)

        # Add position embeddings
        projected_vision = projected_vision + self.vision_position_embedding

        # Apply cross-attention layers
        fused_states = language_hidden_states
        all_attention_weights = []

        for cross_attn in self.cross_attention_layers:
            fused_states, attn_weights = cross_attn(
                query=fused_states,
                key_value=projected_vision,
                attention_mask=attention_mask,
                output_attentions=output_attentions
            )
            if output_attentions:
                all_attention_weights.append(attn_weights)

        # Apply gating for stable training
        gate = torch.sigmoid(self.gate)
        fused_states = language_hidden_states + gate * (fused_states - language_hidden_states)

        outputs = {
            'fused_hidden_states': fused_states,
            'projected_vision': projected_vision,
            'gate_value': gate.item()
        }

        if output_attentions:
            outputs['attention_weights'] = all_attention_weights

        return outputs

    def prepare_multimodal_inputs(
        self,
        vision_features: torch.Tensor,
        text_embeddings: torch.Tensor,
        vision_mask: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare concatenated multimodal inputs for the language model.

        Vision tokens are prepended to text tokens:
        [V1, V2, ..., Vn, T1, T2, ..., Tm]

        Args:
            vision_features: [B, num_vision_tokens, vision_dim]
            text_embeddings: [B, L_text, language_dim]
            vision_mask: Optional [B, num_vision_tokens]
            text_mask: Optional [B, L_text]

        Returns:
            multimodal_embeds: [B, num_vision_tokens + L_text, language_dim]
            multimodal_mask: [B, num_vision_tokens + L_text]
        """
        B = vision_features.shape[0]

        # Project vision features
        projected_vision = self.vision_projector(vision_features)
        projected_vision = projected_vision + self.vision_position_embedding

        # Concatenate vision and text
        multimodal_embeds = torch.cat([projected_vision, text_embeddings], dim=1)

        # Create combined attention mask
        if vision_mask is None:
            vision_mask = torch.ones(B, self.num_vision_tokens, device=vision_features.device)
        if text_mask is None:
            text_mask = torch.ones(B, text_embeddings.shape[1], device=text_embeddings.device)

        multimodal_mask = torch.cat([vision_mask, text_mask], dim=1)

        return multimodal_embeds, multimodal_mask

    def count_parameters(self) -> Tuple[int, int]:
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


class FusionWithAdapter(nn.Module):
    """
    Extended fusion module with LoRA-style adapters for efficient fine-tuning.
    """

    def __init__(
        self,
        base_fusion: MultimodalFusionModule,
        adapter_rank: int = 8,
        adapter_alpha: float = 16.0
    ):
        super().__init__()
        self.base_fusion = base_fusion
        self.adapter_rank = adapter_rank
        self.adapter_alpha = adapter_alpha
        self.scaling = adapter_alpha / adapter_rank

        language_dim = base_fusion.language_dim

        # LoRA adapters for vision projector
        self.lora_A = nn.Linear(language_dim, adapter_rank, bias=False)
        self.lora_B = nn.Linear(adapter_rank, language_dim, bias=False)

        # Initialize
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(
        self,
        vision_features: torch.Tensor,
        language_hidden_states: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        # Base fusion
        outputs = self.base_fusion(
            vision_features=vision_features,
            language_hidden_states=language_hidden_states,
            **kwargs
        )

        # Apply LoRA to fused states
        lora_output = self.lora_B(self.lora_A(outputs['fused_hidden_states']))
        outputs['fused_hidden_states'] = outputs['fused_hidden_states'] + self.scaling * lora_output

        return outputs


def create_fusion_module(config: dict) -> MultimodalFusionModule:
    """Factory function to create fusion module from config."""
    return MultimodalFusionModule(
        vision_dim=config.get('input_dim', 384),
        language_dim=config.get('output_dim', 768),
        num_cross_attention_layers=config.get('num_cross_attention_layers', 2),
        bottleneck_ratio=config.get('bottleneck_ratio', 16),
        dropout=config.get('dropout', 0.1),
        num_vision_tokens=config.get('num_vision_tokens', 8)
    )


if __name__ == "__main__":
    # Test the fusion module
    print("Testing Multimodal Fusion Module...")

    fusion = MultimodalFusionModule(
        vision_dim=384,
        language_dim=768,
        num_cross_attention_layers=2,
        bottleneck_ratio=16,
        num_vision_tokens=8
    )

    # Test inputs
    vision_features = torch.randn(2, 8, 384)  # [B, n_vision, vision_dim]
    language_hidden = torch.randn(2, 64, 768)  # [B, L_text, hidden_dim]

    # Forward pass
    outputs = fusion(
        vision_features=vision_features,
        language_hidden_states=language_hidden,
        output_attentions=True
    )

    print(f"Vision features shape: {vision_features.shape}")
    print(f"Language hidden shape: {language_hidden.shape}")
    print(f"Fused hidden shape: {outputs['fused_hidden_states'].shape}")
    print(f"Projected vision shape: {outputs['projected_vision'].shape}")
    print(f"Gate value: {outputs['gate_value']:.4f}")
    print(f"Number of attention maps: {len(outputs['attention_weights'])}")

    # Test multimodal input preparation
    text_embeds = torch.randn(2, 64, 768)
    mm_embeds, mm_mask = fusion.prepare_multimodal_inputs(
        vision_features=vision_features,
        text_embeddings=text_embeds
    )
    print(f"Multimodal embeddings shape: {mm_embeds.shape}")
    print(f"Multimodal mask shape: {mm_mask.shape}")

    total, trainable = fusion.count_parameters()
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")

