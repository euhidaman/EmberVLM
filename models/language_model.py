"""
EmberVLM Language Model: TinyLLM-30M wrapper
Lightweight GPT-2 style decoder with partial fine-tuning support.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List, Union
import math
import logging

logger = logging.getLogger(__name__)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""

    def __init__(self, dim: int, max_seq_len: int = 512, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        # Precompute sin/cos for max sequence length
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos()[None, None, :, :])
        self.register_buffer('sin_cached', emb.sin()[None, None, :, :])

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
        return self.cos_cached[:, :, :seq_len, :], self.sin_cached[:, :, :seq_len, :]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor,
                         cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding to query and key tensors."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with optional KV cache."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.1,
        max_seq_len: int = 512
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert self.head_dim * num_heads == hidden_size

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.rotary = RotaryEmbedding(self.head_dim, max_seq_len)

        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor]]:
        B, L, _ = hidden_states.shape

        # Project Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply rotary embeddings
        cos, sin = self.rotary(q, L)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Handle KV cache
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)

        if use_cache:
            past_key_value = (k, v)

        # Compute attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply causal mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # Compute output
        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, -1)
        attn_output = self.o_proj(attn_output)

        outputs = (attn_output, past_key_value if use_cache else None)
        if output_attentions:
            outputs = outputs + (attn_probs,)

        return outputs


class MLP(nn.Module):
    """Feed-forward network with GELU activation."""

    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.1):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU-style activation
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class TransformerBlock(nn.Module):
    """Single transformer decoder block."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        dropout: float = 0.1,
        max_seq_len: int = 512
    ):
        super().__init__()
        self.attention = MultiHeadAttention(hidden_size, num_heads, dropout, max_seq_len)
        self.mlp = MLP(hidden_size, intermediate_size, dropout)
        self.norm1 = RMSNorm(hidden_size)
        self.norm2 = RMSNorm(hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        # Pre-norm architecture
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)

        attn_outputs = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions
        )

        hidden_states = residual + attn_outputs[0]

        # MLP
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)

        outputs = (hidden_states,) + attn_outputs[1:]
        return outputs


class TinyLLM(nn.Module):
    """
    TinyLLM-30M: Lightweight GPT-2 style language model.

    Architecture:
    - 6 transformer blocks
    - 768 hidden dimensions
    - 12 attention heads
    - ~30M parameters

    Features partial fine-tuning support for efficient multimodal adaptation.
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        hidden_size: int = 768,
        num_layers: int = 6,
        num_heads: int = 12,
        intermediate_size: int = 3072,
        max_position_embeddings: int = 512,
        dropout: float = 0.1,
        tie_word_embeddings: bool = True,
        freeze_layers: Optional[List[int]] = None,
        trainable_layers: Optional[List[int]] = None
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_position_embeddings = max_position_embeddings

        # Token embeddings
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.embed_dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                intermediate_size=intermediate_size,
                dropout=dropout,
                max_seq_len=max_position_embeddings
            )
            for _ in range(num_layers)
        ])

        # Final norm
        self.norm = RMSNorm(hidden_size)

        # LM head
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        # Tie embeddings
        if tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        # Initialize weights
        self.apply(self._init_weights)

        # Freeze specified layers
        if freeze_layers:
            self._freeze_layers(freeze_layers)
        if trainable_layers:
            self._set_trainable_layers(trainable_layers)

    def _init_weights(self, module: nn.Module):
        """Initialize weights using GPT-2 initialization."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _freeze_layers(self, layer_indices: List[int]):
        """Freeze specified transformer layers."""
        for idx in layer_indices:
            if idx < len(self.layers):
                for param in self.layers[idx].parameters():
                    param.requires_grad = False
                logger.info(f"Froze layer {idx}")

    def _set_trainable_layers(self, layer_indices: List[int]):
        """Set only specified layers as trainable."""
        # First freeze all layers
        for layer in self.layers:
            for param in layer.parameters():
                param.requires_grad = False

        # Then unfreeze specified layers
        for idx in layer_indices:
            if idx < len(self.layers):
                for param in self.layers[idx].parameters():
                    param.requires_grad = True
                logger.info(f"Set layer {idx} as trainable")

    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal attention mask."""
        mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), device=device),
            diagonal=1
        )
        return mask[None, None, :, :]

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the language model.

        Args:
            input_ids: Token IDs [B, L]
            inputs_embeds: Pre-computed embeddings [B, L, H] (alternative to input_ids)
            attention_mask: Attention mask [B, L]
            past_key_values: Cached key-value pairs for generation
            use_cache: Whether to return new KV cache
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return all hidden states

        Returns:
            Dictionary with logits, optional hidden states, attentions, and KV cache
        """
        if input_ids is not None:
            B, L = input_ids.shape
            hidden_states = self.embed_tokens(input_ids)
        elif inputs_embeds is not None:
            B, L, _ = inputs_embeds.shape
            hidden_states = inputs_embeds
        else:
            raise ValueError("Must provide either input_ids or inputs_embeds")

        hidden_states = self.embed_dropout(hidden_states)

        # Create causal mask
        causal_mask = self._create_causal_mask(L, hidden_states.device)

        # Apply attention mask if provided
        if attention_mask is not None:
            # Convert [B, L] -> [B, 1, 1, L] and combine with causal mask
            padding_mask = attention_mask[:, None, None, :]
            padding_mask = (1.0 - padding_mask) * float('-inf')
            causal_mask = causal_mask + padding_mask

        # Process through transformer layers
        all_hidden_states = [] if output_hidden_states else None
        all_attentions = [] if output_attentions else None
        new_past_key_values = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

            past_kv = past_key_values[i] if past_key_values is not None else None

            layer_outputs = layer(
                hidden_states,
                attention_mask=causal_mask,
                past_key_value=past_kv,
                use_cache=use_cache,
                output_attentions=output_attentions
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                new_past_key_values.append(layer_outputs[1])
            if output_attentions:
                all_attentions.append(layer_outputs[-1])

        # Final norm
        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        # LM head
        logits = self.lm_head(hidden_states)

        return {
            'logits': logits,
            'hidden_states': all_hidden_states,
            'attentions': all_attentions,
            'past_key_values': new_past_key_values,
            'last_hidden_state': hidden_states
        }

    def get_input_embeddings(self) -> nn.Embedding:
        """Return input embedding layer."""
        return self.embed_tokens

    def set_input_embeddings(self, embeddings: nn.Embedding):
        """Set input embedding layer."""
        self.embed_tokens = embeddings

    def count_parameters(self) -> Tuple[int, int]:
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate text using the language model.

        Args:
            input_ids: Starting token IDs [B, L]
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus sampling threshold
            repetition_penalty: Penalty for repeating tokens
            eos_token_id: End of sequence token
            pad_token_id: Padding token

        Returns:
            Generated token IDs [B, L + new_tokens]
        """
        B = input_ids.shape[0]
        past_key_values = None

        for _ in range(max_new_tokens):
            # Forward pass
            outputs = self.forward(
                input_ids=input_ids if past_key_values is None else input_ids[:, -1:],
                past_key_values=past_key_values,
                use_cache=True
            )

            logits = outputs['logits'][:, -1, :]
            past_key_values = outputs['past_key_values']

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for b in range(B):
                    for token_id in input_ids[b].unique():
                        logits[b, token_id] /= repetition_penalty

            # Apply temperature
            logits = logits / temperature

            # Top-k filtering
            if top_k > 0:
                top_k_vals, _ = torch.topk(logits, top_k)
                filter_value = top_k_vals[:, -1].unsqueeze(-1)
                logits = torch.where(logits < filter_value,
                                    torch.full_like(logits, float('-inf')),
                                    logits)

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False

                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits = logits.masked_fill(indices_to_remove, float('-inf'))

            # Sample next token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=-1)

            # Check for EOS
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return input_ids


class LanguageModelWrapper(nn.Module):
    """
    Wrapper for TinyLLM with tokenizer integration and convenient methods.
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        hidden_size: int = 768,
        num_layers: int = 6,
        num_heads: int = 12,
        intermediate_size: int = 3072,
        max_position_embeddings: int = 512,
        dropout: float = 0.1,
        freeze_layers: Optional[List[int]] = None,
        trainable_layers: Optional[List[int]] = None,
        pretrained_path: Optional[str] = None
    ):
        super().__init__()

        self.model = TinyLLM(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_position_embeddings,
            dropout=dropout,
            freeze_layers=freeze_layers,
            trainable_layers=trainable_layers
        )

        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        if pretrained_path:
            self._load_pretrained(pretrained_path)

    def _load_pretrained(self, path: str):
        """Load pretrained weights."""
        try:
            state_dict = torch.load(path, map_location='cpu')
            self.model.load_state_dict(state_dict, strict=False)
            logger.info(f"Loaded pretrained weights from {path}")
        except Exception as e:
            logger.warning(f"Could not load pretrained weights: {e}")

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with optional loss computation."""
        outputs = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **kwargs
        )

        if labels is not None:
            # Compute cross-entropy loss
            logits = outputs['logits']
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100
            )
            outputs['loss'] = loss

        return outputs

    def get_hidden_size(self) -> int:
        return self.hidden_size


def create_language_model(config: dict) -> LanguageModelWrapper:
    """Factory function to create language model from config."""
    return LanguageModelWrapper(
        vocab_size=config.get('vocab_size', 50257),
        hidden_size=config.get('hidden_size', 768),
        num_layers=config.get('num_layers', 6),
        num_heads=config.get('num_attention_heads', 12),
        intermediate_size=config.get('intermediate_size', 3072),
        max_position_embeddings=config.get('max_position_embeddings', 512),
        dropout=config.get('dropout', 0.1),
        freeze_layers=config.get('freeze_layers'),
        trainable_layers=config.get('trainable_layers'),
        pretrained_path=config.get('pretrained_path')
    )


if __name__ == "__main__":
    # Test the language model
    print("Testing TinyLLM Language Model...")

    model = LanguageModelWrapper(
        vocab_size=50257,
        hidden_size=768,
        num_layers=6,
        num_heads=12,
        intermediate_size=3072,
        max_position_embeddings=512,
        freeze_layers=[0, 1, 2, 3],  # Freeze first 4 layers
        trainable_layers=[4, 5]       # Train last 2 layers
    )

    # Test input
    input_ids = torch.randint(0, 50257, (2, 64))
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()

    # Forward pass
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        output_hidden_states=True
    )

    print(f"Input shape: {input_ids.shape}")
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Loss: {outputs['loss'].item():.4f}")
    print(f"Number of hidden states: {len(outputs['hidden_states'])}")

    total, trainable = model.model.count_parameters()
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Trainable ratio: {trainable/total*100:.2f}%")

