"""
EmberVLM: Complete Multimodal Vision-Language Model
Ultra-efficient VLM for robot fleet reasoning on Raspberry Pi Zero.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple, Union
from dataclasses import dataclass
import logging
import yaml
from pathlib import Path

from .vision_encoder import VisionEncoderWrapper, create_vision_encoder
from .language_model import LanguageModelWrapper, create_language_model
from .fusion_module import MultimodalFusionModule, create_fusion_module

logger = logging.getLogger(__name__)


@dataclass
class EmberVLMConfig:
    """Configuration for EmberVLM model."""

    # Vision encoder
    vision_encoder_type: str = "repvit_m0_9"
    vision_output_dim: int = 384
    num_vision_tokens: int = 8
    vision_pretrained: bool = True
    vision_frozen: bool = True

    # Language model
    vocab_size: int = 50257
    hidden_size: int = 768
    num_layers: int = 6
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 512
    dropout: float = 0.1
    freeze_lm_layers: List[int] = None
    trainable_lm_layers: List[int] = None

    # Fusion module
    fusion_bottleneck_ratio: int = 16
    num_cross_attention_layers: int = 2

    # Special tokens
    image_token_id: int = 50256

    def __post_init__(self):
        if self.freeze_lm_layers is None:
            self.freeze_lm_layers = [0, 1, 2, 3]
        if self.trainable_lm_layers is None:
            self.trainable_lm_layers = [4, 5]

    @classmethod
    def from_yaml(cls, path: str) -> 'EmberVLMConfig':
        """Load config from YAML file."""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'EmberVLMConfig':
        """Create config from dictionary."""
        model_config = config_dict.get('model', {})
        vision_config = model_config.get('vision_encoder', {})
        lm_config = model_config.get('language_model', {})
        fusion_config = model_config.get('fusion_module', {})

        return cls(
            vision_encoder_type=vision_config.get('type', 'repvit_m0_9'),
            vision_output_dim=vision_config.get('output_dim', 384),
            num_vision_tokens=vision_config.get('num_vision_tokens', 8),
            vision_pretrained=vision_config.get('pretrained', True),
            vision_frozen=vision_config.get('frozen', True),
            vocab_size=lm_config.get('vocab_size', 50257),
            hidden_size=lm_config.get('hidden_size', 768),
            num_layers=lm_config.get('num_layers', 6),
            num_attention_heads=lm_config.get('num_attention_heads', 12),
            intermediate_size=lm_config.get('intermediate_size', 3072),
            max_position_embeddings=lm_config.get('max_position_embeddings', 512),
            dropout=lm_config.get('dropout', 0.1),
            freeze_lm_layers=lm_config.get('freeze_layers'),
            trainable_lm_layers=lm_config.get('trainable_layers'),
            fusion_bottleneck_ratio=fusion_config.get('bottleneck_ratio', 16),
            num_cross_attention_layers=fusion_config.get('num_cross_attention_layers', 2)
        )

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            'model': {
                'vision_encoder': {
                    'type': self.vision_encoder_type,
                    'output_dim': self.vision_output_dim,
                    'num_vision_tokens': self.num_vision_tokens,
                    'pretrained': self.vision_pretrained,
                    'frozen': self.vision_frozen,
                },
                'language_model': {
                    'vocab_size': self.vocab_size,
                    'hidden_size': self.hidden_size,
                    'num_layers': self.num_layers,
                    'num_attention_heads': self.num_attention_heads,
                    'intermediate_size': self.intermediate_size,
                    'max_position_embeddings': self.max_position_embeddings,
                    'dropout': self.dropout,
                    'freeze_layers': self.freeze_lm_layers,
                    'trainable_layers': self.trainable_lm_layers,
                },
                'fusion_module': {
                    'bottleneck_ratio': self.fusion_bottleneck_ratio,
                    'num_cross_attention_layers': self.num_cross_attention_layers,
                }
            }
        }


class EmberVLM(nn.Module):
    """
    EmberVLM: Ultra-efficient multimodal Vision-Language Model.

    Total Parameters: ~35M
    - Vision Encoder (RepViT-XXS-M0.9): ~5M (frozen)
    - Language Model (TinyLLM-30M): ~30M (partial fine-tuning)
    - Fusion Module: ~0.5M (fully trainable)

    Features:
    - Robot fleet selection reasoning
    - Incident response planning
    - Action sequence generation
    - Attention visualization for interpretability

    Target Deployment: Raspberry Pi Zero (<100MB, <500ms inference)
    """

    def __init__(self, config: EmberVLMConfig):
        super().__init__()
        self.config = config

        # Initialize vision encoder (frozen)
        self.vision_encoder = VisionEncoderWrapper(
            encoder_type=config.vision_encoder_type,
            num_vision_tokens=config.num_vision_tokens,
            pretrained=config.vision_pretrained,
            freeze=config.vision_frozen
        )

        # Initialize language model (partial fine-tuning)
        self.language_model = LanguageModelWrapper(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            num_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            max_position_embeddings=config.max_position_embeddings,
            dropout=config.dropout,
            freeze_layers=config.freeze_lm_layers,
            trainable_layers=config.trainable_lm_layers
        )

        # Initialize fusion module (fully trainable)
        self.fusion_module = MultimodalFusionModule(
            vision_dim=config.vision_output_dim,
            language_dim=config.hidden_size,
            num_cross_attention_layers=config.num_cross_attention_layers,
            bottleneck_ratio=config.fusion_bottleneck_ratio,
            dropout=config.dropout,
            num_vision_tokens=config.num_vision_tokens
        )

        # Store metadata
        self._log_model_info()

    def _log_model_info(self):
        """Log model architecture information."""
        vision_total, vision_train = self.vision_encoder.encoder.count_parameters()
        lm_total, lm_train = self.language_model.model.count_parameters()
        fusion_total, fusion_train = self.fusion_module.count_parameters()

        total = vision_total + lm_total + fusion_total
        trainable = vision_train + lm_train + fusion_train

        logger.info(f"EmberVLM Model Summary:")
        logger.info(f"  Vision Encoder: {vision_total:,} params ({vision_train:,} trainable)")
        logger.info(f"  Language Model: {lm_total:,} params ({lm_train:,} trainable)")
        logger.info(f"  Fusion Module:  {fusion_total:,} params ({fusion_train:,} trainable)")
        logger.info(f"  Total:          {total:,} params ({trainable:,} trainable, {trainable/total*100:.1f}%)")

        self.total_params = total
        self.trainable_params = trainable

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        vision_features: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the multimodal model.

        Args:
            pixel_values: Input images [B, 3, H, W]
            input_ids: Text token IDs [B, L]
            attention_mask: Attention mask for text [B, L]
            labels: Target labels for language modeling loss [B, L]
            vision_features: Pre-computed vision features (skip encoder)
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return all hidden states

        Returns:
            Dictionary containing:
            - loss: Language modeling loss (if labels provided)
            - logits: Token predictions [B, L, vocab_size]
            - vision_features: Extracted vision features
            - cross_attention_weights: Vision-language attention (if requested)
            - hidden_states: All layer hidden states (if requested)
        """
        outputs = {}

        # Step 1: Extract vision features (if not provided)
        if vision_features is None and pixel_values is not None:
            vision_features, spatial_features = self.vision_encoder(
                pixel_values,
                return_spatial=output_attentions
            )
            if output_attentions:
                outputs['spatial_vision_features'] = spatial_features

        outputs['vision_features'] = vision_features

        # Step 2: Get text embeddings
        text_embeddings = self.language_model.model.embed_tokens(input_ids)

        # Step 3: Prepare multimodal inputs
        if vision_features is not None:
            # Concatenate vision and text embeddings
            multimodal_embeds, multimodal_mask = self.fusion_module.prepare_multimodal_inputs(
                vision_features=vision_features,
                text_embeddings=text_embeddings,
                text_mask=attention_mask
            )

            # Adjust labels if provided (add -100 for vision tokens)
            if labels is not None:
                B = labels.shape[0]
                vision_labels = torch.full(
                    (B, self.config.num_vision_tokens),
                    -100,
                    device=labels.device,
                    dtype=labels.dtype
                )
                labels = torch.cat([vision_labels, labels], dim=1)
        else:
            multimodal_embeds = text_embeddings
            multimodal_mask = attention_mask

        # Step 4: Forward through language model
        lm_outputs = self.language_model(
            inputs_embeds=multimodal_embeds,
            attention_mask=multimodal_mask,
            labels=labels,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions
        )

        # Step 5: Apply cross-modal fusion (on last hidden state)
        if vision_features is not None and output_attentions:
            # Get language portion of hidden states
            last_hidden = lm_outputs['last_hidden_state']
            text_hidden = last_hidden[:, self.config.num_vision_tokens:, :]

            fusion_outputs = self.fusion_module(
                vision_features=vision_features,
                language_hidden_states=text_hidden,
                output_attentions=True
            )
            outputs['cross_attention_weights'] = fusion_outputs.get('attention_weights')
            outputs['fusion_gate'] = fusion_outputs.get('gate_value')

        # Collect outputs
        outputs['logits'] = lm_outputs['logits']

        if labels is not None:
            outputs['loss'] = lm_outputs['loss']

        if output_hidden_states:
            outputs['hidden_states'] = lm_outputs['hidden_states']

        if output_attentions:
            outputs['attentions'] = lm_outputs['attentions']

        return outputs

    @torch.no_grad()
    def generate(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Generate text conditioned on image and prompt.

        Args:
            pixel_values: Input images [B, 3, H, W]
            input_ids: Starting prompt token IDs [B, L]
            attention_mask: Attention mask for prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus sampling threshold
            repetition_penalty: Penalty for repeated tokens
            do_sample: Whether to sample (False = greedy)
            eos_token_id: End of sequence token
            pad_token_id: Padding token

        Returns:
            Dictionary with generated_ids and optionally attention maps
        """
        # Extract vision features
        vision_features = None
        if pixel_values is not None:
            vision_features, _ = self.vision_encoder(pixel_values, return_spatial=False)

        # Get text embeddings
        text_embeddings = self.language_model.model.embed_tokens(input_ids)

        # Prepare multimodal inputs
        if vision_features is not None:
            multimodal_embeds, multimodal_mask = self.fusion_module.prepare_multimodal_inputs(
                vision_features=vision_features,
                text_embeddings=text_embeddings,
                text_mask=attention_mask
            )
        else:
            multimodal_embeds = text_embeddings
            multimodal_mask = attention_mask

        B, L, _ = multimodal_embeds.shape
        past_key_values = None
        generated_ids = []

        for step in range(max_new_tokens):
            # Forward pass
            if past_key_values is None:
                inputs = multimodal_embeds
            else:
                # Only use the last generated embedding
                last_token_id = generated_ids[-1]
                inputs = self.language_model.model.embed_tokens(last_token_id.unsqueeze(1))

            lm_outputs = self.language_model.model(
                inputs_embeds=inputs,
                past_key_values=past_key_values,
                use_cache=True
            )

            logits = lm_outputs['logits'][:, -1, :]
            past_key_values = lm_outputs['past_key_values']

            # Apply repetition penalty
            if repetition_penalty != 1.0 and generated_ids:
                all_generated = torch.stack(generated_ids, dim=1)
                for b in range(B):
                    for token_id in all_generated[b].unique():
                        logits[b, token_id] /= repetition_penalty

            if do_sample:
                # Apply temperature
                logits = logits / temperature

                # Top-k filtering
                if top_k > 0:
                    top_k_vals, _ = torch.topk(logits, top_k)
                    filter_value = top_k_vals[:, -1].unsqueeze(-1)
                    logits = torch.where(
                        logits < filter_value,
                        torch.full_like(logits, float('-inf')),
                        logits
                    )

                # Top-p filtering
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

                # Sample
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                # Greedy
                next_token = logits.argmax(dim=-1)

            generated_ids.append(next_token)

            # Check for EOS
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        # Combine generated tokens
        generated_ids = torch.stack(generated_ids, dim=1)

        # Concatenate with input (excluding vision tokens for output)
        output_ids = torch.cat([input_ids, generated_ids], dim=1)

        return {
            'generated_ids': output_ids,
            'vision_features': vision_features
        }

    def get_vision_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Extract vision features from images."""
        features, _ = self.vision_encoder(pixel_values, return_spatial=False)
        return features

    def count_parameters(self) -> Dict[str, Tuple[int, int]]:
        """Get parameter counts for each component."""
        vision_total, vision_train = self.vision_encoder.encoder.count_parameters()
        lm_total, lm_train = self.language_model.model.count_parameters()
        fusion_total, fusion_train = self.fusion_module.count_parameters()

        return {
            'vision_encoder': (vision_total, vision_train),
            'language_model': (lm_total, lm_train),
            'fusion_module': (fusion_total, fusion_train),
            'total': (
                vision_total + lm_total + fusion_total,
                vision_train + lm_train + fusion_train
            )
        }

    def save_pretrained(self, save_directory: str):
        """Save model weights and config."""
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save config
        config_path = save_path / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(self.config.to_dict(), f)

        # Save model weights
        model_path = save_path / "pytorch_model.bin"
        torch.save(self.state_dict(), model_path)

        logger.info(f"Model saved to {save_directory}")

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs) -> 'EmberVLM':
        """Load model from pretrained weights."""
        model_path = Path(model_path)

        # Load config
        config_path = model_path / "config.yaml"
        config = EmberVLMConfig.from_yaml(str(config_path))

        # Create model
        model = cls(config)

        # Load weights
        weights_path = model_path / "pytorch_model.bin"
        if weights_path.exists():
            state_dict = torch.load(weights_path, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
            logger.info(f"Loaded pretrained weights from {model_path}")

        return model


def create_embervlm(config_path: Optional[str] = None, **kwargs) -> EmberVLM:
    """Factory function to create EmberVLM model."""
    if config_path:
        config = EmberVLMConfig.from_yaml(config_path)
    else:
        config = EmberVLMConfig(**kwargs)
    return EmberVLM(config)


if __name__ == "__main__":
    # Test EmberVLM
    print("Testing EmberVLM Complete Model...")

    config = EmberVLMConfig(
        vision_pretrained=False,  # Don't download for test
        vision_frozen=True,
        freeze_lm_layers=[0, 1, 2, 3],
        trainable_lm_layers=[4, 5]
    )

    model = EmberVLM(config)

    # Test inputs
    batch_size = 2
    pixel_values = torch.randn(batch_size, 3, 224, 224)
    input_ids = torch.randint(0, 50257, (batch_size, 32))
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()

    # Forward pass
    print("\nForward pass with images and text...")
    outputs = model(
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        output_attentions=True,
        output_hidden_states=True
    )

    print(f"Input image shape: {pixel_values.shape}")
    print(f"Input text shape: {input_ids.shape}")
    print(f"Output logits shape: {outputs['logits'].shape}")
    print(f"Loss: {outputs['loss'].item():.4f}")
    print(f"Vision features shape: {outputs['vision_features'].shape}")

    # Generation test
    print("\nGeneration test...")
    gen_outputs = model.generate(
        pixel_values=pixel_values,
        input_ids=input_ids[:, :10],  # Short prompt
        max_new_tokens=20,
        temperature=0.7,
        do_sample=True
    )
    print(f"Generated sequence shape: {gen_outputs['generated_ids'].shape}")

    # Parameter counts
    print("\nParameter counts:")
    param_counts = model.count_parameters()
    for component, (total, trainable) in param_counts.items():
        print(f"  {component}: {total:,} total, {trainable:,} trainable")

