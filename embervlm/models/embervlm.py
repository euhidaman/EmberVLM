"""
EmberVLM - Complete Model Implementation

A lightweight multimodal Vision-Language Model combining RepViT vision encoder
and TinyLLM language backbone for robot fleet selection with incident reasoning.

Uses pretrained models:
- Vision: RepViT-XXS from THU-MIG/RepViT (HuggingFace)
- Language: tinyllm/30M-0.4 from HuggingFace (GPT-2 style, trained on FineWeb + SHL)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass, field

from embervlm.models.vision_encoder import RepViTEncoder, ImagePreprocessor
from embervlm.models.language_model import (
    TinyLLMBackbone,
    TinyLLMConfig,
    PretrainedTinyLLMBackbone,
    create_language_backbone,
    PRETRAINED_TINYLLM_MODEL,
)
from embervlm.models.fusion_module import FusionModule
from embervlm.models.reasoning_heads import ReasoningModule, ReasoningLoss


@dataclass
class EmberVLMConfig:
    """Configuration for EmberVLM model."""

    # Vision encoder
    vision_model: str = "repvit_m0_9"  # Using timm model: repvit_m0_9.dist_450e_in1k
    vision_pretrained: bool = True
    freeze_vision: bool = True
    num_visual_tokens: int = 8
    vision_output_dim: int = 384
    image_size: int = 224

    # Language model (tinyllm/30M-0.4 defaults)
    language_hidden_size: int = 384  # tinyllm/30M-0.4 uses 384
    language_num_layers: int = 6
    language_num_heads: int = 6  # tinyllm/30M-0.4 uses 6 heads
    language_vocab_size: int = 50257
    language_max_length: int = 1024
    freeze_language_base: bool = True
    unfreeze_last_layer: bool = True

    # Pretrained model settings
    use_pretrained_language: bool = True
    pretrained_language_model: str = "tinyllm/30M-0.4"

    # Fusion module
    fusion_bottleneck_dim: int = 48
    fusion_dropout: float = 0.1
    use_qk_norm: bool = True

    # Reasoning module
    reasoning_enabled: bool = True
    reasoning_hidden_dim: int = 192  # Reduced to match smaller language model
    reasoning_num_layers: int = 2
    reasoning_num_heads: int = 4
    num_reasoning_steps: int = 4
    max_plan_steps: int = 5

    # Robot fleet
    num_robots: int = 5
    robot_names: List[str] = field(default_factory=lambda: ["Drone", "Humanoid", "Wheeled", "Legged", "Underwater"])

    # Special tokens
    special_tokens: Dict[str, str] = field(default_factory=lambda: {
        "reasoning_start": "<|reasoning_start|>",
        "reasoning_end": "<|reasoning_end|>",
        "robot_selection": "<|robot_selection|>",
        "action_plan": "<|action_plan|>",
        "image_token": "<|image|>",
    })

    # Training
    dropout: float = 0.1
    initializer_range: float = 0.02

    def to_dict(self) -> Dict[str, Any]:
        return {
            'vision_model': self.vision_model,
            'vision_pretrained': self.vision_pretrained,
            'freeze_vision': self.freeze_vision,
            'num_visual_tokens': self.num_visual_tokens,
            'vision_output_dim': self.vision_output_dim,
            'image_size': self.image_size,
            'language_hidden_size': self.language_hidden_size,
            'language_num_layers': self.language_num_layers,
            'language_num_heads': self.language_num_heads,
            'language_vocab_size': self.language_vocab_size,
            'language_max_length': self.language_max_length,
            'freeze_language_base': self.freeze_language_base,
            'unfreeze_last_layer': self.unfreeze_last_layer,
            'use_pretrained_language': self.use_pretrained_language,
            'pretrained_language_model': self.pretrained_language_model,
            'fusion_bottleneck_dim': self.fusion_bottleneck_dim,
            'fusion_dropout': self.fusion_dropout,
            'use_qk_norm': self.use_qk_norm,
            'reasoning_enabled': self.reasoning_enabled,
            'reasoning_hidden_dim': self.reasoning_hidden_dim,
            'reasoning_num_layers': self.reasoning_num_layers,
            'reasoning_num_heads': self.reasoning_num_heads,
            'num_reasoning_steps': self.num_reasoning_steps,
            'max_plan_steps': self.max_plan_steps,
            'num_robots': self.num_robots,
            'robot_names': self.robot_names,
            'special_tokens': self.special_tokens,
            'dropout': self.dropout,
            'initializer_range': self.initializer_range,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'EmberVLMConfig':
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})


class EmberVLM(nn.Module):
    """
    EmberVLM - Tiny Multimodal VLM for Robot Fleet Selection.

    Architecture:
    - Vision Encoder: RepViT-XXS (frozen, ~5M params)
    - Language Model: TinyLLM-30M (last layer trainable)
    - Fusion Module: Adapter-based fusion (~1M params)
    - Reasoning Module: CoT generation heads (~3M params)

    Total: ~35M parameters (~5M trainable)
    """

    def __init__(self, config: Optional[EmberVLMConfig] = None):
        super().__init__()

        if config is None:
            config = EmberVLMConfig()

        self.config = config

        # Initialize components
        self._build_vision_encoder()
        self._build_language_model()
        self._build_fusion_module()

        if config.reasoning_enabled:
            self._build_reasoning_module()

        # Image preprocessor
        self.image_preprocessor = ImagePreprocessor(image_size=config.image_size)

        # Loss function
        self.reasoning_loss = ReasoningLoss(
            num_robots=config.num_robots,
        )

        # Special token IDs (will be set when tokenizer is loaded)
        self.special_token_ids = {}

    def _build_vision_encoder(self):
        """Initialize vision encoder."""
        self.vision_encoder = RepViTEncoder(
            model_name=self.config.vision_model,
            pretrained=self.config.vision_pretrained,
            freeze=self.config.freeze_vision,
            num_visual_tokens=self.config.num_visual_tokens,
            output_dim=self.config.vision_output_dim,
            image_size=self.config.image_size,
        )

    def _build_language_model(self):
        """Initialize language model (pretrained or from scratch)."""
        if self.config.use_pretrained_language:
            # Use pretrained TinyLLM from HuggingFace
            self.language_model = create_language_backbone(
                use_pretrained=True,
                model_name=self.config.pretrained_language_model,
                freeze_base=self.config.freeze_language_base,
                unfreeze_last_layer=self.config.unfreeze_last_layer,
            )
            # Update config with actual model dimensions
            self.config.language_hidden_size = self.language_model.config.hidden_size
            self.config.language_num_layers = self.language_model.config.num_hidden_layers
            self.config.language_num_heads = self.language_model.config.num_attention_heads
            self.config.language_vocab_size = self.language_model.config.vocab_size
        else:
            # Create from scratch with custom config
            llm_config = TinyLLMConfig(
                vocab_size=self.config.language_vocab_size,
                hidden_size=self.config.language_hidden_size,
                num_hidden_layers=self.config.language_num_layers,
                num_attention_heads=self.config.language_num_heads,
                max_position_embeddings=self.config.language_max_length,
                hidden_dropout_prob=self.config.dropout,
                attention_probs_dropout_prob=self.config.dropout,
                use_qk_norm=self.config.use_qk_norm,
            )

            self.language_model = TinyLLMBackbone(
                config=llm_config,
                freeze_base=self.config.freeze_language_base,
                unfreeze_last_layer=self.config.unfreeze_last_layer,
            )

    def _build_fusion_module(self):
        """Initialize fusion module."""
        self.fusion_module = FusionModule(
            vision_dim=self.config.vision_output_dim,
            language_dim=self.config.language_hidden_size,
            bottleneck_dim=self.config.fusion_bottleneck_dim,
            num_visual_tokens=self.config.num_visual_tokens,
            dropout=self.config.fusion_dropout,
            use_qk_norm=self.config.use_qk_norm,
        )

    def _build_reasoning_module(self):
        """Initialize reasoning module."""
        self.reasoning_module = ReasoningModule(
            input_dim=self.config.language_hidden_size,
            hidden_dim=self.config.reasoning_hidden_dim,
            num_reasoning_layers=self.config.reasoning_num_layers,
            num_reasoning_heads=self.config.reasoning_num_heads,
            num_reasoning_steps=self.config.num_reasoning_steps,
            num_robots=self.config.num_robots,
            max_plan_steps=self.config.max_plan_steps,
            dropout=self.config.dropout,
        )

    def encode_image(
        self,
        pixel_values: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode images to visual tokens.

        Args:
            pixel_values: Input images [B, C, H, W]

        Returns:
            Dictionary with visual tokens and pooled features
        """
        return self.vision_encoder(pixel_values)

    def fuse_features(
        self,
        visual_tokens: torch.Tensor,
        text_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Fuse visual tokens with language model space.

        Args:
            visual_tokens: Visual features [B, num_visual_tokens, vision_dim]
            text_embeds: Optional text embeddings for cross-attention

        Returns:
            Fused features in language model space
        """
        fusion_output = self.fusion_module(visual_tokens, text_embeds)
        return fusion_output['fused_features']

    def prepare_inputs_embeds(
        self,
        input_ids: torch.LongTensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_positions: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare input embeddings by merging text and visual tokens.

        Args:
            input_ids: Token IDs [B, seq_len]
            pixel_values: Optional images [B, C, H, W]
            image_positions: Positions where to insert image tokens [B]

        Returns:
            Tuple of (inputs_embeds, attention_mask)
        """
        batch_size, seq_len = input_ids.size()
        device = input_ids.device

        # Get text embeddings
        text_embeds = self.language_model.embed_tokens(input_ids)

        if pixel_values is None:
            # No images, return text embeddings directly
            attention_mask = (input_ids != self.language_model.config.pad_token_id).long()
            return text_embeds, attention_mask

        # Encode images
        vision_output = self.encode_image(pixel_values)
        visual_tokens = vision_output['visual_tokens']

        # Fuse visual tokens
        fused_visual = self.fuse_features(visual_tokens)
        num_visual = fused_visual.size(1)

        # Determine image positions
        if image_positions is None:
            # Default: insert at beginning
            image_positions = torch.zeros(batch_size, dtype=torch.long, device=device)

        # Create merged embeddings
        total_len = seq_len + num_visual
        inputs_embeds = torch.zeros(
            batch_size, total_len, self.config.language_hidden_size,
            dtype=text_embeds.dtype, device=device
        )
        attention_mask = torch.zeros(batch_size, total_len, dtype=torch.long, device=device)

        for i in range(batch_size):
            pos = image_positions[i].item()

            # Insert visual tokens
            inputs_embeds[i, pos:pos + num_visual] = fused_visual[i]
            attention_mask[i, pos:pos + num_visual] = 1

            # Insert text before image
            if pos > 0:
                inputs_embeds[i, :pos] = text_embeds[i, :pos]
                attention_mask[i, :pos] = (input_ids[i, :pos] != self.language_model.config.pad_token_id).long()

            # Insert text after image
            remaining = seq_len - pos
            inputs_embeds[i, pos + num_visual:pos + num_visual + remaining] = text_embeds[i, pos:]
            text_mask = (input_ids[i, pos:] != self.language_model.config.pad_token_id).long()
            attention_mask[i, pos + num_visual:pos + num_visual + remaining] = text_mask

        return inputs_embeds, attention_mask

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        image_positions: Optional[torch.LongTensor] = None,
        robot_targets: Optional[torch.LongTensor] = None,
        action_targets: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_reasoning: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through EmberVLM.

        Args:
            input_ids: Token IDs [B, seq_len]
            pixel_values: Images [B, C, H, W]
            attention_mask: Attention mask [B, seq_len]
            labels: Target token IDs for LM loss
            image_positions: Where to insert image tokens
            robot_targets: Target robot indices for robot selection loss
            action_targets: Target action embeddings for action planning loss
            use_cache: Whether to cache key/values
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return hidden states
            return_reasoning: Whether to return reasoning outputs

        Returns:
            Dictionary containing model outputs and losses
        """
        # Prepare inputs
        inputs_embeds, input_attention_mask = self.prepare_inputs_embeds(
            input_ids, pixel_values, image_positions
        )

        if attention_mask is not None:
            # Extend attention mask to include visual tokens
            num_visual = self.config.num_visual_tokens if pixel_values is not None else 0
            if attention_mask.size(1) < inputs_embeds.size(1):
                visual_mask = torch.ones(
                    attention_mask.size(0), num_visual,
                    dtype=attention_mask.dtype, device=attention_mask.device
                )
                attention_mask = torch.cat([visual_mask, attention_mask], dim=1)
        else:
            attention_mask = input_attention_mask

        # Adjust labels to match inputs_embeds length
        adjusted_labels = None
        if labels is not None:
            if labels.size(1) != inputs_embeds.size(1):
                # Labels need to be extended to match inputs_embeds
                batch_size = labels.size(0)
                num_visual = inputs_embeds.size(1) - labels.size(1)
                device = labels.device

                # Determine image positions
                if image_positions is None:
                    image_positions = torch.zeros(batch_size, dtype=torch.long, device=device)

                # Create adjusted labels with -100 at visual token positions
                adjusted_labels = torch.full(
                    (batch_size, inputs_embeds.size(1)),
                    -100,
                    dtype=labels.dtype,
                    device=device
                )

                for i in range(batch_size):
                    pos = image_positions[i].item()
                    # Copy labels before image position
                    if pos > 0:
                        adjusted_labels[i, :pos] = labels[i, :pos]
                    # Visual tokens get -100 (already set)
                    # Copy labels after image position
                    remaining = labels.size(1) - pos
                    adjusted_labels[i, pos + num_visual:pos + num_visual + remaining] = labels[i, pos:]
            else:
                adjusted_labels = labels

        # Forward through language model
        lm_outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=adjusted_labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        outputs = {
            'logits': lm_outputs['logits'],
            'loss': lm_outputs.get('loss'),
            'hidden_states': lm_outputs.get('hidden_states'),
            'attentions': lm_outputs.get('attentions'),
            'past_key_values': lm_outputs.get('past_key_values'),
        }

        # Reasoning module
        if self.config.reasoning_enabled and (return_reasoning or robot_targets is not None):
            hidden_states = lm_outputs['last_hidden_state']

            reasoning_outputs = self.reasoning_module(
                hidden_states,
                attention_mask=attention_mask,
                generate_reasoning=return_reasoning,
                select_robot=True,
                plan_actions=True,
            )

            outputs.update({
                'robot_logits': reasoning_outputs.get('robot_logits'),
                'robot_probs': reasoning_outputs.get('robot_probs'),
                'robot_confidence': reasoning_outputs.get('robot_confidence'),
                'plan_steps': reasoning_outputs.get('plan_steps'),
                'plan_coherence': reasoning_outputs.get('plan_coherence'),
            })

            if return_reasoning and 'reasoning_chain' in reasoning_outputs:
                outputs['reasoning_chain'] = reasoning_outputs['reasoning_chain']

            # Compute reasoning losses
            if robot_targets is not None or action_targets is not None:
                targets = {}
                if robot_targets is not None:
                    targets['robot_target'] = robot_targets
                if action_targets is not None:
                    targets['action_target'] = action_targets

                reasoning_losses = self.reasoning_loss(reasoning_outputs, targets)

                if outputs['loss'] is not None:
                    outputs['loss'] = outputs['loss'] + reasoning_losses['total_loss']
                else:
                    outputs['loss'] = reasoning_losses['total_loss']

                outputs['reasoning_losses'] = reasoning_losses

        return outputs

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_positions: Optional[torch.LongTensor] = None,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> torch.LongTensor:
        """
        Generate text given image and optional prompt.

        Args:
            input_ids: Optional prompt token IDs
            pixel_values: Input images
            attention_mask: Attention mask
            image_positions: Where images are in sequence
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling threshold
            do_sample: Whether to sample or use greedy

        Returns:
            Generated token IDs
        """
        if input_ids is None and pixel_values is None:
            raise ValueError("Must provide either input_ids or pixel_values")

        # Prepare inputs
        if input_ids is not None:
            inputs_embeds, attention_mask = self.prepare_inputs_embeds(
                input_ids, pixel_values, image_positions
            )
        else:
            # Image only - create embeddings from image
            vision_output = self.encode_image(pixel_values)
            visual_tokens = vision_output['visual_tokens']
            inputs_embeds = self.fuse_features(visual_tokens)
            attention_mask = torch.ones(
                inputs_embeds.size(0), inputs_embeds.size(1),
                dtype=torch.long, device=inputs_embeds.device
            )

        # Generate
        generated = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
        )

        return generated

    @torch.no_grad()
    def analyze_incident(
        self,
        pixel_values: torch.Tensor,
        instruction: Optional[str] = None,
        tokenizer: Any = None,
    ) -> Dict[str, Any]:
        """
        Analyze an incident image and select appropriate robot.

        Args:
            pixel_values: Input image [1, C, H, W]
            instruction: Optional instruction text
            tokenizer: Tokenizer for encoding instruction

        Returns:
            Dictionary with robot selection, confidence, and action plan
        """
        device = pixel_values.device

        # Prepare inputs
        if instruction is not None and tokenizer is not None:
            tokens = tokenizer(instruction, return_tensors="pt", padding=True)
            input_ids = tokens['input_ids'].to(device)
            attention_mask = tokens['attention_mask'].to(device)
        else:
            input_ids = None
            attention_mask = None

        # Forward pass with reasoning
        outputs = self.forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            return_reasoning=True,
        )

        # Get robot selection
        robot_probs = outputs['robot_probs']
        robot_idx = robot_probs.argmax(dim=-1).item()
        confidence = outputs['robot_confidence'].item()

        result = {
            'selected_robot': self.config.robot_names[robot_idx],
            'robot_index': robot_idx,
            'confidence': confidence,
            'robot_probabilities': {
                name: prob.item()
                for name, prob in zip(self.config.robot_names, robot_probs[0])
            },
        }

        if 'plan_coherence' in outputs:
            result['plan_coherence'] = outputs['plan_coherence'].item()

        if 'reasoning_chain' in outputs:
            result['reasoning_chain'] = outputs['reasoning_chain']

        return result

    def count_parameters(self) -> Dict[str, int]:
        """Count model parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # Per component
        vision_params = sum(p.numel() for p in self.vision_encoder.parameters())
        language_params = sum(p.numel() for p in self.language_model.parameters())
        fusion_params = sum(p.numel() for p in self.fusion_module.parameters())

        result = {
            'total': total,
            'trainable': trainable,
            'vision_encoder': vision_params,
            'language_model': language_params,
            'fusion_module': fusion_params,
        }

        if self.config.reasoning_enabled:
            reasoning_params = sum(p.numel() for p in self.reasoning_module.parameters())
            result['reasoning_module'] = reasoning_params

        return result

    def save_pretrained(self, save_directory: str):
        """Save model to directory."""
        import os
        import json

        os.makedirs(save_directory, exist_ok=True)

        # Save config
        config_path = os.path.join(save_directory, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)

        # Save model weights
        model_path = os.path.join(save_directory, 'pytorch_model.bin')
        torch.save(self.state_dict(), model_path)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_path: str,
        config: Optional[EmberVLMConfig] = None,
    ) -> 'EmberVLM':
        """Load model from directory."""
        import os
        import json

        # Load config
        config_path = os.path.join(pretrained_path, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            config = EmberVLMConfig.from_dict(config_dict)
        elif config is None:
            config = EmberVLMConfig()

        # Create model
        model = cls(config)

        # Load weights
        model_path = os.path.join(pretrained_path, 'pytorch_model.bin')
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)

        return model

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype


# Convenience functions
def create_embervlm(
    config: Optional[EmberVLMConfig] = None,
    **kwargs,
) -> EmberVLM:
    """
    Create EmberVLM model with optional configuration overrides.

    Args:
        config: Optional base configuration
        **kwargs: Configuration overrides

    Returns:
        EmberVLM model instance
    """
    if config is None:
        config = EmberVLMConfig()

    # Apply overrides
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return EmberVLM(config)


def load_embervlm(
    path_or_repo: str,
    device: str = 'cuda',
    dtype: torch.dtype = torch.float16,
) -> EmberVLM:
    """
    Load EmberVLM model from local path or HuggingFace Hub.

    Args:
        path_or_repo: Local path or HuggingFace repo ID
        device: Device to load model on
        dtype: Data type for model weights

    Returns:
        Loaded EmberVLM model
    """
    import os

    if os.path.exists(path_or_repo):
        model = EmberVLM.from_pretrained(path_or_repo)
    else:
        # Try loading from HuggingFace Hub
        from huggingface_hub import snapshot_download
        local_path = snapshot_download(repo_id=path_or_repo)
        model = EmberVLM.from_pretrained(local_path)

    model = model.to(device=device, dtype=dtype)
    model.eval()

    return model

