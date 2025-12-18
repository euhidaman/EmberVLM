"""
EmberVLM Models Package

Provides model components for the EmberVLM multimodal VLM:
- Vision: RepViT-XXS (pretrained from HuggingFace)
- Language: TinyLLM-30M (pretrained from tinyllm/30M-0.4)
- Fusion: Adapter-based vision-language fusion
- Reasoning: Chain-of-Thought reasoning heads
"""

from embervlm.models.embervlm import EmberVLM, EmberVLMConfig
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

__all__ = [
    # Main model
    "EmberVLM",
    "EmberVLMConfig",
    # Vision
    "RepViTEncoder",
    "ImagePreprocessor",
    # Language
    "TinyLLMBackbone",
    "TinyLLMConfig",
    "PretrainedTinyLLMBackbone",
    "create_language_backbone",
    "PRETRAINED_TINYLLM_MODEL",
    # Fusion
    "FusionModule",
    # Reasoning
    "ReasoningModule",
    "ReasoningLoss",
]

