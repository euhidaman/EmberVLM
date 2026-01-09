"""
EmberVLM Models Package

Provides model components for the EmberVLM multimodal VLM:
- Vision: RepViT-XXS (pretrained from HuggingFace) [default]
- Vision Alternative: MobileViT-XS from timm
- Language: TinyLLM-30M (pretrained from tinyllm/30M-0.4) [default]
- Language Alternative: SmolLM-135M from HuggingFace
- Fusion: Adapter-based vision-language fusion
- Reasoning: Chain-of-Thought reasoning heads
"""

from embervlm.models.embervlm import EmberVLM, EmberVLMConfig
from embervlm.models.vision_encoder import (
    RepViTEncoder,
    MobileViTEncoder,
    ImagePreprocessor,
    create_vision_encoder,
    VISION_BACKBONE_REPVIT,
    VISION_BACKBONE_MOBILEVIT_XS,
)
from embervlm.models.language_model import (
    TinyLLMBackbone,
    TinyLLMConfig,
    PretrainedTinyLLMBackbone,
    SmolLMBackbone,
    create_language_backbone,
    PRETRAINED_TINYLLM_MODEL,
    PRETRAINED_SMOLLM_135M,
    BACKBONE_TINYLLM,
    BACKBONE_SMOLLM_135M,
)
from embervlm.models.fusion_module import FusionModule
from embervlm.models.reasoning_heads import ReasoningModule, ReasoningLoss

__all__ = [
    # Main model
    "EmberVLM",
    "EmberVLMConfig",
    # Vision
    "RepViTEncoder",
    "MobileViTEncoder",
    "ImagePreprocessor",
    "create_vision_encoder",
    "VISION_BACKBONE_REPVIT",
    "VISION_BACKBONE_MOBILEVIT_XS",
    # Language
    "TinyLLMBackbone",
    "TinyLLMConfig",
    "PretrainedTinyLLMBackbone",
    "SmolLMBackbone",
    "create_language_backbone",
    "PRETRAINED_TINYLLM_MODEL",
    "PRETRAINED_SMOLLM_135M",
    "BACKBONE_TINYLLM",
    "BACKBONE_SMOLLM_135M",
    # Fusion
    "FusionModule",
    # Reasoning
    "ReasoningModule",
    "ReasoningLoss",
]
