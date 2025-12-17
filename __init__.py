"""
EmberVLM: Ultra-efficient Multimodal Vision-Language Model
for Robot Fleet Reasoning on Edge Devices
"""

__version__ = "0.1.0"
__author__ = "EmberVLM Team"

from .models import EmberVLM, EmberVLMConfig, create_embervlm
from .models import VisionEncoderWrapper, LanguageModelWrapper, MultimodalFusionModule

__all__ = [
    'EmberVLM',
    'EmberVLMConfig',
    'create_embervlm',
    'VisionEncoderWrapper',
    'LanguageModelWrapper',
    'MultimodalFusionModule',
]

