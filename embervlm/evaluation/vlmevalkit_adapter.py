"""
EmberVLM Model Adapter for VLMEvalKit

This module provides VLMEvalKit-compatible wrappers for EmberVLM,
enabling standardized evaluation on common VLM benchmarks.
"""

import torch
import os.path as osp
import warnings
import logging
from typing import List, Optional, Dict, Any, Union
from PIL import Image

logger = logging.getLogger(__name__)

# Try importing VLMEvalKit base
try:
    from vlmeval.vlm.base import BaseModel
    from vlmeval.smp import splitlen
    HAS_VLMEVAL = True
except ImportError:
    HAS_VLMEVAL = False
    BaseModel = object
    logger.warning("VLMEvalKit not found. EmberVLM adapter will be limited.")


class EmberVLM_VLMEval(BaseModel):
    """
    EmberVLM adapter for VLMEvalKit evaluation framework.

    Supports evaluation on standard VLM benchmarks including:
    - MMBench, MME, MMMU
    - MathVista, ChartQA
    - DocVQA, TextVQA, OCRBench
    - AI2D, ScienceQA, SEEDBench
    - HallusionBench, MMVet

    Usage:
        model = EmberVLM_VLMEval(model_path="path/to/checkpoint")
        result = model.generate(['image.jpg', 'What is in this image?'])
    """

    INSTALL_REQ = True
    INTERLEAVE = True  # Supports interleaved image-text

    def __init__(
        self,
        model_path: str,
        tokenizer_path: Optional[str] = None,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16,
        max_new_tokens: int = 512,
        **kwargs,
    ):
        """
        Initialize EmberVLM for VLMEvalKit.

        Args:
            model_path: Path to EmberVLM checkpoint directory
            tokenizer_path: Path to tokenizer (defaults to model_path)
            device: Device to run on
            torch_dtype: Model dtype
            max_new_tokens: Maximum tokens to generate
            **kwargs: Additional generation arguments
        """
        if not HAS_VLMEVAL:
            raise ImportError("VLMEvalKit is required. Install with: pip install vlmeval")

        super().__init__()

        self.model_path = model_path
        self.device = device
        self.torch_dtype = torch_dtype

        # Generation config
        self.generation_config = {
            'max_new_tokens': max_new_tokens,
            'do_sample': False,
            'temperature': 1.0,
            'top_p': 1.0,
        }
        self.generation_config.update(kwargs)

        # Load model and tokenizer
        self._load_model(model_path, tokenizer_path)

        logger.info(f"EmberVLM loaded from {model_path}")

    def _load_model(self, model_path: str, tokenizer_path: Optional[str] = None):
        """Load EmberVLM model and tokenizer."""
        import sys

        # Add EmberVLM to path if needed
        embervlm_root = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
        if embervlm_root not in sys.path:
            sys.path.insert(0, embervlm_root)

        try:
            from embervlm.models import EmberVLM, EmberVLMConfig
            from transformers import AutoTokenizer

            # Load tokenizer
            tokenizer_path = tokenizer_path or osp.join(model_path, 'tokenizer')
            if osp.exists(tokenizer_path):
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            else:
                # Fallback to GPT-2 tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
                logger.warning("Using fallback GPT-2 tokenizer")

            # Ensure pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Add special tokens if not present
            special_tokens = ['<|reasoning_start|>', '<|reasoning_end|>',
                            '<|robot_selection|>', '<|action_plan|>', '<|image|>']
            existing = self.tokenizer.additional_special_tokens or []
            new_tokens = [t for t in special_tokens if t not in existing]
            if new_tokens:
                self.tokenizer.add_special_tokens({'additional_special_tokens': existing + new_tokens})

            # Load model
            config_path = osp.join(model_path, 'config.json')
            if osp.exists(config_path):
                import json
                with open(config_path, 'r') as f:
                    config_dict = json.load(f)
                config = EmberVLMConfig.from_dict(config_dict)
            else:
                config = EmberVLMConfig()

            self.model = EmberVLM(config)

            # Load weights
            weights_path = osp.join(model_path, 'pytorch_model.bin')
            if not osp.exists(weights_path):
                weights_path = osp.join(model_path, 'model.safetensors')

            if osp.exists(weights_path):
                if weights_path.endswith('.safetensors'):
                    from safetensors.torch import load_file
                    state_dict = load_file(weights_path)
                else:
                    state_dict = torch.load(weights_path, map_location='cpu')
                self.model.load_state_dict(state_dict, strict=False)
            else:
                logger.warning(f"No weights found at {model_path}")

            # Resize embeddings if needed
            if len(self.tokenizer) != self.model.language_model.get_input_embeddings().weight.shape[0]:
                self.model.language_model.resize_token_embeddings(len(self.tokenizer))

            # Move to device
            self.model = self.model.to(self.device, dtype=self.torch_dtype)
            self.model.eval()

            # Get image preprocessor
            self.image_preprocessor = self.model.image_preprocessor

        except Exception as e:
            logger.error(f"Failed to load EmberVLM: {e}")
            raise

    def generate_inner(self, message, dataset=None):
        """
        Generate response for VLMEvalKit evaluation.

        Args:
            message: List of dicts with 'type' (image/text) and 'value'
            dataset: Dataset name for custom prompting

        Returns:
            Generated text response
        """
        # Route to appropriate prompt builder
        if dataset in ['MMBench_DEV_EN', 'MMBench_TEST_EN', 'MMBench_DEV_CN',
                       'MMBench_TEST_CN', 'MMBench', 'MMBench_CN',
                       'MMBench_DEV_EN_V11', 'MMBench_DEV_CN_V11',
                       'MMBench_TEST_EN_V11', 'MMBench_TEST_CN_V11']:
            prompt, images = self.build_prompt_mmbench(message)
        elif dataset in ['MMMU_DEV_VAL', 'MMMU_TEST']:
            prompt, images = self.build_prompt_mmmu(message)
        elif dataset in ['MathVista_MINI']:
            prompt, images = self.build_prompt_mathvista(message)
        elif dataset in ['ChartQA_TEST']:
            prompt, images = self.build_prompt_chartqa(message)
        elif dataset in ['DocVQA_VAL', 'DocVQA_TEST']:
            prompt, images = self.build_prompt_docvqa(message)
        elif dataset in ['TextVQA_VAL', 'TextVQA_TEST']:
            prompt, images = self.build_prompt_textvqa(message)
        elif dataset in ['MME', 'MMVet', 'OCRVQA_TEST', 'OCRVQA_TESTCORE',
                        'InfoVQA_VAL', 'InfoVQA_TEST', 'OCRBench']:
            prompt, images = self.build_prompt_default(message, add_brief=True)
        elif dataset == 'HallusionBench':
            prompt, images = self.build_prompt_default(message, add_yes_or_no=True)
        elif dataset in ['MMStar', 'SEEDBench_IMG', 'AI2D_TEST',
                        'ScienceQA_VAL', 'ScienceQA_TEST']:
            prompt, images = self.build_prompt_puremcq(message)
        else:
            prompt, images = self.build_prompt_default(message)

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(self.device)

        # Process images
        if images:
            if isinstance(images, list):
                pixel_values = torch.stack([
                    self.image_preprocessor(img) for img in images
                ]).to(self.device, dtype=self.torch_dtype)
            else:
                pixel_values = self.image_preprocessor(images).unsqueeze(0).to(
                    self.device, dtype=self.torch_dtype
                )
        else:
            pixel_values = None

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                pixel_values=pixel_values,
                **self.generation_config,
            )

        # Decode
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].size(1):],
            skip_special_tokens=True,
        )

        return generated_text.strip()

    def build_prompt_default(self, message, add_brief=False, add_yes_or_no=False):
        """Build default prompt for EmberVLM."""
        from transformers.image_utils import load_image

        prompt_parts = []
        images = []

        for msg in message:
            if msg['type'] == 'image':
                img = load_image(msg['value'])
                images.append(img)
                prompt_parts.append('<|image|>')
            elif msg['type'] == 'text':
                prompt_parts.append(msg['value'].strip())

        prompt = ' '.join(prompt_parts)

        if add_brief:
            prompt += '\nGive a very brief answer.'
        if add_yes_or_no:
            prompt += '\nAnswer yes or no.'

        prompt += '\nAnswer:'

        return prompt, images if images else None

    def build_prompt_mmbench(self, message):
        """Build prompt for MMBench format."""
        from transformers.image_utils import load_image

        prompt_parts = []
        images = []

        for msg in message:
            if msg['type'] == 'image':
                img = load_image(msg['value'])
                images.append(img)
                prompt_parts.append('<|image|>')
            elif msg['type'] == 'text':
                text = msg['value'].strip()
                # Adapt options format
                text = text.replace('\nOptions:', '\nChoices:')
                text = text.replace(
                    'Please select the correct answer from the options above.',
                    'Answer with the letter.'
                )
                prompt_parts.append(text)

        prompt = ' '.join(prompt_parts)
        prompt += '\nAnswer:'

        return prompt, images if images else None

    def build_prompt_mmmu(self, message):
        """Build prompt for MMMU format."""
        from transformers.image_utils import load_image

        prompt_parts = []
        images = []

        for msg in message:
            if msg['type'] == 'image':
                img = load_image(msg['value'])
                images.append(img)
                prompt_parts.append(f'<|image|>')
            elif msg['type'] == 'text':
                text = msg['value'].strip()
                prompt_parts.append(text)

        prompt = ' '.join(prompt_parts)
        prompt += '\nProvide a brief answer.'

        return prompt, images if images else None

    def build_prompt_mathvista(self, message):
        """Build prompt for MathVista format."""
        from transformers.image_utils import load_image

        prompt_parts = []
        images = []

        for msg in message:
            if msg['type'] == 'image':
                img = load_image(msg['value'])
                images.append(img)
                prompt_parts.append('<|image|>')
            elif msg['type'] == 'text':
                prompt_parts.append(msg['value'].strip())

        prompt = ' '.join(prompt_parts)
        prompt += '\nThink step by step and provide your answer.'

        return prompt, images if images else None

    def build_prompt_chartqa(self, message):
        """Build prompt for ChartQA format."""
        from transformers.image_utils import load_image

        prompt_parts = []
        images = []

        for msg in message:
            if msg['type'] == 'image':
                img = load_image(msg['value'])
                images.append(img)
                prompt_parts.append('<|image|>')
            elif msg['type'] == 'text':
                prompt_parts.append(msg['value'].strip())

        prompt = ' '.join(prompt_parts)
        prompt += '\nAnswer briefly.'

        return prompt, images if images else None

    def build_prompt_docvqa(self, message):
        """Build prompt for DocVQA format."""
        from transformers.image_utils import load_image

        prompt_parts = []
        images = []

        for msg in message:
            if msg['type'] == 'image':
                img = load_image(msg['value'])
                images.append(img)
                prompt_parts.append('<|image|>')
            elif msg['type'] == 'text':
                prompt_parts.append(msg['value'].strip())

        prompt = ' '.join(prompt_parts)
        prompt += '\nAnswer the question based on the document. Be concise.'

        return prompt, images if images else None

    def build_prompt_textvqa(self, message):
        """Build prompt for TextVQA format."""
        from transformers.image_utils import load_image

        prompt_parts = []
        images = []

        for msg in message:
            if msg['type'] == 'image':
                img = load_image(msg['value'])
                images.append(img)
                prompt_parts.append('<|image|>')
            elif msg['type'] == 'text':
                prompt_parts.append(msg['value'].strip())

        prompt = ' '.join(prompt_parts)
        prompt += '\nRead the text in the image and answer. Be brief.'

        return prompt, images if images else None

    def build_prompt_puremcq(self, message):
        """Build prompt for pure MCQ benchmarks (MMStar, SEEDBench, etc.)."""
        from transformers.image_utils import load_image

        prompt_parts = []
        images = []

        for msg in message:
            if msg['type'] == 'image':
                img = load_image(msg['value'])
                images.append(img)
                prompt_parts.append('<|image|>')
            elif msg['type'] == 'text':
                text = msg['value'].strip()
                text = text.replace('\nOptions:', '\nChoices:')
                text = text.replace(
                    'Please select the correct answer from the options above.',
                    'Answer with the letter.'
                )
                prompt_parts.append(text)

        prompt = ' '.join(prompt_parts)
        prompt += '\nAnswer:'

        return prompt, images if images else None


def create_embervlm_config(
    model_path: str,
    stage: int = 4,
    **kwargs,
) -> Dict[str, Any]:
    """
    Create configuration dict for registering EmberVLM in VLMEvalKit.

    Args:
        model_path: Path to EmberVLM checkpoint
        stage: Training stage (1-4)
        **kwargs: Additional arguments

    Returns:
        Configuration dictionary for VLMEvalKit
    """
    return {
        'class': 'EmberVLM_VLMEval',
        'model_path': model_path,
        'stage': stage,
        **kwargs,
    }


# Benchmark suite definitions
STAGE1_BENCHMARKS = [
    'MME',
    'SEEDBench_IMG',
]

STAGE2_BENCHMARKS = [
    'MMBench_DEV_EN_V11',
    'TextVQA_VAL',
    'AI2D_TEST',
    'ScienceQA_VAL',
]

STAGE3_BENCHMARKS = STAGE2_BENCHMARKS + [
    'MMStar',
]

STAGE4_BENCHMARKS = [
    'MMBench_DEV_EN_V11',
    'MME',
    'MMMU_DEV_VAL',
    'MathVista_MINI',
    'ChartQA_TEST',
    'DocVQA_VAL',
    'TextVQA_VAL',
    'OCRBench',
    'AI2D_TEST',
    'ScienceQA_VAL',
    'SEEDBench_IMG',
    'MMStar',
    'HallusionBench',
    'MMVet',
]


def get_benchmarks_for_stage(stage: int) -> List[str]:
    """Get appropriate benchmarks for a training stage."""
    if stage == 1:
        return STAGE1_BENCHMARKS
    elif stage == 2:
        return STAGE2_BENCHMARKS
    elif stage == 3:
        return STAGE3_BENCHMARKS
    else:
        return STAGE4_BENCHMARKS

