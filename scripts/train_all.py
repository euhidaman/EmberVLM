"""
EmberVLM Master Training Script

Orchestrates all four training stages:
1. Visual-Language Alignment
2. Multimodal Instruction Tuning
3. Robot Fleet Selection Training
4. Chain-of-Thought Reasoning Integration

MEMORY SAFE: Implements safeguards for distributed training on shared servers.
"""

import os
import argparse
import logging
import gc
from pathlib import Path
from typing import Optional, Dict, Any

import torch
from transformers import AutoTokenizer

from embervlm.models import (
    EmberVLM,
    EmberVLMConfig,
    BACKBONE_TINYLLM,
    BACKBONE_SMOLLM_135M,
    VISION_BACKBONE_REPVIT,
    VISION_BACKBONE_MOBILEVIT_XS,
)
from embervlm.training.train_utils import (
    TrainingConfig,
    setup_distributed,
    cleanup_distributed,
    is_main_process,
    set_seed,
)
from embervlm.training.stage1_align import run_stage1_training
from embervlm.training.stage2_instruct import run_stage2_training
from embervlm.training.stage3_incidents import run_stage3_training
from embervlm.training.stage4_reasoning import run_stage4_training
from embervlm.monitoring.wandb_logger import WandbLogger
from embervlm.monitoring.carbon_tracker import CarbonTracker

# Set environment variables for memory safety BEFORE any CUDA operations
# Prevent OpenMP thread explosion
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')  # Prevent MKL thread explosion
# Prevent tokenizer warnings
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Default pretrained language model
PRETRAINED_LANGUAGE_MODEL = "tinyllm/30M-0.4"


def find_latest_checkpoint(stage_dir: Path) -> Optional[Path]:
    """
    Find the latest checkpoint in a stage directory.

    Checkpoints are named like 'checkpoint-{step}' where step is an integer.
    Returns the checkpoint with the highest step number.

    Args:
        stage_dir: Path to the stage directory (e.g., outputs/stage2)

    Returns:
        Path to the latest checkpoint, or None if no checkpoints found
    """
    if not stage_dir.exists():
        return None

    checkpoints = []
    for item in stage_dir.iterdir():
        if item.is_dir() and item.name.startswith('checkpoint-'):
            try:
                step = int(item.name.split('-')[1])
                checkpoints.append((step, item))
            except (ValueError, IndexError):
                continue

    if not checkpoints:
        return None

    # Sort by step number and return the latest
    checkpoints.sort(key=lambda x: x[0], reverse=True)
    return checkpoints[0][1]


def get_previous_stage_checkpoint(output_dir: Path, current_stage: str) -> Optional[Path]:
    """
    Get the latest checkpoint from the previous training stage.

    Args:
        output_dir: Base output directory (e.g., ./outputs)
        current_stage: Current stage number as string ('2', '3', '4')

    Returns:
        Path to the latest checkpoint from the previous stage, or None
    """
    stage_num = int(current_stage)
    if stage_num <= 1:
        return None  # Stage 1 has no previous stage

    prev_stage = stage_num - 1
    prev_stage_dir = output_dir / f'stage{prev_stage}'

    checkpoint = find_latest_checkpoint(prev_stage_dir)
    if checkpoint:
        logger.info(
            f"Found latest checkpoint from Stage {prev_stage}: {checkpoint}")
    else:
        logger.warning(f"No checkpoint found in {prev_stage_dir}")

    return checkpoint


def create_model(config: Optional[EmberVLMConfig] = None) -> EmberVLM:
    """Create and initialize EmberVLM model."""
    if config is None:
        config = EmberVLMConfig()

    model = EmberVLM(config)

    # Print parameter info
    param_counts = model.count_parameters()
    logger.info(
        f"Model created with {param_counts['total']:,} total parameters")
    logger.info(f"Trainable parameters: {param_counts['trainable']:,}")
    logger.info(f"Vision backbone: {config.vision_backbone}")
    logger.info(f"Language backbone: {config.language_backbone}")

    return model


def get_tokenizer_model_name(language_backbone: str) -> str:
    """Get the appropriate tokenizer model name based on language backbone."""
    if language_backbone == BACKBONE_SMOLLM_135M:
        return "HuggingFaceTB/SmolLM-135M"
    else:
        return PRETRAINED_LANGUAGE_MODEL


def create_tokenizer(model_name: str = PRETRAINED_LANGUAGE_MODEL) -> AutoTokenizer:
    """
    Create and configure tokenizer.

    Uses the tokenizer from the pretrained language model.
    Supports TinyLLM (GPT-2 based) and SmolLM tokenizers.
    """
    try:
        # Try to load tokenizer from pretrained model
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True)
        logger.info(f"Loaded tokenizer from {model_name}")
    except Exception as e:
        # Fallback to GPT-2 tokenizer (compatible with tinyllm/30M-0.4)
        logger.warning(f"Could not load tokenizer from {model_name}: {e}")
        logger.info("Falling back to GPT-2 tokenizer")
        tokenizer = AutoTokenizer.from_pretrained('gpt2')

    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Add special tokens for EmberVLM
    special_tokens = {
        'additional_special_tokens': [
            '<|reasoning_start|>',
            '<|reasoning_end|>',
            '<|robot_selection|>',
            '<|action_plan|>',
            '<|image|>',
        ]
    }
    num_added = tokenizer.add_special_tokens(special_tokens)
    logger.info(f"Added {num_added} special tokens to tokenizer")

    return tokenizer


def generate_model_card(
    vision_backbone: str,
    language_backbone: str,
    total_params: int,
    trainable_params: int,
    carbon_emissions: float = None,
) -> str:
    """Generate model card for HuggingFace Hub."""

    # Model size category
    if total_params < 100_000_000:
        size_category = "Tiny (~35M parameters)"
        variant_name = "embervlm-tiny"
    else:
        size_category = "Small (~137M parameters)"
        variant_name = "embervlm-small"

    # Backbone descriptions
    vision_desc = {
        'repvit': 'RepViT-M0.9 (~5M params)',
        'mobilevit_xs': 'Apple MobileViT-XS (~2.3M params)'
    }.get(vision_backbone, vision_backbone)

    language_desc = {
        'tinyllm': 'TinyLLM-30M (30M params)',
        'smollm_135m': 'SmolLM-135M (135M params)'
    }.get(language_backbone, language_backbone)

    carbon_info = f"\n- **Carbon Emissions**: {carbon_emissions:.4f} kg CO2eq" if carbon_emissions else ""

    card = f"""---
language:
- en
license: apache-2.0
tags:
- vision-language
- multimodal
- robotics
- edge-deployment
- {vision_backbone}
- {language_backbone}
---

# {variant_name.upper()}: {size_category}

EmberVLM is an efficient vision-language model optimized for edge deployment and robotic applications.

## Model Details

- **Model Type**: Vision-Language Model (VLM)
- **Size**: {size_category}
- **Total Parameters**: {total_params:,}
- **Trainable Parameters**: {trainable_params:,}{carbon_info}

### Architecture

- **Vision Encoder**: {vision_desc}
- **Language Model**: {language_desc}
- **Training Stages**: 4-stage curriculum
  1. Visual-Language Alignment
  2. Multimodal Instruction Tuning
  3. Robot Fleet Selection
  4. Chain-of-Thought Reasoning

## Usage

```python
from embervlm import EmberVLM
from transformers import AutoTokenizer
from PIL import Image

# Load model and tokenizer
model = EmberVLM.from_pretrained("{variant_name}")
tokenizer = AutoTokenizer.from_pretrained("{variant_name}")

# Prepare input
image = Image.open("robot_scene.jpg")
prompt = "<image>What is happening in this scene?"

# Generate response
outputs = model.generate(image=image, prompt=prompt, tokenizer=tokenizer)
print(outputs)
```

## Training Configuration

- **Vision Backbone**: {vision_backbone}
- **Language Backbone**: {language_backbone}
- **Optimization**: AdamW with cosine learning rate schedule
- **Mixed Precision**: bfloat16
- **Stages Completed**: 1-4 (Full curriculum)

## Intended Use

- Edge deployment on resource-constrained devices
- Robotic vision-language understanding
- Real-time multimodal reasoning
- Robot fleet selection and task planning

## Limitations

- Optimized for efficiency over maximum accuracy
- Best suited for edge/mobile deployment scenarios
- Training focused on robot-centric scenarios

## Citation

```bibtex
@software{{embervlm_{variant_name.replace('-', '_')},
  title = {{EmberVLM-{size_category.split()[0]}}},
  author = {{EmberVLM Team}},
  year = {{2026}},
  url = {{https://huggingface.co/{variant_name}}}
}}
```
"""
    return card


def count_model_parameters(model) -> tuple:
    """Count total and trainable parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def push_to_hub(
    model,
    tokenizer,
    vision_backbone: str,
    language_backbone: str,
    hub_username: str,
    carbon_emissions: float = None,
):
    """Push model to HuggingFace Hub with automatic repo selection."""
    # Check if push is disabled
    if os.environ.get('DISABLE_HUB_PUSH', '').lower() in ('1', 'true', 'yes'):
        logger.info(
            "Hub push disabled via DISABLE_HUB_PUSH environment variable")
        return

    # Check for HF token
    hf_token = os.environ.get('HF_TOKEN')
    if not hf_token:
        logger.warning("HF_TOKEN not found in environment. Skipping hub push.")
        logger.warning(
            "To enable hub push, set HF_TOKEN environment variable with your HuggingFace token.")
        return

    # Determine repo name based on backbone
    total_params, trainable_params = count_model_parameters(model)

    if total_params < 100_000_000:
        repo_name = "embervlm-tiny"
    else:
        repo_name = "embervlm-small"

    repo_id = f"{hub_username}/{repo_name}"

    logger.info("="*60)
    logger.info(f"Pushing model to HuggingFace Hub: {repo_id}")
    logger.info(f"  Vision: {vision_backbone}")
    logger.info(f"  Language: {language_backbone}")
    logger.info(f"  Total params: {total_params:,}")
    logger.info("="*60)

    try:
        from huggingface_hub import HfApi, create_repo

        # Create repo if it doesn't exist
        try:
            create_repo(
                repo_id=repo_id,
                token=hf_token,
                private=False,
                exist_ok=True,
            )
            logger.info(f"✓ Repository created/verified: {repo_id}")
        except Exception as e:
            logger.warning(f"Could not create repo (may already exist): {e}")

        # Generate model card
        model_card = generate_model_card(
            vision_backbone=vision_backbone,
            language_backbone=language_backbone,
            total_params=total_params,
            trainable_params=trainable_params,
            carbon_emissions=carbon_emissions,
        )

        # Push model
        logger.info("Uploading model weights...")
        model.push_to_hub(
            repo_id=repo_id,
            token=hf_token,
            commit_message=f"Upload {repo_name} ({vision_backbone}+{language_backbone})",
        )
        logger.info("✓ Model weights uploaded")

        # Push tokenizer
        logger.info("Uploading tokenizer...")
        tokenizer.push_to_hub(
            repo_id=repo_id,
            token=hf_token,
            commit_message=f"Upload tokenizer for {repo_name}",
        )
        logger.info("✓ Tokenizer uploaded")

        # Push model card
        logger.info("Uploading model card...")
        api = HfApi()
        api.upload_file(
            path_or_fileobj=model_card.encode('utf-8'),
            path_in_repo="README.md",
            repo_id=repo_id,
            token=hf_token,
            commit_message=f"Add model card for {repo_name}",
        )
        logger.info("✓ Model card uploaded")

        logger.info("="*60)
        logger.info(
            f"✅ Successfully pushed to: https://huggingface.co/{repo_id}")
        logger.info("="*60)

    except ImportError:
        logger.warning(
            "huggingface_hub not installed. Install with: pip install huggingface_hub")
    except Exception as e:
        logger.error(f"Failed to push to hub: {e}")
        logger.warning("Training completed successfully, but hub push failed.")


def run_all_stages(args: argparse.Namespace):
    """Run all training stages."""

    # Map size to backbone configuration
    if args.size == 'tiny':
        vision_backbone = 'repvit'
        language_backbone = 'tinyllm'
        wandb_project = 'embervlm-tiny'
    else:  # small
        vision_backbone = 'mobilevit_xs'
        language_backbone = 'smollm_135m'
        wandb_project = 'embervlm-small'
    
    logger.info(f"Model size: {args.size}")
    logger.info(f"  Vision backbone: {vision_backbone}")
    logger.info(f"  Language backbone: {language_backbone}")
    logger.info(f"  W&B project: {wandb_project}")

    # Setup distributed training if enabled
    if args.distributed:
        rank, local_rank, world_size = setup_distributed()
        logger.info(
            f"Distributed training initialized: rank={rank}, local_rank={local_rank}, world_size={world_size}")
        device = torch.device(f'cuda:{local_rank}')
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Single process training on device: {device}")

    # Setup
    set_seed(args.seed)

    # Create backbone-specific output directory
    # This prevents different configurations from overwriting each other
    base_output_dir = Path(args.output_dir)
    backbone_suffix = f"{vision_backbone}_{language_backbone}"
    output_dir = base_output_dir / backbone_suffix
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("="*60)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Vision backbone: {vision_backbone}")
    logger.info(f"Language backbone: {language_backbone}")
    logger.info("="*60)

    # Create tokenizer based on language backbone
    logger.info("Creating tokenizer...")
    tokenizer_model_name = get_tokenizer_model_name(language_backbone)
    logger.info(f"Using tokenizer from: {tokenizer_model_name}")
    tokenizer = create_tokenizer(tokenizer_model_name)
    tokenizer.save_pretrained(output_dir / 'tokenizer')

    # Create model config with backbone selection
    logger.info("="*60)
    logger.info("Creating model config...")
    logger.info(f"  Vision backbone: {vision_backbone}")
    logger.info(f"  Language backbone: {language_backbone}")
    logger.info("="*60)
    config = EmberVLMConfig(
        vision_backbone=vision_backbone,
        language_backbone=language_backbone,
    )

    # Create model
    logger.info("Creating model...")
    model = create_model(config)

    # Resize embeddings for special tokens - MUST happen before any DDP wrapping
    # This ensures all ranks have the same embedding size
    target_vocab_size = len(tokenizer)
    logger.info(
        f"Target vocabulary size (with special tokens): {target_vocab_size}")

    # Resize embeddings
    if hasattr(model.language_model, 'resize_token_embeddings'):
        model.language_model.resize_token_embeddings(target_vocab_size)
        logger.info(f"Resized token embeddings to {target_vocab_size}")
    elif hasattr(model.language_model, 'model'):
        if hasattr(model.language_model.model, 'resize_token_embeddings'):
            model.language_model.model.resize_token_embeddings(
                target_vocab_size)
            logger.info(f"Resized token embeddings to {target_vocab_size}")

    # CRITICAL: Also update the model's internal config to reflect new vocab size
    # This ensures consistency throughout training
    if hasattr(model.language_model, 'config'):
        model.language_model.config.vocab_size = target_vocab_size
        logger.info(
            f"Updated language_model.config.vocab_size to {target_vocab_size}")
    if hasattr(model.language_model, 'model') and hasattr(model.language_model.model, 'config'):
        model.language_model.model.config.vocab_size = target_vocab_size
        logger.info(
            f"Updated language_model.model.config.vocab_size to {target_vocab_size}")

    # Verify embedding size after resize
    actual_vocab_size = None
    if hasattr(model.language_model, 'get_input_embeddings'):
        actual_vocab_size = model.language_model.get_input_embeddings(
        ).weight.shape[0]
    elif hasattr(model.language_model, 'model'):
        if hasattr(model.language_model.model, 'get_input_embeddings'):
            actual_vocab_size = model.language_model.model.get_input_embeddings(
            ).weight.shape[0]

    if actual_vocab_size is not None:
        if actual_vocab_size != target_vocab_size:
            raise RuntimeError(
                f"❌ Embedding resize failed! Actual: {actual_vocab_size}, Expected: {target_vocab_size}"
            )
        logger.info(f"✓ Verified embedding size: {actual_vocab_size}")

    # Load from checkpoint - either explicitly provided or auto-detected from previous stage
    checkpoint_path = None

    if args.resume_from_checkpoint:
        # User explicitly provided a checkpoint path
        checkpoint_path = Path(args.resume_from_checkpoint)
        if not checkpoint_path.exists():
            logger.warning(
                f"Specified checkpoint {checkpoint_path} does not exist")
            checkpoint_path = None
    elif args.stage != 'all' and args.stage != '1':
        # Running a specific stage (not 'all' or stage 1) - auto-detect previous stage checkpoint
        logger.info(
            f"Auto-detecting checkpoint from previous stage for Stage {args.stage}...")
        checkpoint_path = get_previous_stage_checkpoint(output_dir, args.stage)

    if checkpoint_path and checkpoint_path.exists():
        logger.info(f"Loading model from checkpoint: {checkpoint_path}")
        from embervlm.training.train_utils import load_checkpoint
        load_checkpoint(model, None, None, str(checkpoint_path))
        logger.info(f"✓ Loaded model weights from {checkpoint_path}")
    elif args.stage != 'all' and args.stage != '1':
        logger.warning(
            f"No checkpoint found for Stage {args.stage}. Starting from base model weights.")
        logger.warning(
            f"For best results, run previous stages first or provide --resume_from_checkpoint")

    # Synchronize all ranks after model initialization
    if args.distributed:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.barrier()
            logger.info("All ranks synchronized after model initialization")

    # Training configuration
    training_config = TrainingConfig(
        seed=args.seed,
        output_dir=str(output_dir),
        distributed=args.distributed,
        mixed_precision=args.mixed_precision,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation,
        save_steps=args.save_steps,
        log_steps=args.log_steps,
        push_to_hub=False,  # We handle hub push manually after training
        hub_model_id=None,
        wandb_project=wandb_project,  # Size-specific project name
    )

    # Carbon tracking - ONLY on rank 0 to prevent duplicate tracking
    carbon_tracker = None
    total_emissions = None
    if rank == 0:
        try:
            carbon_tracker = CarbonTracker(
                output_dir=str(output_dir / 'emissions'),
                project_name="EmberVLM",
            )
            carbon_tracker.start()
            logger.info("Carbon tracking started (rank 0 only)")
        except Exception as e:
            logger.warning(f"Failed to initialize carbon tracker: {e}")

    # Synchronize before training starts
    if args.distributed:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.barrier()

    try:
        # Stage 1: Visual-Language Alignment
        if args.stage in ['all', '1']:
            logger.info("="*60)
            logger.info("Stage 1: Visual-Language Alignment")
            logger.info("="*60)

            stage1_dict = training_config.to_dict()
            stage1_dict.update({
                'output_dir': str(output_dir / 'stage1'),
                'batch_size': 128,
                'num_training_steps': args.stage1_steps,
            })
            stage1_config = TrainingConfig(**stage1_dict)

            stage1_data_path = Path(
                args.stage1_data) if args.stage1_data else None
            if stage1_data_path and stage1_data_path.exists():
                # Prepare HF Hub repo ID
                hub_repo_id = None
                if args.hub_username:
                    repo_name = "embervlm-tiny" if args.size == 'tiny' else "embervlm-small"
                    hub_repo_id = f"{args.hub_username}/{repo_name}"
                
                run_stage1_training(
                    model=model,
                    config=stage1_config,
                    data_dir=str(stage1_data_path),
                    tokenizer=tokenizer,
                    num_epochs=args.stage1_epochs,
                    hub_repo_id=hub_repo_id,
                    vision_backbone=vision_backbone,
                    language_backbone=language_backbone,
                )
                # Unwrap model from DDP if needed
                from embervlm.training.train_utils import unwrap_model
                model = unwrap_model(model)
            else:
                logger.warning(f"Stage 1 data not provided, skipping...")

        # Clean up and sync before Stage 2
        if args.stage in ['all', '2']:
            import torch.distributed as dist
            if dist.is_initialized():
                dist.barrier()  # Synchronize all ranks
                torch.cuda.empty_cache()  # Clear CUDA cache

        # Stage 2: Instruction Tuning
        if args.stage in ['all', '2']:
            logger.info("="*60)
            logger.info("Stage 2: Multimodal Instruction Tuning")
            logger.info("="*60)

            stage2_dict = training_config.to_dict()
            stage2_dict.update({
                'output_dir': str(output_dir / 'stage2'),
                'batch_size': 64,
                'num_training_steps': args.stage2_steps,
                'find_unused_parameters': True,  # Reasoning heads not used in instruction tuning
            })
            stage2_config = TrainingConfig(**stage2_dict)

            stage2_data_path = Path(
                args.stage2_data) if args.stage2_data else None
            if stage2_data_path and stage2_data_path.exists():
                # Prepare HF Hub repo ID
                hub_repo_id = None
                if args.hub_username:
                    repo_name = "embervlm-tiny" if args.size == 'tiny' else "embervlm-small"
                    hub_repo_id = f"{args.hub_username}/{repo_name}"
                
                run_stage2_training(
                    model=model,
                    config=stage2_config,
                    data_dir=str(stage2_data_path),
                    tokenizer=tokenizer,
                    num_epochs=args.stage2_epochs,
                    hub_repo_id=hub_repo_id,
                    vision_backbone=vision_backbone,
                    language_backbone=language_backbone,
                )
                # Unwrap model from DDP if needed
                from embervlm.training.train_utils import unwrap_model
                model = unwrap_model(model)
            else:
                logger.warning(
                    f"Stage 2 data not found at {stage2_data_path}, skipping...")

        # Clean up and sync before Stage 3
        if args.stage in ['all', '3']:
            import torch.distributed as dist
            if dist.is_initialized():
                dist.barrier()  # Synchronize all ranks
                torch.cuda.empty_cache()  # Clear CUDA cache

        # Stage 3: Robot Selection Training
        if args.stage in ['all', '3']:
            logger.info("="*60)
            logger.info("Stage 3: Robot Fleet Selection Training")
            logger.info("="*60)

            # CRITICAL FIX: Verify embedding layer size matches tokenizer BEFORE Stage 3
            # This prevents index out of bounds errors when special tokens are used
            from embervlm.training.train_utils import unwrap_model
            model_unwrapped = unwrap_model(model)

            # Move model to CPU temporarily to safely resize embeddings
            device_before = next(model_unwrapped.parameters()).device
            model_unwrapped = model_unwrapped.cpu()
            torch.cuda.empty_cache()

            # Check embedding size vs tokenizer size
            current_vocab_size = None
            if hasattr(model_unwrapped.language_model, 'get_input_embeddings'):
                current_vocab_size = model_unwrapped.language_model.get_input_embeddings(
                ).weight.shape[0]
            elif hasattr(model_unwrapped.language_model, 'model'):
                if hasattr(model_unwrapped.language_model.model, 'get_input_embeddings'):
                    current_vocab_size = model_unwrapped.language_model.model.get_input_embeddings(
                    ).weight.shape[0]

            required_vocab_size = len(tokenizer)

            if current_vocab_size is not None and current_vocab_size != required_vocab_size:
                logger.warning(
                    f"⚠️ CRITICAL: Embedding size mismatch detected!")
                logger.warning(
                    f"   Current embedding size: {current_vocab_size}")
                logger.warning(
                    f"   Required (tokenizer size): {required_vocab_size}")
                logger.warning(
                    f"   Special tokens in use: {tokenizer.additional_special_tokens}")
                logger.info(f"🔧 Resizing embeddings to match tokenizer...")

                # Resize embeddings to match tokenizer
                try:
                    if hasattr(model_unwrapped.language_model, 'resize_token_embeddings'):
                        model_unwrapped.language_model.resize_token_embeddings(
                            required_vocab_size)
                        logger.info(
                            f"✓ Resized embeddings via resize_token_embeddings()")
                    elif hasattr(model_unwrapped.language_model, 'model'):
                        if hasattr(model_unwrapped.language_model.model, 'resize_token_embeddings'):
                            model_unwrapped.language_model.model.resize_token_embeddings(
                                required_vocab_size)
                            logger.info(
                                f"✓ Resized embeddings via model.resize_token_embeddings()")

                    # Also resize LM head if it exists
                    if hasattr(model_unwrapped.language_model, 'lm_head'):
                        old_lm_head = model_unwrapped.language_model.lm_head
                        new_lm_head = torch.nn.Linear(
                            old_lm_head.in_features,
                            required_vocab_size,
                            bias=old_lm_head.bias is not None
                        )
                        # Copy old weights
                        with torch.no_grad():
                            new_lm_head.weight[:current_vocab_size] = old_lm_head.weight
                            if old_lm_head.bias is not None:
                                new_lm_head.bias[:current_vocab_size] = old_lm_head.bias
                        model_unwrapped.language_model.lm_head = new_lm_head
                        logger.info(
                            f"✓ Resized LM head to {required_vocab_size}")
                    elif hasattr(model_unwrapped.language_model, 'model'):
                        if hasattr(model_unwrapped.language_model.model, 'lm_head'):
                            old_lm_head = model_unwrapped.language_model.model.lm_head
                            new_lm_head = torch.nn.Linear(
                                old_lm_head.in_features,
                                required_vocab_size,
                                bias=old_lm_head.bias is not None
                            )
                            # Copy old weights
                            with torch.no_grad():
                                new_lm_head.weight[:current_vocab_size] = old_lm_head.weight
                                if old_lm_head.bias is not None:
                                    new_lm_head.bias[:current_vocab_size] = old_lm_head.bias
                            model_unwrapped.language_model.model.lm_head = new_lm_head
                            logger.info(
                                f"✓ Resized LM head to {required_vocab_size}")

                except Exception as e:
                    logger.error(f"❌ Failed to resize embeddings: {e}")
                    raise RuntimeError(
                        f"Failed to resize embeddings to match tokenizer vocabulary: {e}")

                # Verify resize worked
                new_vocab_size = None
                if hasattr(model_unwrapped.language_model, 'get_input_embeddings'):
                    new_vocab_size = model_unwrapped.language_model.get_input_embeddings(
                    ).weight.shape[0]
                elif hasattr(model_unwrapped.language_model, 'model'):
                    if hasattr(model_unwrapped.language_model.model, 'get_input_embeddings'):
                        new_vocab_size = model_unwrapped.language_model.model.get_input_embeddings(
                        ).weight.shape[0]

                if new_vocab_size == required_vocab_size:
                    logger.info(
                        f"✅ Embedding resize successful: {new_vocab_size} tokens")
                else:
                    error_msg = f"❌ Embedding resize FAILED! Size is {new_vocab_size}, expected {required_vocab_size}"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
            else:
                logger.info(
                    f"✓ Embedding size matches tokenizer: {required_vocab_size} tokens")

            # Move model back to original device
            model_unwrapped = model_unwrapped.to(device_before)
            torch.cuda.synchronize()

            # Use the unwrapped model for Stage 3
            model = model_unwrapped

            stage3_dict = training_config.to_dict()
            stage3_dict.update({
                'output_dir': str(output_dir / 'stage3'),
                'batch_size': 32,
                'num_training_steps': args.stage3_steps,
                'find_unused_parameters': True,  # Reasoning heads may not be used yet
            })
            stage3_config = TrainingConfig(**stage3_dict)

            robot_dir = Path(args.robot_data) if args.robot_data else Path(
                Path(__file__).parent.parent / 'robot-selection-dataset')

            # Create robot selection data if needed
            if not robot_dir.exists():
                logger.info(f"Creating robot selection dataset at {robot_dir}")
                from embervlm.data.robot_loader import create_robot_selection_dataset
                create_robot_selection_dataset(str(robot_dir))

            if robot_dir.exists():
                # Prepare HF Hub repo ID
                hub_repo_id = None
                if args.hub_username:
                    repo_name = "embervlm-tiny" if args.size == 'tiny' else "embervlm-small"
                    hub_repo_id = f"{args.hub_username}/{repo_name}"
                
                run_stage3_training(
                    model=model,
                    config=stage3_config,
                    robot_data_dir=str(robot_dir),
                    tokenizer=tokenizer,
                    robot_epochs=args.stage3_robot_epochs,
                    hub_repo_id=hub_repo_id,
                    vision_backbone=vision_backbone,
                    language_backbone=language_backbone,
                )
                # Unwrap model from DDP if needed
                from embervlm.training.train_utils import unwrap_model
                model = unwrap_model(model)
            else:
                logger.warning(
                    f"Stage 3 robot data not found at {robot_dir}, skipping...")

        # Stage 4: Reasoning Integration
        if args.stage in ['all', '4']:
            logger.info("="*60)
            logger.info("Stage 4: Chain-of-Thought Reasoning Integration")
            logger.info("="*60)

            stage4_dict = training_config.to_dict()
            stage4_dict.update({
                'output_dir': str(output_dir / 'stage4'),
                'batch_size': 32,
                'num_training_steps': args.stage4_steps,
                'find_unused_parameters': True,  # May have conditional parameter usage
            })
            stage4_config = TrainingConfig(**stage4_dict)

            # Priority order for reasoning data:
            # 1. Explicitly provided reasoning_data path
            # 2. outputs/reasoning-data directory
            # 3. Fall back to robot_data (auto-generates reasoning chains)
            reasoning_dir = args.reasoning_data or str(
                output_dir / 'reasoning-data')

            data_dir_to_use = None
            data_source_type = None

            if Path(reasoning_dir).exists():
                data_dir_to_use = reasoning_dir
                data_source_type = "explicit reasoning data"
            elif args.robot_data and Path(args.robot_data).exists():
                # Fall back to robot selection data - ReasoningDataset will auto-generate chains
                data_dir_to_use = args.robot_data
                data_source_type = "robot selection data (auto-generating reasoning chains)"
                logger.info(
                    "No dedicated reasoning data found. Using robot selection data with auto-generated reasoning chains.")
            else:
                # Check default robot data location
                default_robot_dir = str(
                    output_dir.parent / 'robot-selection-dataset')
                if Path(default_robot_dir).exists():
                    data_dir_to_use = default_robot_dir
                    data_source_type = "robot selection data (auto-generating reasoning chains)"
                    logger.info(
                        "No dedicated reasoning data found. Using robot selection data with auto-generated reasoning chains.")

            if data_dir_to_use is not None:
                logger.info(f"Stage 4 data source: {data_source_type}")
                logger.info(f"Stage 4 data directory: {data_dir_to_use}")
                # Prepare HF Hub repo ID
                hub_repo_id = None
                if args.hub_username:
                    repo_name = "embervlm-tiny" if args.size == 'tiny' else "embervlm-small"
                    hub_repo_id = f"{args.hub_username}/{repo_name}"
                
                run_stage4_training(
                    model=model,
                    config=stage4_config,
                    data_dir=data_dir_to_use,
                    tokenizer=tokenizer,
                    phase1_epochs=args.stage4_phase1_epochs,
                    phase2_epochs=args.stage4_phase2_epochs,
                    hub_repo_id=hub_repo_id,
                    vision_backbone=vision_backbone,
                    language_backbone=language_backbone,
                )
                # Unwrap model from DDP if needed
                from embervlm.training.train_utils import unwrap_model
                model = unwrap_model(model)
            else:
                logger.warning(
                    "Stage 4: No reasoning or robot selection data found, skipping...")
                logger.warning("  To run Stage 4, provide either:")
                logger.warning(
                    "    --reasoning_data <path>  (explicit reasoning chains)")
                logger.warning(
                    "    --robot_data <path>      (will auto-generate reasoning chains)")

        # Unwrap model before saving (in case it's still wrapped)
        from embervlm.training.train_utils import unwrap_model
        model = unwrap_model(model)

        # Save final model
        logger.info("="*60)
        logger.info("Saving final model...")
        logger.info("="*60)

        final_output = output_dir / 'final'
        model.save_pretrained(str(final_output))
        tokenizer.save_pretrained(str(final_output))

        logger.info(f"Final model saved to {final_output}")

        # Stop carbon tracking and get emissions
        total_emissions = None
        if carbon_tracker is not None:
            try:
                total_emissions = carbon_tracker.stop()
                logger.info(
                    f"Total training emissions: {total_emissions:.4f} kg CO2eq")
            except Exception as e:
                logger.warning(f"Error stopping carbon tracker: {e}")

        # Push to HuggingFace Hub (only on rank 0)
        if rank == 0 and args.hub_username:
            push_to_hub(
                model=model,
                tokenizer=tokenizer,
                vision_backbone=vision_backbone,
                language_backbone=language_backbone,
                hub_username=args.hub_username,
                carbon_emissions=total_emissions,
            )

    finally:
        # Final carbon tracking cleanup (in case push_to_hub wasn't reached)
        if carbon_tracker is not None and total_emissions is None:
            try:
                total_emissions = carbon_tracker.stop()
                logger.info(
                    f"Total training emissions: {total_emissions:.4f} kg CO2eq")
            except Exception as e:
                logger.warning(f"Error stopping carbon tracker: {e}")

        # Cleanup distributed training
        if args.distributed:
            cleanup_distributed()
            logger.info("Distributed training cleanup completed")

    return model


def main():
    parser = argparse.ArgumentParser(description="EmberVLM Training")

    # General arguments
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--stage', type=str, default='all',
                        choices=['all', '1', '2', '3', '4'],
                        help='Training stage to run')

    # Model size selection
    parser.add_argument('--size', type=str, default='tiny',
                        choices=['tiny', 'small'],
                        help='Model size: tiny (~35M params, repvit+tinyllm) or small (~137M params, mobilevit_xs+smollm_135m)')

    # Distributed training
    parser.add_argument('--distributed', action='store_true',
                        help='Use distributed training')
    parser.add_argument('--mixed_precision', type=str, default='bf16',
                        choices=['fp32', 'fp16', 'bf16'],
                        help='Mixed precision training')

    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size per GPU')
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                        help='Learning rate')
    parser.add_argument('--gradient_accumulation', type=int, default=4,
                        help='Gradient accumulation steps')
    parser.add_argument('--save_steps', type=int, default=500,
                        help='Save checkpoint every N steps')
    parser.add_argument('--log_steps', type=int, default=50,
                        help='Log metrics every N steps')
    parser.add_argument('--eval_steps', type=int, default=500,
                        help='Evaluate every N steps')

    # Data paths
    parser.add_argument('--stage1_data', type=str, default='data/base_vlm',
                        help='Path to Stage 1 alignment data')
    parser.add_argument('--stage2_data', type=str, default='data/base_vlm/llava',
                        help='Path to Stage 2 instruction data')
    parser.add_argument('--robot_data', type=str, default='robot-selection-dataset',
                        help='Path to robot selection data')
    parser.add_argument('--reasoning_data', type=str, default=None,
                        help='Path to reasoning augmented data')

    # Stage-specific epochs
    parser.add_argument('--stage1_epochs', type=int, default=3)
    parser.add_argument('--stage1_steps', type=int, default=10000)
    parser.add_argument('--stage2_epochs', type=int, default=5)
    parser.add_argument('--stage2_steps', type=int, default=15000)
    parser.add_argument('--stage3_robot_epochs', type=int, default=20)
    parser.add_argument('--stage3_steps', type=int, default=20000)
    parser.add_argument('--stage4_phase1_epochs', type=int, default=5)
    parser.add_argument('--stage4_phase2_epochs', type=int, default=5)
    parser.add_argument('--stage4_steps', type=int, default=10000)

    # HuggingFace Hub (automatic push after training)
    parser.add_argument('--hub_username', type=str, default=None,
                        help='HuggingFace username/org for automatic model push. '
                        'Models push to {username}/embervlm-tiny (--size tiny) or '
                        '{username}/embervlm-small (--size small). '
                        'Requires HF_TOKEN environment variable. '
                        'Set DISABLE_HUB_PUSH=1 to skip pushing.')

    # Resume from checkpoint
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume from (e.g., outputs/stage2/checkpoint-789)')

    args = parser.parse_args()

    # Run training
    model = run_all_stages(args)

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
