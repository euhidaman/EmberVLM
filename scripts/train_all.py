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

from embervlm.models import EmberVLM, EmberVLMConfig
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
os.environ.setdefault('OMP_NUM_THREADS', '1')  # Prevent OpenMP thread explosion
os.environ.setdefault('MKL_NUM_THREADS', '1')  # Prevent MKL thread explosion
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')  # Prevent tokenizer warnings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Default pretrained language model
PRETRAINED_LANGUAGE_MODEL = "tinyllm/30M-0.4"


def create_model(config: Optional[EmberVLMConfig] = None) -> EmberVLM:
    """Create and initialize EmberVLM model."""
    if config is None:
        config = EmberVLMConfig()

    model = EmberVLM(config)

    # Print parameter info
    param_counts = model.count_parameters()
    logger.info(f"Model created with {param_counts['total']:,} total parameters")
    logger.info(f"Trainable parameters: {param_counts['trainable']:,}")

    return model


def create_tokenizer(model_name: str = PRETRAINED_LANGUAGE_MODEL) -> AutoTokenizer:
    """
    Create and configure tokenizer.

    Uses the tokenizer from the pretrained language model (tinyllm/30M-0.4)
    which is GPT-2 based with vocab_size=50257.
    """
    try:
        # Try to load tokenizer from pretrained model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
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


def run_all_stages(args: argparse.Namespace):
    """Run all training stages."""

    # Setup distributed training if enabled
    if args.distributed:
        rank, local_rank, world_size = setup_distributed()
        logger.info(f"Distributed training initialized: rank={rank}, local_rank={local_rank}, world_size={world_size}")
        device = torch.device(f'cuda:{local_rank}')
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Single process training on device: {device}")

    # Setup
    set_seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create tokenizer
    logger.info("Creating tokenizer...")
    tokenizer = create_tokenizer()
    tokenizer.save_pretrained(output_dir / 'tokenizer')

    # Create model
    logger.info("Creating model...")
    logger.info("Loading pretrained language model from tinyllm/30M-0.4...")
    model = create_model()

    # Resize embeddings for special tokens
    # Handle both PretrainedTinyLLMBackbone and TinyLLMBackbone
    if hasattr(model.language_model, 'resize_token_embeddings'):
        # PretrainedTinyLLMBackbone has this method
        model.language_model.resize_token_embeddings(len(tokenizer))
        logger.info(f"Resized token embeddings to {len(tokenizer)}")
    elif hasattr(model.language_model, 'model'):
        # TinyLLMBackbone wraps model
        if hasattr(model.language_model.model, 'resize_token_embeddings'):
            model.language_model.model.resize_token_embeddings(len(tokenizer))
            logger.info(f"Resized token embeddings to {len(tokenizer)}")

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
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
    )

    # Carbon tracking - ONLY on rank 0 to prevent duplicate tracking
    carbon_tracker = None
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

            stage1_data_path = Path(args.stage1_data) if args.stage1_data else None
            if stage1_data_path and stage1_data_path.exists():
                run_stage1_training(
                    model=model,
                    config=stage1_config,
                    data_dir=str(stage1_data_path),
                    tokenizer=tokenizer,
                    num_epochs=args.stage1_epochs,
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

            stage2_data_path = Path(args.stage2_data) if args.stage2_data else None
            if stage2_data_path and stage2_data_path.exists():
                run_stage2_training(
                    model=model,
                    config=stage2_config,
                    data_dir=str(stage2_data_path),
                    tokenizer=tokenizer,
                    num_epochs=args.stage2_epochs,
                )
                # Unwrap model from DDP if needed
                from embervlm.training.train_utils import unwrap_model
                model = unwrap_model(model)
            else:
                logger.warning(f"Stage 2 data not found at {stage2_data_path}, skipping...")

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
                current_vocab_size = model_unwrapped.language_model.get_input_embeddings().weight.shape[0]
            elif hasattr(model_unwrapped.language_model, 'model'):
                if hasattr(model_unwrapped.language_model.model, 'get_input_embeddings'):
                    current_vocab_size = model_unwrapped.language_model.model.get_input_embeddings().weight.shape[0]

            required_vocab_size = len(tokenizer)

            if current_vocab_size is not None and current_vocab_size != required_vocab_size:
                logger.warning(f"⚠️ CRITICAL: Embedding size mismatch detected!")
                logger.warning(f"   Current embedding size: {current_vocab_size}")
                logger.warning(f"   Required (tokenizer size): {required_vocab_size}")
                logger.warning(f"   Special tokens in use: {tokenizer.additional_special_tokens}")
                logger.info(f"🔧 Resizing embeddings to match tokenizer...")

                # Resize embeddings to match tokenizer
                try:
                    if hasattr(model_unwrapped.language_model, 'resize_token_embeddings'):
                        model_unwrapped.language_model.resize_token_embeddings(required_vocab_size)
                        logger.info(f"✓ Resized embeddings via resize_token_embeddings()")
                    elif hasattr(model_unwrapped.language_model, 'model'):
                        if hasattr(model_unwrapped.language_model.model, 'resize_token_embeddings'):
                            model_unwrapped.language_model.model.resize_token_embeddings(required_vocab_size)
                            logger.info(f"✓ Resized embeddings via model.resize_token_embeddings()")

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
                        logger.info(f"✓ Resized LM head to {required_vocab_size}")
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
                            logger.info(f"✓ Resized LM head to {required_vocab_size}")

                except Exception as e:
                    logger.error(f"❌ Failed to resize embeddings: {e}")
                    raise RuntimeError(f"Failed to resize embeddings to match tokenizer vocabulary: {e}")

                # Verify resize worked
                new_vocab_size = None
                if hasattr(model_unwrapped.language_model, 'get_input_embeddings'):
                    new_vocab_size = model_unwrapped.language_model.get_input_embeddings().weight.shape[0]
                elif hasattr(model_unwrapped.language_model, 'model'):
                    if hasattr(model_unwrapped.language_model.model, 'get_input_embeddings'):
                        new_vocab_size = model_unwrapped.language_model.model.get_input_embeddings().weight.shape[0]

                if new_vocab_size == required_vocab_size:
                    logger.info(f"✅ Embedding resize successful: {new_vocab_size} tokens")
                else:
                    error_msg = f"❌ Embedding resize FAILED! Size is {new_vocab_size}, expected {required_vocab_size}"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
            else:
                logger.info(f"✓ Embedding size matches tokenizer: {required_vocab_size} tokens")

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

            robot_dir = Path(args.robot_data) if args.robot_data else Path(Path(__file__).parent.parent / 'robot-selection-dataset')

            # Create robot selection data if needed
            if not robot_dir.exists():
                logger.info(f"Creating robot selection dataset at {robot_dir}")
                from embervlm.data.robot_loader import create_robot_selection_dataset
                create_robot_selection_dataset(str(robot_dir))

            if robot_dir.exists():
                run_stage3_training(
                    model=model,
                    config=stage3_config,
                    robot_data_dir=str(robot_dir),
                    tokenizer=tokenizer,
                    robot_epochs=args.stage3_robot_epochs,
                )
                # Unwrap model from DDP if needed
                from embervlm.training.train_utils import unwrap_model
                model = unwrap_model(model)
            else:
                logger.warning(f"Stage 3 robot data not found at {robot_dir}, skipping...")

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

            reasoning_dir = args.reasoning_data or str(output_dir / 'reasoning-data')

            if Path(reasoning_dir).exists():
                run_stage4_training(
                    model=model,
                    config=stage4_config,
                    data_dir=reasoning_dir,
                    tokenizer=tokenizer,
                    phase1_epochs=args.stage4_phase1_epochs,
                    phase2_epochs=args.stage4_phase2_epochs,
                )
                # Unwrap model from DDP if needed
                from embervlm.training.train_utils import unwrap_model
                model = unwrap_model(model)
            else:
                logger.warning("Stage 4 data not provided, skipping...")

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

    finally:
        # Stop carbon tracking (only rank 0 has a tracker)
        if carbon_tracker is not None:
            try:
                total_emissions = carbon_tracker.stop()
                logger.info(f"Total training emissions: {total_emissions:.4f} kg CO2eq")
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

    # HuggingFace Hub
    parser.add_argument('--push_to_hub', action='store_true',
                       help='Push to HuggingFace Hub')
    parser.add_argument('--hub_model_id', type=str, default='embervlm',
                       help='HuggingFace Hub model ID')

    args = parser.parse_args()

    # Run training
    model = run_all_stages(args)

    logger.info("Training complete!")


if __name__ == "__main__":
    main()

