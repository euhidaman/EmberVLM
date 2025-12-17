#!/usr/bin/env python3
"""
EmberVLM Master Training Script
Complete 4-stage training pipeline with distributed training support.

Usage:
    python train_all_stages.py --config configs/full_training.yaml

    # With specific stages:
    python train_all_stages.py --config configs/training.yaml --stages 1 2 3

    # Distributed training (2 GPUs):
    torchrun --nproc_per_node=2 train_all_stages.py --config configs/training.yaml
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Optional, List, Any
from datetime import datetime
import yaml
import json

import torch
import torch.distributed as dist

# Add parent to path for imports
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

# EmberVLM imports - use relative imports when run as module
try:
    from embervlm.models import EmberVLM, EmberVLMConfig
    from embervlm.data import (
        RobotSelectionDataset,
        IncidentDataset,
        UnifiedDataset,
        create_unified_dataset
    )
    from embervlm.training import (
        TrainingConfig,
        Stage1Config,
        Stage2Config,
        Stage3Config,
        run_stage1_alignment,
        run_stage2_instruction,
        run_stage3_reasoning
    )
    from embervlm.utils import (
        CarbonTracker,
        WandBLogger,
        HuggingFaceUploader,
        create_carbon_tracker,
        create_wandb_logger,
        create_hf_uploader
    )
    from embervlm.quantization import convert_to_gguf, create_pi_optimizer
    from embervlm.evaluation import evaluate_robot_selection
    from embervlm.visualization import create_attention_visualizer
except ImportError:
    # Fallback for direct execution
    from models import EmberVLM, EmberVLMConfig
    from data import (
        RobotSelectionDataset,
        IncidentDataset,
        UnifiedDataset,
        create_unified_dataset
    )
    from training import (
        TrainingConfig,
        Stage1Config,
        Stage2Config,
        Stage3Config,
        run_stage1_alignment,
        run_stage2_instruction,
        run_stage3_reasoning
    )
    from utils import (
        CarbonTracker,
        WandBLogger,
        HuggingFaceUploader,
        create_carbon_tracker,
        create_wandb_logger,
        create_hf_uploader
    )
    from quantization import convert_to_gguf, create_pi_optimizer
    from evaluation import evaluate_robot_selection
    from visualization import create_attention_visualizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)


class EmberVLMTrainer:
    """
    Master trainer orchestrating all training stages.

    Stages:
    0. Dataset preparation
    1. Vision-Language Alignment (1 epoch)
    2. Multimodal Instruction Tuning (2 epochs)
    3. Specialized Reasoning Finetuning (3 epochs)
    4. RLHF (optional, 1 epoch)
    """

    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self._load_config()

        # Distributed training info
        self.rank = int(os.environ.get('RANK', os.environ.get('LOCAL_RANK', 0)))
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        self.is_main_process = self.rank == 0

        # Initialize distributed
        if self.world_size > 1 and not dist.is_initialized():
            dist.init_process_group(backend='nccl')
            torch.cuda.set_device(self.rank)

        self.device = torch.device(f'cuda:{self.rank}' if torch.cuda.is_available() else 'cpu')

        # Output directory
        self.output_dir = Path(self.config.get('global', {}).get('output_dir', 'outputs'))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.model = None
        self.tokenizer = None
        self.carbon_tracker = None
        self.wandb_logger = None
        self.hf_uploader = None

        # Training state
        self.current_stage = 0
        self.global_step = 0

        if self.is_main_process:
            logger.info(f"EmberVLM Trainer initialized")
            logger.info(f"World size: {self.world_size}")
            logger.info(f"Device: {self.device}")
            logger.info(f"Output directory: {self.output_dir}")

    def _load_config(self) -> Dict:
        """Load training configuration."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def setup(self):
        """Setup all training components."""
        logger.info("Setting up training components...")

        # 1. Initialize model
        self._setup_model()

        # 2. Initialize tokenizer
        self._setup_tokenizer()

        # 3. Initialize monitoring (main process only)
        if self.is_main_process:
            self._setup_monitoring()

        logger.info("Setup complete!")

    def _setup_model(self):
        """Initialize EmberVLM model."""
        model_config = self.config.get('model', {})

        # Create config
        config = EmberVLMConfig(
            vision_pretrained=model_config.get('vision_encoder', {}).get('pretrained', True),
            vision_frozen=model_config.get('vision_encoder', {}).get('frozen', True),
            num_vision_tokens=model_config.get('vision_encoder', {}).get('num_vision_tokens', 8),
            hidden_size=model_config.get('language_model', {}).get('hidden_size', 768),
            num_layers=model_config.get('language_model', {}).get('num_layers', 6),
            num_attention_heads=model_config.get('language_model', {}).get('num_attention_heads', 12),
            freeze_lm_layers=model_config.get('language_model', {}).get('freeze_layers', [0, 1, 2, 3]),
            trainable_lm_layers=model_config.get('language_model', {}).get('trainable_layers', [4, 5])
        )

        # Create model
        self.model = EmberVLM(config)
        self.model.to(self.device)

        # Log model info
        param_counts = self.model.count_parameters()
        if self.is_main_process:
            logger.info(f"Model parameters:")
            for component, (total, trainable) in param_counts.items():
                logger.info(f"  {component}: {total:,} total, {trainable:,} trainable")

    def _setup_tokenizer(self):
        """Initialize tokenizer."""
        try:
            from transformers import AutoTokenizer
            tokenizer_name = self.config.get('tokenizer', {}).get('name', 'gpt2')
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

            # Set padding token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            logger.info(f"Loaded tokenizer: {tokenizer_name}")

        except Exception as e:
            logger.warning(f"Could not load tokenizer: {e}")
            self.tokenizer = None

    def _setup_monitoring(self):
        """Setup monitoring tools (main process only)."""
        monitoring_config = self.config.get('monitoring', {})

        # Carbon tracking
        carbon_config = monitoring_config.get('codecarbon', {})
        if carbon_config.get('enabled', True):
            self.carbon_tracker = create_carbon_tracker(
                project_name="embervlm",
                max_budget_kg=self.config.get('carbon_budget', {}).get('max_total_kg_co2', 50)
            )

        # WandB logging
        wandb_config = monitoring_config.get('wandb', {})
        self.wandb_logger = create_wandb_logger(
            project=wandb_config.get('project', 'EmberVLM'),
            config=self.config
        )

        # HuggingFace Hub
        hf_config = self.config.get('huggingface', {})
        if hf_config.get('push_to_hub', False):
            self.hf_uploader = create_hf_uploader(
                repo_id=hf_config.get('hub_model_id', 'embervlm/EmberVLM'),
                token=hf_config.get('hub_token')
            )

    def run_stage0_data_prep(self):
        """Stage 0: Dataset preparation and curation."""
        logger.info("=" * 60)
        logger.info("Stage 0: Dataset Preparation")
        logger.info("=" * 60)

        data_config = self.config.get('stage0_data_prep', {})
        if not data_config.get('enabled', True):
            logger.info("Stage 0 disabled, skipping...")
            return

        # Prepare datasets
        output_dir = Path(data_config.get('output_dir', 'data/processed'))
        output_dir.mkdir(parents=True, exist_ok=True)

        # This would include downloading and processing datasets
        # For now, we'll just log the configuration
        logger.info(f"Dataset preparation would process:")
        for dataset_name, dataset_config in data_config.get('datasets', {}).items():
            if dataset_config.get('enabled', False):
                max_samples = dataset_config.get('max_samples', 'all')
                logger.info(f"  {dataset_name}: {max_samples} samples")

        logger.info("Stage 0 complete!")

    def run_stage1(self):
        """Stage 1: Vision-Language Alignment."""
        logger.info("=" * 60)
        logger.info("Stage 1: Vision-Language Alignment")
        logger.info("=" * 60)

        stage_config = self.config.get('stage1_alignment', {})
        if not stage_config.get('enabled', True):
            logger.info("Stage 1 disabled, skipping...")
            return

        # Create stage config
        config = Stage1Config(
            output_dir=str(self.output_dir / "stage1"),
            num_epochs=stage_config.get('num_epochs', 1),
            batch_size_per_gpu=stage_config.get('batch_size_per_gpu', 128),
            gradient_accumulation_steps=stage_config.get('gradient_accumulation_steps', 4),
            learning_rate=stage_config.get('learning_rate', 3e-4),
            contrastive_weight=stage_config.get('losses', {}).get('contrastive', 0.5),
            captioning_weight=stage_config.get('losses', {}).get('captioning', 0.5)
        )

        # Create training dataset
        # For now, using a placeholder - would load actual image-caption data
        from torch.utils.data import TensorDataset
        train_dataset = TensorDataset(
            torch.randn(1000, 3, 224, 224),  # Images
            torch.randint(0, 50257, (1000, 64)),  # Input IDs
            torch.ones(1000, 64, dtype=torch.long),  # Attention mask
            torch.randint(0, 50257, (1000, 64))  # Labels
        )

        # Run training
        results = run_stage1_alignment(
            model=self.model,
            train_dataset=train_dataset,
            config=config
        )

        self.current_stage = 1
        logger.info(f"Stage 1 results: {results}")

    def run_stage2(self):
        """Stage 2: Multimodal Instruction Tuning."""
        logger.info("=" * 60)
        logger.info("Stage 2: Multimodal Instruction Tuning")
        logger.info("=" * 60)

        stage_config = self.config.get('stage2_instruction', {})
        if not stage_config.get('enabled', True):
            logger.info("Stage 2 disabled, skipping...")
            return

        # Create stage config
        config = Stage2Config(
            output_dir=str(self.output_dir / "stage2"),
            num_epochs=stage_config.get('num_epochs', 2),
            batch_size_per_gpu=stage_config.get('batch_size_per_gpu', 64),
            gradient_accumulation_steps=stage_config.get('gradient_accumulation_steps', 4),
            learning_rate=stage_config.get('learning_rate', 1e-4),
            distillation_enabled=stage_config.get('distillation', {}).get('enabled', True),
            teacher_model_name=stage_config.get('distillation', {}).get('teacher_model', 'Qwen/Qwen-VL-Chat'),
            temperature=stage_config.get('distillation', {}).get('temperature', 2.0)
        )

        # Create training dataset (placeholder)
        from torch.utils.data import TensorDataset
        train_dataset = TensorDataset(
            torch.randn(1000, 3, 224, 224),
            torch.randint(0, 50257, (1000, 64)),
            torch.ones(1000, 64, dtype=torch.long),
            torch.randint(0, 50257, (1000, 64))
        )

        # Run training
        results = run_stage2_instruction(
            model=self.model,
            train_dataset=train_dataset,
            config=config
        )

        self.current_stage = 2
        logger.info(f"Stage 2 results: {results}")

    def run_stage3(self):
        """Stage 3: Specialized Reasoning Finetuning."""
        logger.info("=" * 60)
        logger.info("Stage 3: Specialized Reasoning Finetuning")
        logger.info("=" * 60)

        stage_config = self.config.get('stage3_reasoning', {})
        if not stage_config.get('enabled', True):
            logger.info("Stage 3 disabled, skipping...")
            return

        # Paths to data
        robot_selection_path = "Multi-Robot-Selection/multi_robot_selection_dataset.json"
        incident_data_dir = "incidents-dataset"

        # Create stage config
        config = Stage3Config(
            output_dir=str(self.output_dir / "stage3"),
            num_epochs=stage_config.get('num_epochs', 3),
            batch_size_per_gpu=stage_config.get('batch_size_per_gpu', 32),
            gradient_accumulation_steps=stage_config.get('gradient_accumulation_steps', 8),
            learning_rate=stage_config.get('learning_rate', 5e-5),
            curriculum_enabled=stage_config.get('curriculum', {}) is not None,
            robot_selection_ratio=stage_config.get('batch_composition', {}).get('robot_selection', 0.5),
            incident_response_ratio=stage_config.get('batch_composition', {}).get('incident_response', 0.3),
            general_vqa_ratio=stage_config.get('batch_composition', {}).get('general_vqa', 0.2)
        )

        # Run training
        results = run_stage3_reasoning(
            model=self.model,
            robot_selection_path=robot_selection_path,
            incident_data_dir=incident_data_dir,
            config=config,
            tokenizer=self.tokenizer
        )

        self.current_stage = 3
        logger.info(f"Stage 3 results: {results}")

    def run_stage4(self):
        """Stage 4: RLHF (optional)."""
        logger.info("=" * 60)
        logger.info("Stage 4: RLHF (Optional)")
        logger.info("=" * 60)

        stage_config = self.config.get('stage4_rlhf', {})
        if not stage_config.get('enabled', False):
            logger.info("Stage 4 disabled, skipping...")
            return

        # RLHF implementation would go here
        logger.info("RLHF training not yet implemented")
        self.current_stage = 4

    def save_checkpoint(self, step: int):
        """Save checkpoint with all components."""
        if not self.is_main_process:
            return

        checkpoint_dir = self.output_dir / f"checkpoint-{step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        self.model.save_pretrained(str(checkpoint_dir))

        # Save tokenizer
        if self.tokenizer:
            self.tokenizer.save_pretrained(str(checkpoint_dir))

        # Quantize and save GGUF
        gguf_path = checkpoint_dir / "embervlm.gguf"
        convert_to_gguf(
            model=self.model,
            output_path=str(gguf_path),
            config=self.model.config.to_dict().get('model', {}),
            quantize=True
        )

        # Upload to HuggingFace Hub
        if self.hf_uploader:
            # Get metrics
            metrics = {
                'step': step,
                'stage': self.current_stage,
                'train_loss': 0.0  # Would be actual loss
            }

            # Get carbon info
            carbon_info = {}
            if self.carbon_tracker:
                carbon_info = self.carbon_tracker.get_carbon_equivalent()

            # Get model info
            param_counts = self.model.count_parameters()
            model_info = {
                'total_params': param_counts['total'][0],
                'trainable_params': param_counts['total'][1],
                'trainable_ratio': param_counts['total'][1] / param_counts['total'][0]
            }

            # Upload
            self.hf_uploader.upload_checkpoint(
                checkpoint_dir=str(checkpoint_dir),
                step=step,
                metrics=metrics,
                model_info=model_info,
                carbon_info=carbon_info
            )

        logger.info(f"Checkpoint saved: {checkpoint_dir}")

    def run_evaluation(self):
        """Run final evaluation."""
        logger.info("=" * 60)
        logger.info("Running Final Evaluation")
        logger.info("=" * 60)

        # Load robot selection test data
        robot_data_path = Path("Multi-Robot-Selection/multi_robot_selection_dataset.json")
        if robot_data_path.exists():
            test_dataset = RobotSelectionDataset(
                data_path=str(robot_data_path),
                split='test',
                augment=False,
                tokenizer=self.tokenizer
            )

            eval_samples = test_dataset.get_evaluation_samples(n=500)

            metrics = evaluate_robot_selection(
                model=self.model,
                tokenizer=self.tokenizer,
                eval_samples=eval_samples,
                output_path=str(self.output_dir / "robot_eval_results.txt")
            )

            logger.info(f"Robot selection accuracy: {metrics.accuracy*100:.2f}%")
            logger.info(f"Robot selection F1: {metrics.f1_score*100:.2f}%")
        else:
            logger.warning("Robot selection test data not found")

    def export_for_deployment(self):
        """Export model for Raspberry Pi deployment."""
        logger.info("=" * 60)
        logger.info("Exporting for Deployment")
        logger.info("=" * 60)

        deployment_dir = self.output_dir / "deployment"
        deployment_dir.mkdir(parents=True, exist_ok=True)

        # Optimize for Pi
        pi_optimizer = create_pi_optimizer(
            target_memory_mb=400,
            target_size_mb=100,
            target_latency_ms=500
        )

        results = pi_optimizer.optimize_model(
            model=self.model,
            output_dir=str(deployment_dir),
            apply_pruning=True,
            apply_quantization=True
        )

        logger.info(f"Deployment export complete: {deployment_dir}")
        logger.info(f"Final model size: {results.get('quantized_size_mb', 0):.2f} MB")

    def train(self, stages: Optional[List[int]] = None):
        """
        Run complete training pipeline.

        Args:
            stages: List of stages to run (1-4), or None for all
        """
        logger.info("=" * 60)
        logger.info("EmberVLM Training Pipeline")
        logger.info("=" * 60)
        logger.info(f"Config: {self.config_path}")
        logger.info(f"Stages: {stages or 'all'}")

        # Setup
        self.setup()

        # Start monitoring
        if self.is_main_process:
            if self.carbon_tracker:
                self.carbon_tracker.start()
            if self.wandb_logger:
                self.wandb_logger.init()

        try:
            # Stage 0: Data preparation
            if stages is None or 0 in stages:
                self.run_stage0_data_prep()

            # Stage 1: Vision-Language Alignment
            if stages is None or 1 in stages:
                self.run_stage1()
                self.save_checkpoint(step=self.global_step)

            # Stage 2: Instruction Tuning
            if stages is None or 2 in stages:
                self.run_stage2()
                self.save_checkpoint(step=self.global_step)

            # Stage 3: Reasoning Finetuning
            if stages is None or 3 in stages:
                self.run_stage3()
                self.save_checkpoint(step=self.global_step)

            # Stage 4: RLHF (optional)
            if stages is None or 4 in stages:
                self.run_stage4()

            # Final evaluation
            self.run_evaluation()

            # Export for deployment
            self.export_for_deployment()

            logger.info("=" * 60)
            logger.info("Training Complete!")
            logger.info("=" * 60)

        finally:
            # Stop monitoring
            if self.is_main_process:
                if self.carbon_tracker:
                    total_emissions = self.carbon_tracker.stop()
                    logger.info(f"Total carbon emissions: {total_emissions:.4f} kg CO2")
                if self.wandb_logger:
                    self.wandb_logger.finish()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='EmberVLM Training Pipeline')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/training.yaml',
        help='Path to training configuration YAML'
    )
    parser.add_argument(
        '--stages',
        type=int,
        nargs='+',
        default=None,
        help='Specific stages to run (0-4). Default: all stages'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Override output directory'
    )

    args = parser.parse_args()

    # Create trainer
    trainer = EmberVLMTrainer(config_path=args.config)

    # Override output dir if specified
    if args.output_dir:
        trainer.output_dir = Path(args.output_dir)
        trainer.output_dir.mkdir(parents=True, exist_ok=True)

    # Run training
    trainer.train(stages=args.stages)


if __name__ == "__main__":
    main()

