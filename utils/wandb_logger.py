"""
EmberVLM WandB Logger
Unified logging for distributed training with single-process WandB.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Optional, Any, List
from datetime import datetime
import json

logger = logging.getLogger(__name__)

# Try to import wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


class WandBLogger:
    """
    WandB logger with support for distributed training.
    Ensures only one process logs to WandB.
    """

    def __init__(
        self,
        project: str = "EmberVLM",
        entity: Optional[str] = None,
        run_name: Optional[str] = None,
        config: Optional[Dict] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
        output_dir: str = "wandb_logs",
        rank: int = 0,
        world_size: int = 1,
        enabled: bool = True
    ):
        self.project = project
        self.entity = entity
        self.run_name = run_name or f"embervlm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.config = config or {}
        self.tags = tags or []
        self.notes = notes
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.rank = rank
        self.world_size = world_size
        self.is_main_process = rank == 0
        self.enabled = enabled and WANDB_AVAILABLE

        self.run = None
        self._initialized = False

        # Local log buffer for non-main processes
        self.log_buffer: List[Dict] = []

    def init(self, resume: Optional[str] = None):
        """Initialize WandB run (only on main process)."""
        if not self.enabled:
            logger.info("WandB logging disabled")
            return

        if not self.is_main_process:
            logger.debug(f"Rank {self.rank}: Skipping WandB init (not main process)")
            return

        if self._initialized:
            return

        try:
            # Set environment for distributed training
            os.environ['WANDB_RUN_GROUP'] = self.run_name

            self.run = wandb.init(
                project=self.project,
                entity=self.entity,
                name=self.run_name,
                config=self.config,
                tags=self.tags,
                notes=self.notes,
                dir=str(self.output_dir),
                resume=resume
            )

            self._initialized = True
            logger.info(f"WandB initialized: {self.run.url}")

        except Exception as e:
            logger.warning(f"Could not initialize WandB: {e}")
            self.enabled = False

    def log(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        commit: bool = True
    ):
        """Log metrics to WandB."""
        if not self.enabled:
            return

        if not self.is_main_process:
            # Buffer logs from non-main processes
            self.log_buffer.append({
                'metrics': metrics,
                'step': step,
                'timestamp': datetime.now().isoformat()
            })
            return

        if not self._initialized:
            self.init()

        if self.run:
            try:
                wandb.log(metrics, step=step, commit=commit)
            except Exception as e:
                logger.debug(f"WandB log failed: {e}")

    def log_image(
        self,
        key: str,
        image: Any,
        step: Optional[int] = None,
        caption: Optional[str] = None
    ):
        """Log image to WandB."""
        if not self.enabled or not self.is_main_process or not self._initialized:
            return

        try:
            wandb.log({
                key: wandb.Image(image, caption=caption)
            }, step=step)
        except Exception as e:
            logger.debug(f"WandB image log failed: {e}")

    def log_table(
        self,
        key: str,
        data: List[Dict],
        columns: Optional[List[str]] = None,
        step: Optional[int] = None
    ):
        """Log table to WandB."""
        if not self.enabled or not self.is_main_process or not self._initialized:
            return

        try:
            if columns is None and data:
                columns = list(data[0].keys())

            table = wandb.Table(columns=columns)
            for row in data:
                table.add_data(*[row.get(c) for c in columns])

            wandb.log({key: table}, step=step)
        except Exception as e:
            logger.debug(f"WandB table log failed: {e}")

    def log_artifact(
        self,
        artifact_path: str,
        artifact_name: str,
        artifact_type: str = "model",
        metadata: Optional[Dict] = None
    ):
        """Log artifact to WandB."""
        if not self.enabled or not self.is_main_process or not self._initialized:
            return

        try:
            artifact = wandb.Artifact(
                name=artifact_name,
                type=artifact_type,
                metadata=metadata
            )

            if Path(artifact_path).is_dir():
                artifact.add_dir(artifact_path)
            else:
                artifact.add_file(artifact_path)

            wandb.log_artifact(artifact)
            logger.info(f"Logged artifact: {artifact_name}")

        except Exception as e:
            logger.warning(f"Could not log artifact: {e}")

    def log_model_card(
        self,
        step: int,
        metrics: Dict[str, float],
        model_info: Dict[str, Any],
        carbon_info: Optional[Dict] = None
    ):
        """Log model card update."""
        if not self.enabled or not self.is_main_process or not self._initialized:
            return

        # Create model card content
        card_content = f"""
## EmberVLM Checkpoint - Step {step}

### Model Specifications
- **Total Parameters**: {model_info.get('total_params', 'N/A'):,}
- **Trainable Parameters**: {model_info.get('trainable_params', 'N/A'):,} ({model_info.get('trainable_ratio', 0)*100:.1f}%)
- **Vision Encoder**: RepViT-XXS-M0.9 (frozen)
- **Language Model**: TinyLLM-30M (partial fine-tuning)
- **Fusion Module**: ~500K parameters

### Training Statistics
- **Current Step**: {step}
- **Training Loss**: {metrics.get('train_loss', 'N/A'):.4f}
- **Validation Loss**: {metrics.get('val_loss', 'N/A'):.4f}
- **Robot Selection Accuracy**: {metrics.get('robot_accuracy', 'N/A')*100:.1f}%

### Environmental Impact
- **CO₂ Emitted**: {carbon_info.get('emissions_kg', 0):.4f} kg
- **Energy Used**: {carbon_info.get('energy_kwh', 0):.4f} kWh
- **Training Time**: {carbon_info.get('duration_hours', 0):.2f} hours

### Performance Metrics
| Metric | Value |
|--------|-------|
| VQA Accuracy | {metrics.get('vqa_accuracy', 'N/A')} |
| Captioning (CIDEr) | {metrics.get('cider', 'N/A')} |
| Robot Selection Acc | {metrics.get('robot_accuracy', 'N/A')} |
| Inference Latency | {metrics.get('inference_ms', 'N/A')} ms/token |
"""

        try:
            # Save model card
            card_path = self.output_dir / f"model_card_step_{step}.md"
            with open(card_path, 'w') as f:
                f.write(card_content)

            # Log to WandB
            wandb.log({
                'model_card': wandb.Html(card_content.replace('\n', '<br>'))
            }, step=step)

        except Exception as e:
            logger.debug(f"Could not log model card: {e}")

    def watch_model(self, model: Any, log_freq: int = 100):
        """Watch model gradients."""
        if not self.enabled or not self.is_main_process or not self._initialized:
            return

        try:
            wandb.watch(model, log='all', log_freq=log_freq)
        except Exception as e:
            logger.debug(f"Could not watch model: {e}")

    def finish(self):
        """Finish WandB run."""
        # Save buffered logs
        if self.log_buffer:
            buffer_path = self.output_dir / f"log_buffer_rank{self.rank}.json"
            with open(buffer_path, 'w') as f:
                json.dump(self.log_buffer, f)

        if self.run and self.is_main_process:
            wandb.finish()
            self._initialized = False
            logger.info("WandB run finished")

    def get_run_url(self) -> Optional[str]:
        """Get WandB run URL."""
        if self.run:
            return self.run.url
        return None

    def update_config(self, config_updates: Dict):
        """Update run config."""
        if not self.enabled or not self.is_main_process or not self._initialized:
            return

        try:
            wandb.config.update(config_updates, allow_val_change=True)
        except Exception as e:
            logger.debug(f"Could not update config: {e}")


def create_wandb_logger(
    project: str = "EmberVLM",
    config: Optional[Dict] = None,
    **kwargs
) -> WandBLogger:
    """Factory function for WandB logger."""
    # Get distributed info from environment
    rank = int(os.environ.get('RANK', os.environ.get('LOCAL_RANK', 0)))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    return WandBLogger(
        project=project,
        config=config,
        rank=rank,
        world_size=world_size,
        **kwargs
    )


if __name__ == "__main__":
    # Test WandB logger
    print("Testing WandB Logger...")

    logger_instance = WandBLogger(
        project="test_project",
        run_name="test_run",
        config={'test': True},
        enabled=False  # Disable for testing without WandB
    )

    # Test logging
    logger_instance.log({'loss': 0.5, 'accuracy': 0.85}, step=0)
    logger_instance.log({'loss': 0.4, 'accuracy': 0.90}, step=1)

    print("Logged metrics (buffered since WandB disabled)")
    print(f"Buffer: {logger_instance.log_buffer}")

    logger_instance.finish()
    print("WandB logger tests complete!")

