"""
Stage 3: Robot Fleet Selection Training

Trains the model for robot fleet selection based on task requirements.
Uses the robot-selection-dataset with augmentation.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from embervlm.models import EmberVLM
from embervlm.models.reasoning_heads import ReasoningLoss
from embervlm.training.train_utils import (
    TrainingConfig,
    setup_distributed,
    cleanup_distributed,
    is_main_process,
    get_rank,
    barrier,
    set_seed,
    get_optimizer,
    get_scheduler,
    get_grad_scaler,
    get_autocast_context,
    wrap_model_ddp,
    save_checkpoint,
    MetricTracker,
    print_trainable_parameters,
)
from embervlm.data.robot_loader import get_robot_selection_dataloader
from embervlm.training.robot_metrics import RobotSelectionMetrics, ReasoningQualityMetrics
from embervlm.monitoring.wandb_logger import EnhancedWandbLogger
from embervlm.monitoring.carbon_tracker import CarbonTracker

logger = logging.getLogger(__name__)

# Try to import stage visualizer
try:
    from embervlm.monitoring.stage_visualizations import Stage3Visualizer
    HAS_STAGE_VIZ = True
except ImportError:
    HAS_STAGE_VIZ = False
    Stage3Visualizer = None


class Stage3Trainer:
    """Trainer for Stage 3: Robot Fleet Selection."""

    def __init__(
        self,
        model: EmberVLM,
        config: TrainingConfig,
        robot_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        tokenizer: Any = None,
    ):
        self.config = config
        self.tokenizer = tokenizer

        # Setup distributed
        self.rank, self.local_rank, self.world_size = setup_distributed()
        self.device = torch.device(f'cuda:{self.local_rank}')

        set_seed(config.seed, self.rank)

        # Unwrap model if it was previously wrapped with DDP
        from embervlm.training.train_utils import unwrap_model
        model = unwrap_model(model)

        # CRITICAL: Validate embedding size matches tokenizer BEFORE any training
        # This prevents cryptic CUDA index out of bounds errors
        if tokenizer is not None:
            required_vocab_size = len(tokenizer)
            current_vocab_size = None

            if hasattr(model.language_model, 'get_input_embeddings'):
                current_vocab_size = model.language_model.get_input_embeddings().weight.shape[0]
            elif hasattr(model.language_model, 'model'):
                if hasattr(model.language_model.model, 'get_input_embeddings'):
                    current_vocab_size = model.language_model.model.get_input_embeddings().weight.shape[0]

            if current_vocab_size is not None:
                if current_vocab_size != required_vocab_size:
                    error_msg = (
                        f"\n{'='*80}\n"
                        f"❌ CRITICAL ERROR: Token embedding size mismatch!\n"
                        f"{'='*80}\n"
                        f"  Tokenizer vocabulary size: {required_vocab_size}\n"
                        f"  Model embedding layer size: {current_vocab_size}\n"
                        f"  Difference: {required_vocab_size - current_vocab_size} tokens\n"
                        f"\n"
                        f"This mismatch causes index out of bounds errors during training.\n"
                        f"The model's embedding layer must be resized to match the tokenizer.\n"
                        f"\n"
                        f"Special tokens in tokenizer:\n"
                    )
                    for token in tokenizer.additional_special_tokens:
                        token_id = tokenizer.convert_tokens_to_ids(token)
                        error_msg += f"    {token} → ID {token_id}\n"
                    error_msg += f"\n"
                    error_msg += f"SOLUTION: The train_all.py script should have resized the embeddings.\n"
                    error_msg += f"If this error persists, manually resize embeddings in train_all.py\n"
                    error_msg += f"before calling Stage 3 training.\n"
                    error_msg += f"{'='*80}\n"

                    logger.error(error_msg)
                    raise ValueError(error_msg)
                else:
                    logger.info(f"✓ Embedding size validation passed: {current_vocab_size} tokens")

                    # Additional validation: Check if special tokens are actually within bounds
                    special_token_ids = [tokenizer.convert_tokens_to_ids(token) for token in tokenizer.additional_special_tokens]
                    max_special_id = max(special_token_ids) if special_token_ids else 0

                    if max_special_id >= current_vocab_size:
                        error_msg = (
                            f"\n{'='*80}\n"
                            f"❌ CRITICAL ERROR: Special token ID out of bounds!\n"
                            f"{'='*80}\n"
                            f"  Maximum special token ID: {max_special_id}\n"
                            f"  Model embedding layer size: {current_vocab_size}\n"
                            f"\n"
                            f"Special token IDs:\n"
                        )
                        for token, token_id in zip(tokenizer.additional_special_tokens, special_token_ids):
                            status = "❌ OUT OF BOUNDS" if token_id >= current_vocab_size else "✓ OK"
                            error_msg += f"    {token} → ID {token_id} {status}\n"
                        error_msg += f"{'='*80}\n"
                        logger.error(error_msg)
                        raise ValueError(error_msg)
                    else:
                        logger.info(f"✓ All special tokens within valid range (max ID: {max_special_id})")


        # Ensure model is on correct device before DDP
        try:
            model = model.to(self.device)
            torch.cuda.synchronize()  # Ensure CUDA operations complete
        except RuntimeError as e:
            logger.error(f"[Rank {self.rank}] Failed to move model to device: {e}")
            raise

        # Synchronize to ensure all ranks have model loaded
        if torch.distributed.is_initialized() and self.world_size > 1:
            try:
                torch.distributed.barrier()
            except Exception as e:
                logger.error(f"[Rank {self.rank}] Barrier failed: {e}")
                raise

        # Model
        self.model = wrap_model_ddp(model, config, self.device)

        # Data loaders
        self.robot_dataloader = robot_dataloader
        self.val_dataloader = val_dataloader

        # Optimizer
        self.optimizer = get_optimizer(self.model, config)
        self.scheduler = get_scheduler(self.optimizer, config)
        self.scaler = get_grad_scaler(config)

        # Reasoning loss
        model_ref = self.model.module if hasattr(self.model, 'module') else self.model
        self.reasoning_loss = ReasoningLoss(
            num_robots=model_ref.config.num_robots,
            reasoning_weight=1.0,
            robot_weight=1.0,
            action_weight=1.0,
            consistency_weight=0.5,
        )

        # Loss weights
        self.ce_weight = 0.6
        self.reasoning_consistency_weight = 0.4

        # Logging - only main process initializes W&B and carbon tracker
        self.wandb_logger = None
        self.carbon_tracker = None

        if is_main_process():
            logger.info("Initializing Enhanced W&B logger with visualizations (main process)...")
            try:
                self.wandb_logger = EnhancedWandbLogger(
                    project="embervlm",
                    name="stage3_robot_selection",
                    config=config.to_dict(),
                    output_dir=str(Path(config.output_dir) / 'visualizations'),
                )
                logger.info("Enhanced W&B logger initialized with visualizations")
            except Exception as e:
                logger.warning(f"Failed to initialize Enhanced W&B logger: {e}")
                self.wandb_logger = None

            try:
                self.carbon_tracker = CarbonTracker(output_dir=config.output_dir)
                logger.info("Carbon tracker initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize carbon tracker: {e}")
                self.carbon_tracker = None

        # Synchronize all ranks after logging initialization
        if torch.distributed.is_initialized() and self.world_size > 1:
            logger.info(f"[Rank {self.rank}] Waiting at post-logging barrier...")
            torch.distributed.barrier()
            logger.info(f"[Rank {self.rank}] Passed post-logging barrier")

        self.metric_tracker = MetricTracker()
        self.robot_metrics = RobotSelectionMetrics(
            num_robots=5,
            robot_names=["Drone", "Underwater Robot", "Humanoid", "Robot with Wheels", "Robot with Legs"]
        )
        self.reasoning_metrics = ReasoningQualityMetrics()
        self.global_step = 0

        # Stage 3 specific visualizer
        self.stage_visualizer = None
        if is_main_process() and HAS_STAGE_VIZ:
            try:
                self.stage_visualizer = Stage3Visualizer(
                    output_dir=str(Path(config.output_dir) / 'visualizations')
                )
                logger.info("✓ Stage3Visualizer initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Stage3Visualizer: {e}")

        # Track data for visualizations
        self.last_robot_preds = None
        self.last_robot_targets = None
        self.last_confidences = None

        if is_main_process():
            print_trainable_parameters(self.model)

    def train_robot_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Training step for robot selection."""
        pixel_values = batch['pixel_values'].to(self.device)
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)
        robot_targets = batch['robot_target'].to(self.device)
        multi_robot_targets = batch.get('multi_robot_target')
        if multi_robot_targets is not None:
            multi_robot_targets = multi_robot_targets.to(self.device)

        # CRITICAL: Validate input_ids and labels are within embedding bounds before forward pass
        # This prevents cryptic CUDA index out of bounds errors
        model_ref = self.model.module if hasattr(self.model, 'module') else self.model

        # Get actual embedding layer size
        vocab_size = None
        if hasattr(model_ref.language_model, 'get_input_embeddings'):
            vocab_size = model_ref.language_model.get_input_embeddings().weight.shape[0]
        elif hasattr(model_ref.language_model, 'model'):
            if hasattr(model_ref.language_model.model, 'get_input_embeddings'):
                vocab_size = model_ref.language_model.model.get_input_embeddings().weight.shape[0]

        if vocab_size is not None:
            # Validate and clamp input_ids
            max_token_id = input_ids.max().item()
            if max_token_id >= vocab_size:
                logger.error(
                    f"❌ CRITICAL: input_ids contain token ID {max_token_id} >= vocab_size {vocab_size}!"
                )
                # Clamp to prevent crash - this is a safeguard
                input_ids = torch.clamp(input_ids, max=vocab_size - 1)
                logger.warning(f"   Token IDs clamped to valid range [0, {vocab_size - 1}]")

            # Check for negative token IDs
            min_token_id = input_ids.min().item()
            if min_token_id < 0:
                logger.error(f"❌ CRITICAL: input_ids contain negative token ID {min_token_id}!")
                input_ids = torch.clamp(input_ids, min=0)
                logger.warning(f"   Token IDs clamped to valid range [0, {vocab_size - 1}]")

            # Validate and clamp labels (skip -100 which is ignore index)
            valid_labels_mask = labels != -100
            if valid_labels_mask.any():
                valid_labels = labels[valid_labels_mask]
                max_label = valid_labels.max().item()
                min_label = valid_labels.min().item()

                if max_label >= vocab_size or min_label < 0:
                    if max_label >= vocab_size:
                        logger.error(f"❌ CRITICAL: labels contain token ID {max_label} >= vocab_size {vocab_size}!")
                    if min_label < 0:
                        logger.error(f"❌ CRITICAL: labels contain negative token ID {min_label}!")

                    # Clamp labels: preserve -100 for ignore, clamp everything else to valid range
                    labels = torch.where(
                        valid_labels_mask,
                        torch.clamp(labels, min=0, max=vocab_size - 1),
                        labels
                    )
                    logger.warning(f"   Labels clamped to valid range [0, {vocab_size - 1}]")

        # Clamp robot targets to valid range to avoid gather OOB in losses
        num_robots = getattr(model_ref.config, 'num_robots', None)
        if num_robots is not None:
            if robot_targets is not None:
                max_robot = robot_targets.max().item()
                min_robot = robot_targets.min().item()
                if max_robot >= num_robots or min_robot < 0:
                    logger.warning(
                        f"⚠️ robot_target out of bounds detected (min={min_robot}, max={max_robot}, num_robots={num_robots}); clamping"
                    )
                    robot_targets = torch.clamp(robot_targets, 0, num_robots - 1)

            if multi_robot_targets is not None:
                # Ensure multi-hot vector width matches num_robots
                if multi_robot_targets.size(-1) > num_robots:
                    multi_robot_targets = multi_robot_targets[..., :num_robots]
                elif multi_robot_targets.size(-1) < num_robots:
                    pad_width = num_robots - multi_robot_targets.size(-1)
                    pad_shape = list(multi_robot_targets.shape[:-1]) + [pad_width]
                    pad_tensor = torch.zeros(pad_shape, device=multi_robot_targets.device, dtype=multi_robot_targets.dtype)
                    multi_robot_targets = torch.cat([multi_robot_targets, pad_tensor], dim=-1)

                # Clamp to [0,1]
                multi_robot_targets = multi_robot_targets.clamp(0.0, 1.0)

        with get_autocast_context(self.config):
            outputs = self.model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                labels=labels,
                robot_targets=robot_targets,
                return_reasoning=True,
            )

            # Validate robot logits width matches num_robots
            if num_robots is not None and 'robot_logits' in outputs:
                if outputs['robot_logits'].shape[-1] != num_robots:
                    logger.error(
                        f"❌ robot_logits dim mismatch: got {outputs['robot_logits'].shape[-1]}, expected {num_robots}; trimming for safety"
                    )
                    outputs['robot_logits'] = outputs['robot_logits'][..., :num_robots]

            loss = outputs['loss']

            # Robot selection accuracy
            if 'robot_logits' in outputs:
                robot_preds = outputs['robot_logits'].argmax(dim=-1)
                robot_acc = (robot_preds == robot_targets).float().mean()

                # Store for visualization
                self.last_robot_preds = robot_preds.detach()
                self.last_robot_targets = robot_targets.detach()
                self.last_confidences = torch.softmax(outputs['robot_logits'], dim=-1).max(dim=-1)[0].detach()
            else:
                robot_acc = torch.tensor(0.0)

        return loss, {
            'robot_loss': loss.item(),
            'robot_accuracy': robot_acc.item(),
        }

    def train_epoch(self, epoch: int):
        """Train for one epoch."""
        self.model.train()
        self.metric_tracker.reset()

        dataloader = self.robot_dataloader
        desc = f"Robot Selection Epoch {epoch}"

        progress_bar = tqdm(
            dataloader,
            desc=desc,
            disable=not is_main_process(),
        )

        for batch_idx, batch in enumerate(progress_bar):
            loss, metrics = self.train_robot_step(batch)

            # Backward
            if self.config.gradient_accumulation_steps > 1:
                loss = loss / self.config.gradient_accumulation_steps

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient step
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

                metrics['lr'] = self.scheduler.get_last_lr()[0]
                self.metric_tracker.update(metrics)

                # Logging
                if self.global_step % self.config.log_steps == 0:
                    avg_metrics = self.metric_tracker.get_average()

                    if is_main_process():
                        if self.wandb_logger is not None:
                            # Use enhanced logging with visualizations
                            if hasattr(self.wandb_logger, 'log_with_visualization'):
                                self.wandb_logger.log_with_visualization(
                                    avg_metrics,
                                    step=self.global_step,
                                    stage_name="stage3",
                                )
                            else:
                                self.wandb_logger.log(avg_metrics, step=self.global_step)

                            # Log gradient distribution every 500 steps
                            if self.global_step % 500 == 0 and hasattr(self.wandb_logger, 'log_gradient_distribution'):
                                gradients = {}
                                for name, param in self.model.named_parameters():
                                    if param.grad is not None and param.requires_grad:
                                        gradients[name.split('.')[-1]] = param.grad
                                if gradients:
                                    self.wandb_logger.log_gradient_distribution(
                                        gradients, self.global_step, "stage3"
                                    )

                                # Stage 3 specific visualizations
                                if self.stage_visualizer is not None and self.last_robot_preds is not None:
                                    try:
                                        # Confusion matrix
                                        _, cm_img = self.stage_visualizer.plot_confusion_matrix(
                                            self.last_robot_preds,
                                            self.last_robot_targets,
                                            self.global_step
                                        )
                                        self.wandb_logger.log_image(
                                            "stage3/confusion_matrix", cm_img, step=self.global_step
                                        )
                                        logger.info(f"✓ Logged confusion matrix at step {self.global_step}")

                                        # Confidence calibration
                                        if self.last_confidences is not None:
                                            correct = (self.last_robot_preds == self.last_robot_targets)
                                            _, cal_img = self.stage_visualizer.plot_confidence_calibration(
                                                self.last_confidences,
                                                correct,
                                                self.global_step
                                            )
                                            self.wandb_logger.log_image(
                                                "stage3/calibration", cal_img, step=self.global_step
                                            )
                                            logger.info(f"✓ Logged calibration plot at step {self.global_step}")
                                    except Exception as e:
                                        logger.warning(f"Failed to generate Stage 3 visualizations: {e}")

                        display_metrics = {k: f"{v:.4f}" for k, v in avg_metrics.items()
                                          if k != 'lr'}
                        progress_bar.set_postfix(display_metrics)

                    self.metric_tracker.reset()

                # Checkpoint
                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint()

                # Evaluation
                if self.val_dataloader is not None and \
                   self.global_step % self.config.eval_steps == 0:
                    self.evaluate()

    @torch.no_grad()
    def evaluate(self):
        """Evaluate robot selection performance with comprehensive metrics."""
        self.model.eval()

        # Reset metrics
        self.robot_metrics.reset()
        self.reasoning_metrics.reset()

        val_data = self.val_dataloader if self.val_dataloader else self.robot_dataloader

        for batch in tqdm(
            val_data,
            desc="Evaluating",
            disable=not is_main_process(),
        ):
            pixel_values = batch['pixel_values'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)

            robot_targets = batch.get('robot_target')
            multi_robot_targets = batch.get('multi_robot_target')

            if robot_targets is not None:
                robot_targets = robot_targets.to(self.device)
            if multi_robot_targets is not None:
                multi_robot_targets = multi_robot_targets.to(self.device)

            # Clamp targets during eval as well
            num_robots = getattr(self.model.module.config if hasattr(self.model, 'module') else self.model.config, 'num_robots', None)
            if num_robots is not None and robot_targets is not None:
                robot_targets = torch.clamp(robot_targets, 0, num_robots - 1)
                if multi_robot_targets is not None:
                    if multi_robot_targets.size(-1) > num_robots:
                        multi_robot_targets = multi_robot_targets[..., :num_robots]
                    elif multi_robot_targets.size(-1) < num_robots:
                        pad_width = num_robots - multi_robot_targets.size(-1)
                        pad_shape = list(multi_robot_targets.shape[:-1]) + [pad_width]
                        pad_tensor = torch.zeros(pad_shape, device=multi_robot_targets.device, dtype=multi_robot_targets.dtype)
                        multi_robot_targets = torch.cat([multi_robot_targets, pad_tensor], dim=-1)
                    multi_robot_targets = multi_robot_targets.clamp(0.0, 1.0)

            with get_autocast_context(self.config):
                outputs = self.model(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    attention_mask=attention_mask,
                    return_reasoning=True,
                )

                if 'robot_logits' in outputs and robot_targets is not None:
                    if num_robots is not None and outputs['robot_logits'].shape[-1] != num_robots:
                        outputs['robot_logits'] = outputs['robot_logits'][..., :num_robots]

                    robot_preds = outputs['robot_logits'].argmax(dim=-1)

                    # Get confidence scores (softmax probabilities)
                    confidences = torch.softmax(outputs['robot_logits'], dim=-1)
                    pred_confidences = confidences.gather(1, robot_preds.unsqueeze(1)).squeeze(1)

                    # Multi-robot predictions (if available)
                    multi_robot_preds = None
                    if 'multi_robot_logits' in outputs and multi_robot_targets is not None:
                        multi_robot_preds = torch.sigmoid(outputs['multi_robot_logits'])
                        if multi_robot_preds.size(-1) > num_robots:
                            multi_robot_preds = multi_robot_preds[..., :num_robots]

                    # Update comprehensive metrics
                    self.robot_metrics.update(
                        predictions=robot_preds,
                        targets=robot_targets,
                        confidences=pred_confidences,
                        multi_robot_preds=multi_robot_preds,
                        multi_robot_targets=multi_robot_targets,
                    )

        # Compute all metrics
        metrics = self.robot_metrics.compute()

        if is_main_process():
            logger.info(f"Validation: {metrics}")

            if self.wandb_logger is not None:
                self.wandb_logger.log(metrics, step=self.global_step)

                # Log confusion matrix visualization
                if hasattr(self.wandb_logger, 'log_robot_confusion_matrix'):
                    all_preds = self.robot_metrics.get_all_predictions()
                    all_targets = self.robot_metrics.get_all_targets()
                    if all_preds is not None and all_targets is not None:
                        self.wandb_logger.log_robot_confusion_matrix(
                            predictions=all_preds,
                            labels=all_targets,
                            step=self.global_step,
                        )

                # Log per-robot radar chart
                if hasattr(self.wandb_logger, 'log_robot_radar_chart'):
                    per_robot_metrics = self.robot_metrics.get_per_robot_metrics()
                    if per_robot_metrics:
                        self.wandb_logger.log_robot_radar_chart(
                            metrics=per_robot_metrics,
                            step=self.global_step,
                        )

                # Log calibration plot
                if hasattr(self.wandb_logger, 'log_calibration_plot'):
                    all_confidences = self.robot_metrics.get_all_confidences()
                    all_correct = self.robot_metrics.get_all_correct()
                    if all_confidences is not None and all_correct is not None:
                        self.wandb_logger.log_calibration_plot(
                            confidences=all_confidences,
                            correct=all_correct,
                            step=self.global_step,
                        )

        self.model.train()
        return metrics

    def save_checkpoint(self):
        """Save checkpoint."""
        output_dir = Path(self.config.output_dir) / f'checkpoint-{self.global_step}'

        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            config=self.config,
            step=self.global_step,
            metrics=self.metric_tracker.get_average(),
            output_dir=str(output_dir),
            scaler=self.scaler,
        )

        # Clean old checkpoints
        if is_main_process():
            checkpoints = sorted(
                Path(self.config.output_dir).glob('checkpoint-*'),
                key=lambda x: int(x.name.split('-')[1])
            )

            while len(checkpoints) > self.config.max_checkpoints:
                oldest = checkpoints.pop(0)
                import shutil
                shutil.rmtree(oldest)

    def train(self, robot_epochs: int = 20):
        """Run Stage 3 robot selection training."""
        if is_main_process() and self.carbon_tracker is not None:
            self.carbon_tracker.start()

        try:
            logger.info("Stage 3: Robot Fleet Selection Training")
            for epoch in range(robot_epochs):
                if hasattr(self.robot_dataloader.sampler, 'set_epoch'):
                    self.robot_dataloader.sampler.set_epoch(epoch)

                self.train_epoch(epoch)

                # Evaluate after each epoch
                if self.val_dataloader is not None:
                    self.evaluate()

                barrier()

            # Final save
            self.save_checkpoint()

        finally:
            if is_main_process():
                if self.carbon_tracker is not None:
                    emissions = self.carbon_tracker.stop()
                    logger.info(f"Total emissions: {emissions:.4f} kg CO2eq")

                if self.wandb_logger is not None:
                    self.wandb_logger.finish()

            # Note: Do NOT call cleanup_distributed() here
            # The process group should persist across stages
            # cleanup_distributed() should only be called at the end of all training


def run_stage3_training(
    model: EmberVLM,
    config: TrainingConfig,
    robot_data_dir: str,
    tokenizer: Any,
    robot_epochs: int = 20,
):
    """Run Stage 3 robot selection training."""
    robot_dataloader = get_robot_selection_dataloader(
        data_dir=robot_data_dir,
        tokenizer=tokenizer,
        batch_size=config.batch_size,
        split='train',
        distributed=config.distributed,
    )

    val_dataloader = get_robot_selection_dataloader(
        data_dir=robot_data_dir,
        tokenizer=tokenizer,
        batch_size=config.batch_size,
        split='val',
        distributed=False,
    )

    trainer = Stage3Trainer(
        model=model,
        config=config,
        robot_dataloader=robot_dataloader,
        val_dataloader=val_dataloader,
        tokenizer=tokenizer,
    )

    trainer.train(robot_epochs)


if __name__ == "__main__":
    import argparse
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument('--robot_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./outputs/stage3')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--robot_epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()

    config = TrainingConfig(
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    if args.checkpoint:
        model = EmberVLM.from_pretrained(args.checkpoint)
    else:
        model = EmberVLM()

    run_stage3_training(
        model=model,
        config=config,
        robot_data_dir=args.robot_dir,
        tokenizer=tokenizer,
        robot_epochs=args.robot_epochs,
    )
