"""
Stage 3: Incident Understanding & Robot Reasoning

Specializes the model for incident analysis using the Incidents1M dataset
and trains robot selection capabilities.
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
from embervlm.data.incident_loader import get_incident_dataloader
from embervlm.data.robot_loader import get_robot_selection_dataloader
from embervlm.monitoring.wandb_logger import WandbLogger
from embervlm.monitoring.carbon_tracker import CarbonTracker

logger = logging.getLogger(__name__)


class Stage3Trainer:
    """Trainer for Stage 3: Incident Understanding & Robot Reasoning."""

    def __init__(
        self,
        model: EmberVLM,
        config: TrainingConfig,
        incident_dataloader: DataLoader,
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

        # Model
        self.model = wrap_model_ddp(model, config, self.device)

        # Data loaders
        self.incident_dataloader = incident_dataloader
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

        # Logging
        self.wandb_logger = None
        self.carbon_tracker = None

        if is_main_process():
            self.wandb_logger = WandbLogger(
                project="embervlm",
                name="stage3_incidents",
                config=config.to_dict(),
            )
            self.carbon_tracker = CarbonTracker(output_dir=config.output_dir)

        self.metric_tracker = MetricTracker()
        self.global_step = 0

        if is_main_process():
            print_trainable_parameters(self.model)

    def train_incident_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Training step for incident understanding."""
        pixel_values = batch['pixel_values'].to(self.device)
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)
        incident_labels = batch.get('incident_labels')

        with get_autocast_context(self.config):
            outputs = self.model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs['loss']

            # Multi-label classification for incidents
            if incident_labels is not None:
                incident_labels = incident_labels.to(self.device)
                # Additional classification loss would go here

        return loss, {
            'incident_loss': loss.item(),
        }

    def train_robot_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Training step for robot selection."""
        pixel_values = batch['pixel_values'].to(self.device)
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)
        robot_targets = batch['robot_target'].to(self.device)

        with get_autocast_context(self.config):
            outputs = self.model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                labels=labels,
                robot_targets=robot_targets,
                return_reasoning=True,
            )

            loss = outputs['loss']

            # Robot selection accuracy
            if 'robot_logits' in outputs:
                robot_preds = outputs['robot_logits'].argmax(dim=-1)
                robot_acc = (robot_preds == robot_targets).float().mean()
            else:
                robot_acc = torch.tensor(0.0)

        return loss, {
            'robot_loss': loss.item(),
            'robot_accuracy': robot_acc.item(),
        }

    def train_epoch(self, epoch: int, phase: str = 'incident'):
        """Train for one epoch."""
        self.model.train()
        self.metric_tracker.reset()

        if phase == 'incident':
            dataloader = self.incident_dataloader
            train_fn = self.train_incident_step
            desc = f"Incident Epoch {epoch}"
        else:
            dataloader = self.robot_dataloader
            train_fn = self.train_robot_step
            desc = f"Robot Epoch {epoch}"

        progress_bar = tqdm(
            dataloader,
            desc=desc,
            disable=not is_main_process(),
        )

        for batch_idx, batch in enumerate(progress_bar):
            loss, metrics = train_fn(batch)

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
                        self.wandb_logger.log(avg_metrics, step=self.global_step)

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
        """Evaluate robot selection performance."""
        self.model.eval()
        eval_metrics = MetricTracker()

        # Create combined evaluation data
        val_data = self.val_dataloader if self.val_dataloader else self.robot_dataloader

        correct = 0
        total = 0

        for batch in tqdm(
            val_data,
            desc="Evaluating",
            disable=not is_main_process(),
        ):
            pixel_values = batch['pixel_values'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)

            robot_targets = batch.get('robot_target')
            if robot_targets is not None:
                robot_targets = robot_targets.to(self.device)

            with get_autocast_context(self.config):
                outputs = self.model(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    attention_mask=attention_mask,
                    return_reasoning=True,
                )

                if 'robot_logits' in outputs and robot_targets is not None:
                    robot_preds = outputs['robot_logits'].argmax(dim=-1)
                    correct += (robot_preds == robot_targets).sum().item()
                    total += robot_targets.size(0)

                    # Per-robot accuracy
                    for i in range(outputs['robot_logits'].size(-1)):
                        mask = robot_targets == i
                        if mask.sum() > 0:
                            class_correct = ((robot_preds == robot_targets) & mask).sum()
                            eval_metrics.update({
                                f'robot_{i}_acc': (class_correct / mask.sum()).item()
                            })

        if total > 0:
            overall_acc = correct / total
            eval_metrics.update({'val_robot_accuracy': overall_acc})

        avg_metrics = eval_metrics.get_average()

        if is_main_process():
            self.wandb_logger.log(avg_metrics, step=self.global_step)
            logger.info(f"Evaluation: {avg_metrics}")

        self.model.train()
        return avg_metrics

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

    def train(self, incident_epochs: int = 10, robot_epochs: int = 20):
        """Run full Stage 3 training."""
        if is_main_process() and self.carbon_tracker is not None:
            self.carbon_tracker.start()

        try:
            # Phase 1: Incident understanding
            logger.info("Phase 1: Incident Understanding")
            for epoch in range(incident_epochs):
                if hasattr(self.incident_dataloader.sampler, 'set_epoch'):
                    self.incident_dataloader.sampler.set_epoch(epoch)

                self.train_epoch(epoch, phase='incident')
                barrier()

            # Phase 2: Robot selection
            logger.info("Phase 2: Robot Selection")
            for epoch in range(robot_epochs):
                if hasattr(self.robot_dataloader.sampler, 'set_epoch'):
                    self.robot_dataloader.sampler.set_epoch(epoch)

                self.train_epoch(epoch, phase='robot')

                # Evaluate after each robot epoch
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

            cleanup_distributed()


def run_stage3_training(
    model: EmberVLM,
    config: TrainingConfig,
    incident_data_dir: str,
    robot_data_dir: str,
    tokenizer: Any,
    incident_epochs: int = 10,
    robot_epochs: int = 20,
):
    """Run Stage 3 training."""
    incident_dataloader = get_incident_dataloader(
        data_dir=incident_data_dir,
        tokenizer=tokenizer,
        batch_size=config.batch_size,
        split='train',
        distributed=config.distributed,
    )

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
        incident_dataloader=incident_dataloader,
        robot_dataloader=robot_dataloader,
        val_dataloader=val_dataloader,
        tokenizer=tokenizer,
    )

    trainer.train(incident_epochs, robot_epochs)


if __name__ == "__main__":
    import argparse
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument('--incident_dir', type=str, required=True)
    parser.add_argument('--robot_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./outputs/stage3')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--incident_epochs', type=int, default=10)
    parser.add_argument('--robot_epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=2e-4)
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
        incident_data_dir=args.incident_dir,
        robot_data_dir=args.robot_dir,
        tokenizer=tokenizer,
        incident_epochs=args.incident_epochs,
        robot_epochs=args.robot_epochs,
    )

