"""
Stage 4: Chain-of-Thought Reasoning Integration

Integrates DeepSeek-R1 style reasoning with two-phase training:
1. Train reasoning heads with frozen backbone
2. Joint fine-tuning with reduced learning rate
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
    unwrap_model,
    save_checkpoint,
    MetricTracker,
    print_trainable_parameters,
)
from embervlm.data.loaders import get_reasoning_dataloader
from embervlm.monitoring.wandb_logger import WandbLogger
from embervlm.monitoring.carbon_tracker import CarbonTracker

logger = logging.getLogger(__name__)


class ReasoningConsistencyLoss(nn.Module):
    """Loss for ensuring consistent reasoning chains."""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        reasoning_chain: torch.Tensor,
        target_chain: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute reasoning consistency loss.

        Args:
            reasoning_chain: Generated reasoning [B, steps, seq_len, dim]
            target_chain: Optional target reasoning

        Returns:
            Consistency loss
        """
        # Encourage smooth transitions between steps
        if reasoning_chain.dim() == 4:
            step_diffs = reasoning_chain[:, 1:] - reasoning_chain[:, :-1]
            smoothness_loss = torch.mean(step_diffs ** 2)
        else:
            smoothness_loss = torch.tensor(0.0, device=reasoning_chain.device)

        # If target is provided, compute MSE
        if target_chain is not None:
            target_loss = F.mse_loss(reasoning_chain, target_chain)
            return smoothness_loss + target_loss

        return smoothness_loss


class Stage4Trainer:
    """Trainer for Stage 4: Chain-of-Thought Reasoning."""

    def __init__(
        self,
        model: EmberVLM,
        config: TrainingConfig,
        train_dataloader: DataLoader,
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
        model = unwrap_model(model)

        # Ensure model is on correct device before DDP
        model = model.to(self.device)

        # Model
        self.model = wrap_model_ddp(model, config, self.device)

        # Data
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        # Phase tracking
        self.current_phase = 1

        # Optimizer (will be re-created for each phase)
        self.optimizer = None
        self.scheduler = None
        self.scaler = get_grad_scaler(config)

        # Losses
        self.reasoning_consistency_loss = ReasoningConsistencyLoss()

        # Logging - only main process initializes W&B and carbon tracker
        self.wandb_logger = None
        self.carbon_tracker = None

        if is_main_process():
            logger.info("Initializing W&B logger (main process)...")
            try:
                self.wandb_logger = WandbLogger(
                    project="embervlm",
                    name="stage4_reasoning",
                    config=config.to_dict(),
                )
                logger.info("W&B logger initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize W&B logger: {e}")
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
        self.global_step = 0

    def _freeze_backbone(self):
        """Freeze backbone, only train reasoning heads."""
        model = unwrap_model(self.model)

        # Freeze vision encoder
        for param in model.vision_encoder.parameters():
            param.requires_grad = False

        # Freeze language model
        for param in model.language_model.parameters():
            param.requires_grad = False

        # Freeze fusion module
        for param in model.fusion_module.parameters():
            param.requires_grad = False

        # Keep reasoning module trainable
        if hasattr(model, 'reasoning_module'):
            for param in model.reasoning_module.parameters():
                param.requires_grad = True

        if is_main_process():
            logger.info("Phase 1: Backbone frozen, training reasoning heads only")
            print_trainable_parameters(model)

    def _unfreeze_all(self):
        """Unfreeze all trainable parameters."""
        model = unwrap_model(self.model)

        # Unfreeze last layer of language model
        for param in model.language_model.model.transformer.h[-1].parameters():
            param.requires_grad = True
        for param in model.language_model.model.transformer.ln_f.parameters():
            param.requires_grad = True
        for param in model.language_model.model.lm_head.parameters():
            param.requires_grad = True

        # Unfreeze fusion module
        for param in model.fusion_module.parameters():
            param.requires_grad = True

        # Keep reasoning module trainable
        if hasattr(model, 'reasoning_module'):
            for param in model.reasoning_module.parameters():
                param.requires_grad = True

        if is_main_process():
            logger.info("Phase 2: Joint fine-tuning")
            print_trainable_parameters(model)

    def _setup_phase(self, phase: int, lr: float, num_steps: int):
        """Setup optimizer for current phase."""
        self.current_phase = phase

        if phase == 1:
            self._freeze_backbone()
        else:
            self._unfreeze_all()

        # Create new optimizer
        phase_config = TrainingConfig(
            learning_rate=lr,
            min_learning_rate=lr / 10,
            num_training_steps=num_steps,
            warmup_steps=min(100, num_steps // 10),
            weight_decay=self.config.weight_decay,
            beta1=self.config.beta1,
            beta2=self.config.beta2,
            eps=self.config.eps,
            optimizer=self.config.optimizer,
            scheduler=self.config.scheduler,
        )

        self.optimizer = get_optimizer(self.model, phase_config)
        self.scheduler = get_scheduler(self.optimizer, phase_config)

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        pixel_values = batch['pixel_values'].to(self.device)
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)

        robot_targets = batch.get('robot_target')
        if robot_targets is not None:
            robot_targets = robot_targets.to(self.device)

        reasoning_targets = batch.get('reasoning_chain')

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

            # Reasoning consistency loss
            if 'reasoning_chain' in outputs:
                reasoning_chain = outputs['reasoning_chain']

                if reasoning_targets is not None:
                    reasoning_targets = reasoning_targets.to(self.device)
                    consistency_loss = self.reasoning_consistency_loss(
                        reasoning_chain, reasoning_targets
                    )
                else:
                    consistency_loss = self.reasoning_consistency_loss(reasoning_chain)

                loss = loss + 0.1 * consistency_loss
            else:
                consistency_loss = torch.tensor(0.0)

        metrics = {
            'loss': loss.item(),
            'consistency_loss': consistency_loss.item() if isinstance(consistency_loss, torch.Tensor) else 0.0,
        }

        if 'robot_logits' in outputs and robot_targets is not None:
            robot_preds = outputs['robot_logits'].argmax(dim=-1)
            robot_acc = (robot_preds == robot_targets).float().mean()
            metrics['robot_accuracy'] = robot_acc.item()

        return loss, metrics

    def train_epoch(self, epoch: int):
        """Train for one epoch."""
        self.model.train()
        self.metric_tracker.reset()

        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Phase {self.current_phase} Epoch {epoch}",
            disable=not is_main_process(),
        )

        for batch_idx, batch in enumerate(progress_bar):
            loss, metrics = self.train_step(batch)

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
                metrics['phase'] = self.current_phase
                self.metric_tracker.update(metrics)

                # Logging
                if self.global_step % self.config.log_steps == 0:
                    avg_metrics = self.metric_tracker.get_average()

                    if is_main_process():
                        if self.wandb_logger is not None:
                            self.wandb_logger.log(avg_metrics, step=self.global_step)

                        display = {k: f"{v:.4f}" for k, v in avg_metrics.items()
                                  if k not in ['lr', 'phase']}
                        progress_bar.set_postfix(display)

                    self.metric_tracker.reset()

                # Checkpoint
                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint()

    @torch.no_grad()
    def evaluate(self):
        """Evaluate reasoning quality."""
        if self.val_dataloader is None:
            return {}

        self.model.eval()
        eval_metrics = MetricTracker()

        for batch in tqdm(
            self.val_dataloader,
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
                    robot_acc = (robot_preds == robot_targets).float().mean()
                    eval_metrics.update({'robot_accuracy': robot_acc.item()})

                if 'robot_confidence' in outputs:
                    confidence = outputs['robot_confidence'].mean()
                    eval_metrics.update({'confidence': confidence.item()})

                if 'plan_coherence' in outputs:
                    coherence = outputs['plan_coherence'].mean()
                    eval_metrics.update({'plan_coherence': coherence.item()})

        avg_metrics = eval_metrics.get_average()
        avg_metrics = {f'val_{k}': v for k, v in avg_metrics.items()}

        if is_main_process():
            if self.wandb_logger is not None:
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

    def train(
        self,
        phase1_epochs: int = 5,
        phase2_epochs: int = 5,
        phase1_lr: float = 1e-4,
        phase2_lr: float = 5e-5,
    ):
        """Run full Stage 4 training."""
        if is_main_process() and self.carbon_tracker is not None:
            self.carbon_tracker.start()

        try:
            # Calculate steps
            steps_per_epoch = len(self.train_dataloader)
            phase1_steps = phase1_epochs * steps_per_epoch
            phase2_steps = phase2_epochs * steps_per_epoch

            # Phase 1: Train reasoning heads with frozen backbone
            logger.info("="*50)
            logger.info("Phase 1: Training reasoning heads (frozen backbone)")
            logger.info("="*50)

            self._setup_phase(1, phase1_lr, phase1_steps)

            for epoch in range(phase1_epochs):
                if hasattr(self.train_dataloader.sampler, 'set_epoch'):
                    self.train_dataloader.sampler.set_epoch(epoch)

                self.train_epoch(epoch)
                self.evaluate()
                barrier()

            # Phase 2: Joint fine-tuning
            logger.info("="*50)
            logger.info("Phase 2: Joint fine-tuning")
            logger.info("="*50)

            self._setup_phase(2, phase2_lr, phase2_steps)

            for epoch in range(phase2_epochs):
                if hasattr(self.train_dataloader.sampler, 'set_epoch'):
                    self.train_dataloader.sampler.set_epoch(phase1_epochs + epoch)

                self.train_epoch(epoch)
                self.evaluate()
                barrier()

            # Final checkpoint
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


def run_stage4_training(
    model: EmberVLM,
    config: TrainingConfig,
    data_dir: str,
    tokenizer: Any,
    phase1_epochs: int = 5,
    phase2_epochs: int = 5,
    phase1_lr: float = 1e-4,
    phase2_lr: float = 5e-5,
):
    """Run Stage 4 training."""
    train_dataloader = get_reasoning_dataloader(
        data_dir=data_dir,
        tokenizer=tokenizer,
        batch_size=config.batch_size,
        split='train',
        distributed=config.distributed,
    )

    val_dataloader = get_reasoning_dataloader(
        data_dir=data_dir,
        tokenizer=tokenizer,
        batch_size=config.batch_size,
        split='val',
        distributed=False,
    )

    trainer = Stage4Trainer(
        model=model,
        config=config,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        tokenizer=tokenizer,
    )

    trainer.train(
        phase1_epochs=phase1_epochs,
        phase2_epochs=phase2_epochs,
        phase1_lr=phase1_lr,
        phase2_lr=phase2_lr,
    )


if __name__ == "__main__":
    import argparse
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./outputs/stage4')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--phase1_epochs', type=int, default=5)
    parser.add_argument('--phase2_epochs', type=int, default=5)
    parser.add_argument('--phase1_lr', type=float, default=1e-4)
    parser.add_argument('--phase2_lr', type=float, default=5e-5)
    args = parser.parse_args()

    config = TrainingConfig(
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )

    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    if args.checkpoint:
        model = EmberVLM.from_pretrained(args.checkpoint)
    else:
        model = EmberVLM()

    run_stage4_training(
        model=model,
        config=config,
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        phase1_epochs=args.phase1_epochs,
        phase2_epochs=args.phase2_epochs,
        phase1_lr=args.phase1_lr,
        phase2_lr=args.phase2_lr,
    )

