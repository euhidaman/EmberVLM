"""
Stage 1: Visual-Language Alignment Training

Aligns RepViT vision features with TinyLLM text space using
contrastive learning and image captioning.
"""

import os
import math
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from embervlm.models import EmberVLM
from embervlm.training.train_utils import (
    TrainingConfig,
    setup_distributed,
    cleanup_distributed,
    is_main_process,
    get_rank,
    get_world_size,
    barrier,
    set_seed,
    get_optimizer,
    get_scheduler,
    get_grad_scaler,
    get_autocast_context,
    wrap_model_ddp,
    save_checkpoint,
    load_checkpoint,
    MetricTracker,
    print_trainable_parameters,
    push_checkpoint_to_hub,
)
from embervlm.data.loaders import get_alignment_dataloader
from embervlm.monitoring.wandb_logger import EnhancedWandbLogger
from embervlm.monitoring.carbon_tracker import CarbonTracker

logger = logging.getLogger(__name__)

# Try to import stage visualizer
try:
    from embervlm.monitoring.stage_visualizations import Stage1Visualizer
    HAS_STAGE_VIZ = True
except ImportError:
    HAS_STAGE_VIZ = False
    Stage1Visualizer = None


class ContrastiveLoss(nn.Module):
    """Image-text contrastive loss (CLIP-style)."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1 / temperature)))

    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute contrastive loss.

        Args:
            image_features: [B, D] normalized image features
            text_features: [B, D] normalized text features

        Returns:
            Loss and metrics dictionary
        """
        # Normalize features
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        # Compute logits
        logit_scale = self.logit_scale.exp().clamp(max=100)
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logits_per_image.T

        # Labels (diagonal is positive)
        batch_size = image_features.size(0)
        labels = torch.arange(batch_size, device=image_features.device)

        # Cross entropy loss
        loss_i2t = F.cross_entropy(logits_per_image, labels)
        loss_t2i = F.cross_entropy(logits_per_text, labels)
        loss = (loss_i2t + loss_t2i) / 2

        # Compute accuracy
        with torch.no_grad():
            pred_i2t = logits_per_image.argmax(dim=-1)
            pred_t2i = logits_per_text.argmax(dim=-1)
            acc_i2t = (pred_i2t == labels).float().mean()
            acc_t2i = (pred_t2i == labels).float().mean()

        metrics = {
            'contrastive_loss': loss.item(),
            'loss_i2t': loss_i2t.item(),
            'loss_t2i': loss_t2i.item(),
            'acc_i2t': acc_i2t.item(),
            'acc_t2i': acc_t2i.item(),
            'logit_scale': logit_scale.item(),
        }

        return loss, metrics


class Stage1Trainer:
    """Trainer for Stage 1: Visual-Language Alignment."""

    def __init__(
        self,
        model: EmberVLM,
        config: TrainingConfig,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        tokenizer: Any = None,
        hub_repo_id: Optional[str] = None,
        vision_backbone: str = "repvit",
        language_backbone: str = "tinyllm",
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.hub_repo_id = hub_repo_id
        self.vision_backbone = vision_backbone
        self.language_backbone = language_backbone

        # Setup distributed
        self.rank, self.local_rank, self.world_size = setup_distributed()
        self.device = torch.device(f'cuda:{self.local_rank}')

        # Set seed
        set_seed(config.seed, self.rank)

        # Unwrap model if it was previously wrapped with DDP
        from embervlm.training.train_utils import unwrap_model
        model = unwrap_model(model)

        # Validate embedding size matches tokenizer
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
                    logger.error(f"❌ Token embedding size mismatch! Tokenizer: {required_vocab_size}, Model: {current_vocab_size}")
                    raise ValueError(f"Embedding size ({current_vocab_size}) doesn't match tokenizer ({required_vocab_size})")
                else:
                    logger.info(f"✓ Embedding size validation passed: {current_vocab_size} tokens")

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

        # Data
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        # Optimizer and scheduler
        self.optimizer = get_optimizer(self.model, config)
        self.scheduler = get_scheduler(self.optimizer, config)
        self.scaler = get_grad_scaler(config)

        # Losses
        logger.info("Initializing contrastive loss...")
        self.contrastive_loss = ContrastiveLoss(temperature=0.07).to(self.device)
        logger.info("Contrastive loss initialized")

        # Logging - only main process initializes W&B and carbon tracker
        self.wandb_logger = None
        self.carbon_tracker = None

        if is_main_process():
            logger.info("Initializing Enhanced W&B logger with visualizations (main process)...")
            try:
                wandb_project = config.wandb_project if hasattr(config, 'wandb_project') and config.wandb_project else "embervlm"
                self.wandb_logger = EnhancedWandbLogger(
                    project=wandb_project,
                    name="stage1_alignment",
                    config=config.to_dict(),
                    output_dir=str(Path(config.output_dir) / 'visualizations'),
                )
                logger.info(f"Enhanced W&B logger initialized with project: {wandb_project}")
            except Exception as e:
                logger.warning(f"Failed to initialize W&B logger: {e}")
                self.wandb_logger = None

            logger.info("Initializing carbon tracker...")
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

        # Metrics
        self.metric_tracker = MetricTracker()
        self.global_step = 0

        # Stage 1 specific visualizer
        self.stage_visualizer = None
        if is_main_process() and HAS_STAGE_VIZ:
            try:
                self.stage_visualizer = Stage1Visualizer(
                    output_dir=str(Path(config.output_dir) / 'visualizations')
                )
                logger.info("✓ Stage1Visualizer initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Stage1Visualizer: {e}")

        # Store embeddings for visualization (updated each batch)
        self.last_image_embeds = None
        self.last_text_embeds = None

        # Print info
        if is_main_process():
            print_trainable_parameters(self.model)

        logger.info(f"[Rank {self.rank}] Stage1Trainer initialization complete")

    def train_epoch(self, epoch: int):
        """Train for one epoch."""
        self.model.train()
        self.metric_tracker.reset()

        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {epoch}",
            disable=not is_main_process(),
        )

        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            pixel_values = batch['pixel_values'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch.get('labels')
            if labels is not None:
                labels = labels.to(self.device)

            # CRITICAL: Validate input_ids and labels are within embedding bounds
            # This prevents cryptic CUDA index out of bounds errors
            model_ref = self.model.module if hasattr(self.model, 'module') else self.model

            vocab_size = None
            if hasattr(model_ref.language_model, 'get_input_embeddings'):
                vocab_size = model_ref.language_model.get_input_embeddings().weight.shape[0]
            elif hasattr(model_ref.language_model, 'model'):
                if hasattr(model_ref.language_model.model, 'get_input_embeddings'):
                    vocab_size = model_ref.language_model.model.get_input_embeddings().weight.shape[0]

            if vocab_size is not None:
                # Validate and clamp input_ids
                max_token_id = input_ids.max().item()
                min_token_id = input_ids.min().item()
                if max_token_id >= vocab_size or min_token_id < 0:
                    if max_token_id >= vocab_size:
                        logger.error(f"❌ input_ids max={max_token_id} >= vocab_size={vocab_size}")
                    if min_token_id < 0:
                        logger.error(f"❌ input_ids min={min_token_id} < 0")
                    input_ids = torch.clamp(input_ids, min=0, max=vocab_size - 1)
                    logger.warning(f"   Token IDs clamped to valid range [0, {vocab_size - 1}]")

                # Validate and clamp labels if present (preserve -100 ignore index)
                if labels is not None:
                    valid_labels_mask = labels != -100
                    if valid_labels_mask.any():
                        valid_labels = labels[valid_labels_mask]
                        max_label = valid_labels.max().item()
                        min_label = valid_labels.min().item()
                        if max_label >= vocab_size or min_label < 0:
                            labels = torch.where(
                                valid_labels_mask,
                                torch.clamp(labels, min=0, max=vocab_size - 1),
                                labels
                            )
                            logger.warning(f"   Labels clamped to valid range [0, {vocab_size - 1}]")

            # Forward pass
            with get_autocast_context(self.config):
                # Get visual features
                vision_output = self.model.module.encode_image(pixel_values) \
                    if hasattr(self.model, 'module') else self.model.encode_image(pixel_values)
                visual_tokens = vision_output['visual_tokens']

                # Fuse to language space
                fused_visual = self.model.module.fuse_features(visual_tokens) \
                    if hasattr(self.model, 'module') else self.model.fuse_features(visual_tokens)

                # Get text embeddings
                model_ref = self.model.module if hasattr(self.model, 'module') else self.model
                text_embeds = model_ref.language_model.embed_tokens(input_ids)

                # Pool features for contrastive loss
                image_pooled = fused_visual.mean(dim=1)
                text_pooled = (text_embeds * attention_mask.unsqueeze(-1)).sum(dim=1) / \
                              attention_mask.sum(dim=1, keepdim=True).clamp(min=1)

                # Contrastive loss
                contrastive_loss, contrastive_metrics = self.contrastive_loss(
                    image_pooled, text_pooled
                )

                # Store embeddings for visualization (detach to avoid memory issues)
                self.last_image_embeds = image_pooled.detach()
                self.last_text_embeds = text_pooled.detach()

                # Captioning loss (language modeling)
                inputs_embeds, _ = model_ref.prepare_inputs_embeds(
                    input_ids, pixel_values
                )

                # Adjust labels to match inputs_embeds length (account for visual tokens)
                # Note: The language model will internally shift labels for causal LM
                if labels.size(1) != inputs_embeds.size(1):
                    batch_size = labels.size(0)
                    num_visual = inputs_embeds.size(1) - labels.size(1)

                    # Create adjusted labels with -100 at visual token positions
                    # Visual tokens are inserted at the beginning (position 0)
                    adjusted_labels = torch.full(
                        (batch_size, inputs_embeds.size(1)),
                        -100,
                        dtype=labels.dtype,
                        device=labels.device
                    )

                    # Visual tokens get -100 (positions 0 to num_visual-1)
                    # Copy original labels after visual tokens
                    adjusted_labels[:, num_visual:] = labels
                else:
                    adjusted_labels = labels

                lm_outputs = model_ref.language_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=torch.ones(
                        inputs_embeds.size(0), inputs_embeds.size(1),
                        device=self.device, dtype=torch.long
                    ),
                    labels=adjusted_labels,
                )

                captioning_loss = lm_outputs.get('loss', torch.tensor(0.0, device=self.device))

                # Total loss
                loss = 0.5 * contrastive_loss + 0.5 * captioning_loss

            # Backward pass
            if self.config.gradient_accumulation_steps > 1:
                loss = loss / self.config.gradient_accumulation_steps

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation
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

                # Update metrics
                metrics = {
                    'loss': loss.item() * self.config.gradient_accumulation_steps,
                    'contrastive_loss': contrastive_metrics['contrastive_loss'],
                    'captioning_loss': captioning_loss.item() if isinstance(captioning_loss, torch.Tensor) else 0.0,
                    'lr': self.scheduler.get_last_lr()[0],
                    **contrastive_metrics,
                }
                self.metric_tracker.update(metrics)

                # Logging
                if self.global_step % self.config.log_steps == 0:
                    avg_metrics = self.metric_tracker.get_average()

                    if is_main_process():
                        if self.wandb_logger is not None:
                            # Use enhanced logging if available
                            if hasattr(self.wandb_logger, 'log_with_visualization'):
                                self.wandb_logger.log_with_visualization(
                                    avg_metrics,
                                    step=self.global_step,
                                    stage_name="stage1",
                                )
                            else:
                                self.wandb_logger.log(avg_metrics, step=self.global_step)

                            # Stage 1 specific visualizations every 500 steps
                            if self.global_step % 500 == 0:
                                logger.info(f"[Stage1] Step {self.global_step}: Attempting visualizations...")
                                logger.info(f"  stage_visualizer: {self.stage_visualizer is not None}")
                                logger.info(f"  last_image_embeds: {self.last_image_embeds is not None}")
                                logger.info(f"  last_text_embeds: {self.last_text_embeds is not None}")

                                if self.stage_visualizer is not None:
                                    if self.last_image_embeds is not None and self.last_text_embeds is not None:
                                        try:
                                            logger.info(f"  Generating similarity matrix...")
                                            _, sim_img = self.stage_visualizer.plot_similarity_matrix(
                                                self.last_image_embeds,
                                                self.last_text_embeds,
                                                self.global_step
                                            )
                                            self.wandb_logger.log_image(
                                                "stage1/similarity_matrix", sim_img, step=self.global_step
                                            )
                                            logger.info(f"  ✓ Logged similarity matrix to W&B")

                                            logger.info(f"  Generating t-SNE...")
                                            _, tsne_img = self.stage_visualizer.plot_embedding_tsne(
                                                self.last_image_embeds,
                                                self.last_text_embeds,
                                                self.global_step,
                                                n_samples=min(100, self.last_image_embeds.size(0))
                                            )
                                            self.wandb_logger.log_image(
                                                "stage1/embedding_tsne", tsne_img, step=self.global_step
                                            )
                                            logger.info(f"  ✓ Logged t-SNE to W&B")
                                        except Exception as e:
                                            logger.error(f"  ✗ Failed to generate Stage 1 visualizations: {e}", exc_info=True)
                                    else:
                                        logger.warning(f"  Skipping visualizations: embeddings not available")
                                else:
                                    logger.warning(f"  Skipping visualizations: stage_visualizer is None")

                            # Log loss curve and metrics summary every 100 steps
                            if self.global_step % 100 == 0:
                                try:
                                    # Log detailed loss components as charts
                                    self.wandb_logger.log({
                                        'stage1/contrastive_loss_chart': self.wandb_logger.wandb.plot.line_series(
                                            xs=[list(range(len(self.metric_tracker.history.get('contrastive_loss', []))))],
                                            ys=[self.metric_tracker.history.get('contrastive_loss', [])],
                                            keys=['Contrastive'],
                                            title='Stage 1 Contrastive Loss',
                                            xname='Step'
                                        ),
                                    }, step=self.global_step) if hasattr(self.metric_tracker, 'history') and self.metric_tracker.history.get('contrastive_loss') else None
                                except Exception as e:
                                    pass  # Silently ignore chart errors

                            # Gradient distribution every 500 steps
                            if self.global_step % 500 == 0 and hasattr(self.wandb_logger, 'log_gradient_distribution'):
                                gradients = {}
                                for name, param in self.model.named_parameters():
                                    if param.grad is not None:
                                        gradients[name.split('.')[-1]] = param.grad
                                if gradients:
                                    self.wandb_logger.log_gradient_distribution(
                                        gradients, self.global_step, "stage1"
                                    )

                        progress_bar.set_postfix({
                            'loss': f"{avg_metrics['loss']:.4f}",
                            'acc': f"{avg_metrics.get('acc_i2t', 0):.3f}",
                        })

                    self.metric_tracker.reset()

                # Checkpointing
                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint()

                # Evaluation
                if self.val_dataloader is not None and \
                   self.global_step % self.config.eval_steps == 0:
                    self.evaluate(eval_step=self.global_step)

    @torch.no_grad()
    def evaluate(self, eval_step: Optional[int] = None):
        """Evaluate on validation set."""
        if eval_step is None:
            eval_step = self.global_step

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

            with get_autocast_context(self.config):
                # Get features
                model_ref = self.model.module if hasattr(self.model, 'module') else self.model
                vision_output = model_ref.encode_image(pixel_values)
                visual_tokens = vision_output['visual_tokens']
                fused_visual = model_ref.fuse_features(visual_tokens)
                text_embeds = model_ref.language_model.embed_tokens(input_ids)

                # Pool
                image_pooled = fused_visual.mean(dim=1)
                text_pooled = (text_embeds * attention_mask.unsqueeze(-1)).sum(dim=1) / \
                              attention_mask.sum(dim=1, keepdim=True).clamp(min=1)

                # Contrastive metrics
                _, metrics = self.contrastive_loss(image_pooled, text_pooled)
                eval_metrics.update(metrics)

        avg_metrics = eval_metrics.get_average()
        avg_metrics = {f'val_{k}': v for k, v in avg_metrics.items()}

        if is_main_process():
            if self.wandb_logger is not None:
                self.wandb_logger.log(avg_metrics, step=eval_step)
            logger.info(f"Validation metrics: {avg_metrics}")

        self.model.train()
        return avg_metrics

    def save_checkpoint(self):
        """Save training checkpoint."""
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

    def train(self, num_epochs: int):
        """Run full training."""
        if is_main_process() and self.carbon_tracker is not None:
            self.carbon_tracker.start()

        try:
            for epoch in range(num_epochs):
                if hasattr(self.train_dataloader.sampler, 'set_epoch'):
                    self.train_dataloader.sampler.set_epoch(epoch)

                self.train_epoch(epoch)

                # Validation at end of epoch
                if self.val_dataloader is not None:
                    self.evaluate()

                # Save at end of epoch
                self.save_checkpoint()

                # Push to HuggingFace Hub after each epoch
                if is_main_process() and self.hub_repo_id:
                    try:
                        metrics = {
                            'loss': self.metric_tracker.get_average().get('loss', 0.0),
                            'contrastive_loss': self.metric_tracker.get_average().get('contrastive_loss', 0.0),
                            'captioning_loss': self.metric_tracker.get_average().get('captioning_loss', 0.0),
                        }
                        
                        carbon_emissions = None
                        if self.carbon_tracker is not None:
                            carbon_emissions = self.carbon_tracker.get_emissions()
                        
                        push_checkpoint_to_hub(
                            model=self.model,
                            tokenizer=self.tokenizer,
                            repo_id=self.hub_repo_id,
                            epoch=epoch + 1,
                            stage="stage1",
                            metrics=metrics,
                            vision_backbone=self.vision_backbone,
                            language_backbone=self.language_backbone,
                            carbon_emissions=carbon_emissions,
                        )
                    except Exception as e:
                        logger.warning(f"Failed to push to HuggingFace Hub: {e}")

                barrier()

        except Exception as e:
            # Log error to WandB before crashing
            error_msg = f"Training failed at epoch {epoch}, step {self.global_step}: {str(e)}"
            logger.error(error_msg)

            if is_main_process() and self.wandb_logger is not None:
                self.wandb_logger.log({
                    'error': error_msg,
                    'error_type': type(e).__name__,
                    'failed_at_step': self.global_step,
                }, step=self.global_step)

            raise  # Re-raise to preserve stack trace

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


def run_stage1_training(
    model: EmberVLM,
    config: TrainingConfig,
    data_dir: str,
    tokenizer: Any,
    num_epochs: int = 3,
    hub_repo_id: Optional[str] = None,
    vision_backbone: str = "repvit",
    language_backbone: str = "tinyllm",
):
    """
    Run Stage 1 training.

    Args:
        model: EmberVLM model
        config: Training configuration
        data_dir: Directory with training data
        tokenizer: Tokenizer
        num_epochs: Number of training epochs
        hub_repo_id: HuggingFace Hub repo ID for epoch-level pushes
        vision_backbone: Vision backbone name
        language_backbone: Language backbone name
    """
    # Create data loaders
    train_dataloader = get_alignment_dataloader(
        data_dir=data_dir,
        tokenizer=tokenizer,
        batch_size=config.batch_size,
        split='train',
        distributed=config.distributed,
    )

    val_dataloader = get_alignment_dataloader(
        data_dir=data_dir,
        tokenizer=tokenizer,
        batch_size=config.batch_size,
        split='val',
        distributed=False,
    )

    # Create trainer
    trainer = Stage1Trainer(
        hub_repo_id=hub_repo_id,
        vision_backbone=vision_backbone,
        language_backbone=language_backbone,
        model=model,
        config=config,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        tokenizer=tokenizer,
    )

    # Train
    trainer.train(num_epochs)


if __name__ == "__main__":
    import argparse
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./outputs/stage1')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=2e-4)
    args = parser.parse_args()

    # Config
    config = TrainingConfig(
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Model
    model = EmberVLM()

    # Train
    run_stage1_training(
        model=model,
        config=config,
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        num_epochs=args.epochs,
    )

