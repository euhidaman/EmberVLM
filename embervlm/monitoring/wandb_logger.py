"""
Weights & Biases Logger for EmberVLM

Provides comprehensive logging for training metrics,
model artifacts, and visualizations.
"""

import os
import threading
from typing import Optional, Dict, Any, List, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def _init_wandb_with_timeout(timeout=30, **kwargs):
    """
    Initialize W&B with a timeout to prevent hanging.

    Args:
        timeout: Maximum seconds to wait for initialization
        **kwargs: Arguments to pass to wandb.init()

    Returns:
        W&B run object or None if timeout/failure
    """
    import wandb

    result = [None]
    exception = [None]

    def init_wandb():
        try:
            result[0] = wandb.init(**kwargs)
        except Exception as e:
            exception[0] = e

    thread = threading.Thread(target=init_wandb, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        logger.error(f"W&B initialization timed out after {timeout}s")
        return None

    if exception[0]:
        raise exception[0]

    return result[0]


class WandbLogger:
    """
    Weights & Biases logger for EmberVLM training.

    Features:
    - Training metrics logging
    - Model checkpointing
    - Gradient visualization
    - Attention heatmaps
    - Custom dashboards
    """

    def __init__(
        self,
        project: str = "embervlm",
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        entity: Optional[str] = None,
        resume: bool = False,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
        mode: str = "online",
        **kwargs,  # Accept extra kwargs like output_dir for compatibility
    ):
        self.project = project
        self.name = name
        self.config = config or {}
        self.entity = entity
        self.enabled = True

        # Check if W&B is disabled via environment variable
        import os as _os
        if _os.environ.get('DISABLE_WANDB', '').lower() in ('1', 'true', 'yes'):
            logger.info("W&B disabled via DISABLE_WANDB environment variable")
            self.wandb = None
            self.run = None
            self.enabled = False
            return

        try:
            import wandb
            self.wandb = wandb

            # Initialize run with timeout protection and fork-safe settings
            logger.info(f"Initializing W&B run: {name}")

            # Use thread-safe initialization and disable certain features that can hang
            wandb_settings = wandb.Settings(
                start_method="thread",
                _disable_stats=True,  # Disable system stats collection
                _disable_meta=True,   # Disable metadata collection
            )

            # Set timeout environment variable
            import os as _os
            _os.environ.setdefault('WANDB_INIT_TIMEOUT', '60')

            # Try to initialize with timeout
            logger.info("Calling wandb.init with 30s timeout...")
            self.run = _init_wandb_with_timeout(
                timeout=30,
                project=project,
                name=name,
                config=config,
                entity=entity,
                resume="allow" if resume else None,
                tags=tags,
                notes=notes,
                mode=mode,
                settings=wandb_settings,
            )

            if self.run is None:
                raise RuntimeError("W&B init timed out or returned None")

            logger.info(f"Successfully initialized W&B run: {self.run.name}")

        except ImportError:
            logger.warning("wandb not installed. Logging disabled.")
            self.wandb = None
            self.run = None
            self.enabled = False
        except Exception as e:
            logger.error(f"Failed to initialize W&B: {e}. Continuing without W&B logging.")
            self.wandb = None
            self.run = None
            self.enabled = False

    def log(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        commit: bool = True,
    ):
        """
        Log metrics to W&B.

        Args:
            metrics: Dictionary of metric names and values
            step: Training step (optional)
            commit: Whether to commit the log
        """
        if not self.enabled:
            return

        try:
            self.wandb.log(metrics, step=step, commit=commit)
        except Exception as e:
            logger.warning(f"Failed to log to W&B: {e}")

    def log_image(
        self,
        key: str,
        images: Union[List, Any],
        caption: Optional[str] = None,
        step: Optional[int] = None,
    ):
        """Log images to W&B."""
        if not self.enabled:
            logger.warning(f"log_image called but W&B is not enabled, skipping {key}")
            return

        try:
            if isinstance(images, list):
                wandb_images = [
                    self.wandb.Image(img, caption=caption)
                    for img in images
                ]
            else:
                wandb_images = self.wandb.Image(images, caption=caption)

            self.wandb.log({key: wandb_images}, step=step)
            logger.info(f"✓ W&B image logged: {key} at step {step}")
        except Exception as e:
            logger.error(f"Failed to log image {key}: {e}", exc_info=True)

    def log_attention_map(
        self,
        key: str,
        attention_weights: Any,
        step: Optional[int] = None,
    ):
        """
        Log attention heatmap visualization.

        Args:
            key: Metric key name
            attention_weights: Attention tensor [B, H, T, T] or [H, T, T]
            step: Training step
        """
        if not self.enabled:
            return

        try:
            import matplotlib.pyplot as plt
            import numpy as np

            # Convert to numpy
            if hasattr(attention_weights, 'cpu'):
                attn = attention_weights.cpu().detach().numpy()
            else:
                attn = np.array(attention_weights)

            # Average over batch and heads if needed
            if attn.ndim == 4:
                attn = attn[0].mean(axis=0)  # [T, T]
            elif attn.ndim == 3:
                attn = attn.mean(axis=0)  # [T, T]

            # Create heatmap
            fig, ax = plt.subplots(figsize=(8, 8))
            im = ax.imshow(attn, cmap='viridis')
            ax.set_xlabel('Key Position')
            ax.set_ylabel('Query Position')
            ax.set_title('Attention Weights')
            plt.colorbar(im, ax=ax)

            self.wandb.log({key: self.wandb.Image(fig)}, step=step)
            plt.close(fig)

        except Exception as e:
            logger.warning(f"Failed to log attention map: {e}")

    def log_gradient_histogram(
        self,
        model: Any,
        step: Optional[int] = None,
    ):
        """Log gradient histograms for model parameters."""
        if not self.enabled:
            return

        try:
            gradients = {}
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad = param.grad.cpu().detach()
                    gradients[f"gradients/{name}"] = self.wandb.Histogram(grad.numpy())

            self.wandb.log(gradients, step=step)
        except Exception as e:
            logger.warning(f"Failed to log gradients: {e}")

    def log_model(
        self,
        model_path: str,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Log model checkpoint as artifact.

        Args:
            model_path: Path to saved model
            name: Artifact name
            metadata: Additional metadata
        """
        if not self.enabled:
            return

        try:
            artifact = self.wandb.Artifact(
                name=name or "model",
                type="model",
                metadata=metadata,
            )
            artifact.add_file(model_path)
            self.run.log_artifact(artifact)
            logger.info(f"Logged model artifact: {name}")
        except Exception as e:
            logger.warning(f"Failed to log model: {e}")

    def log_table(
        self,
        key: str,
        columns: List[str],
        data: List[List[Any]],
        step: Optional[int] = None,
    ):
        """Log tabular data."""
        if not self.enabled:
            return

        try:
            table = self.wandb.Table(columns=columns, data=data)
            self.wandb.log({key: table}, step=step)
        except Exception as e:
            logger.warning(f"Failed to log table: {e}")

    def log_confusion_matrix(
        self,
        key: str,
        y_true: List[int],
        y_pred: List[int],
        class_names: List[str],
        step: Optional[int] = None,
    ):
        """Log confusion matrix."""
        if not self.enabled:
            return

        try:
            self.wandb.log({
                key: self.wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=y_true,
                    preds=y_pred,
                    class_names=class_names,
                )
            }, step=step)
        except Exception as e:
            logger.warning(f"Failed to log confusion matrix: {e}")

    def define_metric(
        self,
        name: str,
        step_metric: str = "step",
        summary: str = "best",
        goal: str = "maximize",
    ):
        """Define custom metric with summary."""
        if not self.enabled:
            return

        try:
            self.wandb.define_metric(
                name,
                step_metric=step_metric,
                summary=summary,
                goal=goal,
            )
        except Exception as e:
            logger.warning(f"Failed to define metric: {e}")

    def watch(
        self,
        model: Any,
        log: str = "gradients",
        log_freq: int = 100,
    ):
        """Watch model for automatic gradient logging."""
        if not self.enabled:
            return

        try:
            self.wandb.watch(model, log=log, log_freq=log_freq)
        except Exception as e:
            logger.warning(f"Failed to watch model: {e}")

    def alert(
        self,
        title: str,
        text: str,
        level: str = "INFO",
    ):
        """Send alert."""
        if not self.enabled:
            return

        try:
            level_map = {
                "INFO": self.wandb.AlertLevel.INFO,
                "WARN": self.wandb.AlertLevel.WARN,
                "ERROR": self.wandb.AlertLevel.ERROR,
            }
            self.wandb.alert(
                title=title,
                text=text,
                level=level_map.get(level, self.wandb.AlertLevel.INFO),
            )
        except Exception as e:
            logger.warning(f"Failed to send alert: {e}")

    def finish(self):
        """Finish the W&B run."""
        if not self.enabled:
            return

        try:
            self.run.finish()
            logger.info("W&B run finished")
        except Exception as e:
            logger.warning(f"Failed to finish W&B run: {e}")

    @property
    def url(self) -> Optional[str]:
        """Get run URL."""
        if self.run:
            return self.run.url
        return None


def create_training_dashboard(
    logger: WandbLogger,
    stages: List[str] = None,
):
    """
    Configure W&B dashboard for training.

    Args:
        logger: WandbLogger instance
        stages: Training stages to track
    """
    if not logger.enabled:
        return

    stages = stages or ['stage1', 'stage2', 'stage3', 'stage4']

    # Define metrics
    metrics = [
        ('loss', 'minimize'),
        ('accuracy', 'maximize'),
        ('robot_accuracy', 'maximize'),
        ('lr', 'last'),
    ]

    for metric, goal in metrics:
        logger.define_metric(
            metric,
            summary='best' if goal != 'last' else 'last',
            goal=goal if goal != 'last' else 'maximize',
        )

    # Validation metrics
    for stage in stages:
        logger.define_metric(
            f'{stage}/val_loss',
            summary='min',
            goal='minimize',
        )


class EnhancedWandbLogger(WandbLogger):
    """
    Enhanced W&B logger with comprehensive visualizations.

    Adds publication-quality plots, attention maps, and convergence analysis.
    Supports stage-specific visualizations for EmberVLM training.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize visualizer with error handling
        self.visualizer = None
        self.stage_visualizers = {}
        output_dir = kwargs.get('output_dir', './outputs/visualizations')
        self.current_stage = kwargs.get('stage', 1)

        try:
            from embervlm.monitoring.visualization import TrainingVisualizer
            logger.info(f"Initializing TrainingVisualizer with output_dir: {output_dir}")
            self.visualizer = TrainingVisualizer(output_dir=output_dir)
            logger.info(f"✓ TrainingVisualizer initialized successfully! Visualizations will be saved to: {output_dir}")

            # Initialize stage-specific visualizers
            try:
                from embervlm.monitoring.stage_visualizations import (
                    Stage1Visualizer, Stage2Visualizer,
                    Stage3Visualizer, Stage4Visualizer,
                    CrossStageVisualizer
                )
                self.stage_visualizers = {
                    1: Stage1Visualizer(output_dir=f"{output_dir}/stage1"),
                    2: Stage2Visualizer(output_dir=f"{output_dir}/stage2"),
                    3: Stage3Visualizer(output_dir=f"{output_dir}/stage3"),
                    4: Stage4Visualizer(output_dir=f"{output_dir}/stage4"),
                }
                self.cross_stage_viz = CrossStageVisualizer(output_dir=output_dir)
                logger.info("✓ Stage visualizers initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize stage visualizers: {e}")
                self.stage_visualizers = {}
        except Exception as e:
            logger.error(f"✗ Failed to initialize TrainingVisualizer: {e}", exc_info=True)
            logger.warning("Visualizations will be disabled. Only basic metrics will be logged.")
            self.visualizer = None

        # Track metrics history for plotting
        self.metrics_history = {}
        logger.info(f"EnhancedWandbLogger initialized. Visualizer available: {self.visualizer is not None}")

    def log_with_visualization(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        stage_name: str = "training",
        commit: bool = True,
    ):
        """
        Log metrics and generate visualizations.

        Args:
            metrics: Dictionary of metrics
            step: Training step
            stage_name: Name of training stage
            commit: Whether to commit
        """
        # Update history for all numeric metrics
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                if key not in self.metrics_history:
                    self.metrics_history[key] = []
                self.metrics_history[key].append(value)

        # Log basic metrics
        self.log(metrics, step=step, commit=False)

        # Generate visualizations periodically (only if visualizer is available)
        # Check for any loss-like metric (loss, robot_loss, sft_loss, etc.)
        loss_keys = [k for k in self.metrics_history.keys() if 'loss' in k.lower()]
        has_enough_samples = any(len(self.metrics_history.get(k, [])) >= 10 for k in loss_keys)

        if self.visualizer and step and step % 500 == 0 and has_enough_samples:
            try:
                # Find the primary loss metric
                primary_loss_key = 'loss'
                for candidate in ['loss', 'robot_loss', 'sft_loss', 'total_loss']:
                    if candidate in self.metrics_history and len(self.metrics_history[candidate]) >= 10:
                        primary_loss_key = candidate
                        break

                num_samples = len(self.metrics_history.get(primary_loss_key, []))
                logger.info(f"[{stage_name}] Generating visualizations at step {step} with {num_samples} {primary_loss_key} samples")

                # Loss decomposition plot
                try:
                    _, loss_img = self.visualizer.plot_loss_decomposition(
                        self.metrics_history,
                        stage_name,
                        save_path=str(self.visualizer.output_dir / f"{stage_name}_loss_decomp_step{step}.png")
                    )
                    self.log_image(f"{stage_name}/loss_decomposition", loss_img, step=step)
                    logger.info(f"✓ Logged loss decomposition to W&B for {stage_name}")
                except Exception as e:
                    logger.warning(f"Failed to generate loss decomposition for {stage_name}: {e}")

                # Convergence analysis
                try:
                    _, conv_img = self.visualizer.plot_convergence_analysis(
                        self.metrics_history,
                        stage_name,
                        save_path=str(self.visualizer.output_dir / f"{stage_name}_convergence_step{step}.png")
                    )
                    self.log_image(f"{stage_name}/convergence_analysis", conv_img, step=step)
                    logger.info(f"✓ Logged convergence analysis to W&B for {stage_name}")
                except Exception as e:
                    logger.warning(f"Failed to generate convergence analysis for {stage_name}: {e}")

                # Log metrics summary
                logger.info(f"[{stage_name}] Metrics tracked: {list(self.metrics_history.keys())}")

            except Exception as e:
                logger.error(f"Failed to generate visualization for {stage_name}: {e}", exc_info=True)

        if commit:
            try:
                self.wandb.log({}, commit=True)
            except Exception as e:
                logger.warning(f"Failed to commit W&B log: {e}")

    def log_attention_visualization(
        self,
        image: Any,
        attention_map: Any,
        text_tokens: List[str],
        step: int,
        stage_name: str = "training",
    ):
        """
        Log attention map overlaid on image.

        Args:
            image: Input image tensor
            attention_map: Attention weights
            text_tokens: List of text tokens
            step: Training step
            stage_name: Stage name
        """
        if not self.enabled or not self.visualizer:
            return

        try:
            _, attn_img = self.visualizer.visualize_attention_on_image(
                image, attention_map, text_tokens,
                save_path=str(self.visualizer.output_dir / f"attention_step{step}.png")
            )
            self.log_image(f"{stage_name}/attention_visualization", attn_img, step=step)
        except Exception as e:
            logger.warning(f"Failed to log attention visualization: {e}")

    def log_gradient_distribution(
        self,
        gradients: Dict[str, Any],
        step: int,
        stage_name: str = "training",
    ):
        """
        Log gradient distribution across modules.

        Args:
            gradients: Dictionary of module gradients
            step: Training step
            stage_name: Stage name
        """
        if not self.enabled or not self.visualizer:
            return

        try:
            _, grad_img = self.visualizer.plot_gradient_distribution(
                gradients, step,
                save_path=str(self.visualizer.output_dir / f"gradients_step{step}.png")
            )
            self.log_image(f"{stage_name}/gradient_distribution", grad_img, step=step)
        except Exception as e:
            logger.warning(f"Failed to log gradient distribution: {e}")

    def log_confusion_matrix(
        self,
        predictions: Any,
        labels: Any,
        class_names: List[str],
        step: int,
        stage_name: str = "training",
        confusion_matrix: Any = None,
    ):
        """
        Log confusion matrix for classification tasks.

        Args:
            predictions: Predicted class indices (or None if confusion_matrix provided)
            labels: True class indices (or None if confusion_matrix provided)
            class_names: List of class names
            step: Training step
            stage_name: Stage name
            confusion_matrix: Pre-computed confusion matrix (optional)
        """
        if not self.enabled or not self.visualizer:
            return

        try:
            import numpy as np

            if confusion_matrix is not None:
                # Use pre-computed confusion matrix
                cm = np.array(confusion_matrix)
            else:
                # Convert to numpy and compute confusion matrix
                if hasattr(predictions, 'cpu'):
                    pred_np = predictions.cpu().numpy()
                else:
                    pred_np = np.array(predictions)

                if hasattr(labels, 'cpu'):
                    label_np = labels.cpu().numpy()
                else:
                    label_np = np.array(labels)

                # Compute confusion matrix
                from sklearn.metrics import confusion_matrix as sklearn_cm
                cm = sklearn_cm(label_np, pred_np)

            _, cm_img = self.visualizer.plot_confusion_matrix(
                None, None, class_names, stage_name,
                save_path=str(self.visualizer.output_dir / f"confusion_matrix_step{step}.png"),
                confusion_matrix=cm,
            )
            self.log_image(f"{stage_name}/confusion_matrix", cm_img, step=step)
        except Exception as e:
            logger.warning(f"Failed to log confusion matrix: {e}")

    def finish(self):
        """Clean up resources and finish run."""
        if self.visualizer:
            self.visualizer.close()
        super().finish()

    # ==================== STAGE-SPECIFIC LOGGING ====================

    def set_stage(self, stage: int):
        """Set the current training stage."""
        self.current_stage = stage
        logger.info(f"W&B logger set to Stage {stage}")

    def get_stage_visualizer(self, stage: int = None):
        """Get visualizer for specific stage."""
        stage = stage or self.current_stage
        return self.stage_visualizers.get(stage)

    # Stage 1: Visual-Language Alignment
    def log_similarity_matrix(
        self,
        image_embeds: Any,
        text_embeds: Any,
        step: int,
    ):
        """Log image-text similarity matrix for Stage 1."""
        if not self.enabled:
            return

        viz = self.get_stage_visualizer(1)
        if viz is None:
            return

        try:
            _, img = viz.plot_similarity_matrix(image_embeds, text_embeds, step)
            self.log_image("stage1/similarity_matrix", img, step=step)
        except Exception as e:
            logger.warning(f"Failed to log similarity matrix: {e}")

    def log_embedding_tsne(
        self,
        image_embeds: Any,
        text_embeds: Any,
        step: int,
    ):
        """Log t-SNE visualization of embeddings."""
        if not self.enabled:
            return

        viz = self.get_stage_visualizer(1)
        if viz is None:
            return

        try:
            _, img = viz.plot_embedding_tsne(image_embeds, text_embeds, step)
            self.log_image("stage1/embedding_tsne", img, step=step)
        except Exception as e:
            logger.warning(f"Failed to log t-SNE: {e}")

    def log_retrieval_examples(
        self,
        images: List,
        captions: List[str],
        image_embeds: Any,
        text_embeds: Any,
        step: int,
    ):
        """Log retrieval examples for Stage 1."""
        if not self.enabled:
            return

        viz = self.get_stage_visualizer(1)
        if viz is None:
            return

        try:
            _, img = viz.plot_retrieval_examples(
                images, captions, image_embeds, text_embeds, step
            )
            self.log_image("stage1/retrieval_examples", img, step=step)
        except Exception as e:
            logger.warning(f"Failed to log retrieval examples: {e}")

    # Stage 2: Instruction Tuning
    def log_generation_examples(
        self,
        images: List,
        instructions: List[str],
        generated: List[str],
        ground_truth: List[str],
        step: int,
    ):
        """Log generation examples for Stage 2."""
        if not self.enabled:
            return

        viz = self.get_stage_visualizer(2)
        if viz is None:
            return

        try:
            _, img = viz.plot_generation_examples(
                images, instructions, generated, ground_truth, step
            )
            self.log_image("stage2/generation_examples", img, step=step)
        except Exception as e:
            logger.warning(f"Failed to log generation examples: {e}")

    def log_token_probabilities(
        self,
        logits: Any,
        labels: Any,
        step: int,
    ):
        """Log token probability distribution."""
        if not self.enabled:
            return

        viz = self.get_stage_visualizer(2)
        if viz is None:
            return

        try:
            _, img = viz.plot_token_probability_distribution(logits, labels, step)
            self.log_image("stage2/token_probabilities", img, step=step)
        except Exception as e:
            logger.warning(f"Failed to log token probabilities: {e}")

    # Stage 3: Robot Selection
    def log_robot_confusion_matrix(
        self,
        predictions: Any,
        labels: Any,
        step: int,
    ):
        """Log robot selection confusion matrix."""
        if not self.enabled:
            return

        viz = self.get_stage_visualizer(3)
        if viz is None:
            return

        try:
            _, img = viz.plot_confusion_matrix(predictions, labels, step)
            self.log_image("stage3/confusion_matrix", img, step=step)
        except Exception as e:
            logger.warning(f"Failed to log confusion matrix: {e}")

    def log_robot_radar_chart(
        self,
        metrics: Dict[str, Dict[str, float]],
        step: int,
    ):
        """Log per-robot performance radar chart."""
        if not self.enabled:
            return

        viz = self.get_stage_visualizer(3)
        if viz is None:
            return

        try:
            _, img = viz.plot_per_robot_radar(metrics, step)
            self.log_image("stage3/robot_radar", img, step=step)
        except Exception as e:
            logger.warning(f"Failed to log radar chart: {e}")

    def log_calibration_plot(
        self,
        confidences: Any,
        correct: Any,
        step: int,
    ):
        """Log confidence calibration plot."""
        if not self.enabled:
            return

        viz = self.get_stage_visualizer(3)
        if viz is None:
            return

        try:
            _, img = viz.plot_confidence_calibration(confidences, correct, step)
            self.log_image("stage3/calibration", img, step=step)
        except Exception as e:
            logger.warning(f"Failed to log calibration: {e}")

    def log_reasoning_examples(
        self,
        tasks: List[str],
        reasoning_chains: List[str],
        predictions: List[str],
        ground_truth: List[str],
        correct: List[bool],
        step: int,
    ):
        """Log reasoning chain examples."""
        if not self.enabled:
            return

        viz = self.get_stage_visualizer(3)
        if viz is None:
            return

        try:
            _, img = viz.plot_reasoning_examples(
                tasks, reasoning_chains, predictions, ground_truth, correct, step
            )
            self.log_image("stage3/reasoning_examples", img, step=step)
        except Exception as e:
            logger.warning(f"Failed to log reasoning examples: {e}")

    # Stage 4: Chain-of-Thought Reasoning
    def log_reasoning_quality(
        self,
        coherence_scores: List[float],
        consistency_scores: List[float],
        step_counts: List[int],
        step: int,
    ):
        """Log reasoning quality metrics."""
        if not self.enabled:
            return

        viz = self.get_stage_visualizer(4)
        if viz is None:
            return

        try:
            _, img = viz.plot_reasoning_quality_metrics(
                coherence_scores, consistency_scores, step_counts, step
            )
            self.log_image("stage4/reasoning_quality", img, step=step)
        except Exception as e:
            logger.warning(f"Failed to log reasoning quality: {e}")

    def log_phase_comparison(
        self,
        phase1_history: Dict[str, List[float]],
        phase2_history: Dict[str, List[float]],
        step: int,
    ):
        """Log Phase 1 vs Phase 2 comparison."""
        if not self.enabled:
            return

        viz = self.get_stage_visualizer(4)
        if viz is None:
            return

        try:
            _, img = viz.plot_phase_comparison(phase1_history, phase2_history, step)
            self.log_image("stage4/phase_comparison", img, step=step)
        except Exception as e:
            logger.warning(f"Failed to log phase comparison: {e}")

    def log_cot_comparison(
        self,
        tasks: List[str],
        without_cot: List,
        with_cot: List,
        step: int,
    ):
        """Log CoT comparison examples."""
        if not self.enabled:
            return

        viz = self.get_stage_visualizer(4)
        if viz is None:
            return

        try:
            _, img = viz.plot_cot_examples(tasks, without_cot, with_cot, step)
            self.log_image("stage4/cot_comparison", img, step=step)
        except Exception as e:
            logger.warning(f"Failed to log CoT comparison: {e}")

    # Cross-Stage Visualizations
    def log_stage_progression(
        self,
        eval_results: Dict[int, Dict[str, float]],
    ):
        """Log benchmark progression across stages."""
        if not self.enabled or not hasattr(self, 'cross_stage_viz'):
            return

        try:
            _, img = self.cross_stage_viz.plot_stage_progression(eval_results)
            self.log_image("cross_stage/progression", img)
        except Exception as e:
            logger.warning(f"Failed to log stage progression: {e}")

    def log_training_summary(
        self,
        all_metrics: Dict[int, Dict[str, List[float]]],
    ):
        """Log comprehensive training summary."""
        if not self.enabled or not hasattr(self, 'cross_stage_viz'):
            return

        try:
            _, img = self.cross_stage_viz.plot_training_summary(all_metrics)
            self.log_image("cross_stage/training_summary", img)
        except Exception as e:
            logger.warning(f"Failed to log training summary: {e}")

    def log_carbon_footprint(
        self,
        emissions_per_stage: Dict[int, float],
    ):
        """Log carbon footprint analysis."""
        if not self.enabled or not hasattr(self, 'cross_stage_viz'):
            return

        try:
            _, img = self.cross_stage_viz.plot_carbon_footprint(emissions_per_stage)
            self.log_image("cross_stage/carbon_footprint", img)
        except Exception as e:
            logger.warning(f"Failed to log carbon footprint: {e}")

    def log_evaluation_results(
        self,
        results: Dict[str, Any],
        stage: int,
    ):
        """Log VLMEvalKit evaluation results."""
        if not self.enabled:
            return

        try:
            # Log individual benchmark scores
            if 'results' in results:
                for benchmark, score in results['results'].items():
                    if isinstance(score, dict):
                        for metric, value in score.items():
                            if isinstance(value, (int, float)):
                                self.log({f"eval/{benchmark}/{metric}": value})
                    elif isinstance(score, (int, float)):
                        self.log({f"eval/{benchmark}": score})

            # Log robot selection metrics
            if 'robot_selection' in results:
                for metric, value in results['robot_selection'].items():
                    if isinstance(value, (int, float)):
                        self.log({f"eval/robot_selection/{metric}": value})

            # Log calibration
            if 'calibration' in results:
                for metric, value in results['calibration'].items():
                    if isinstance(value, (int, float)):
                        self.log({f"eval/calibration/{metric}": value})

            logger.info(f"Logged evaluation results for Stage {stage}")

        except Exception as e:
            logger.warning(f"Failed to log evaluation results: {e}")

