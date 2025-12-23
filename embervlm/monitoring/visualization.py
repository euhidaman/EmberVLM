"""
Publication-Quality Visualization Module for EmberVLM

Generates conference-ready plots and attention visualizations logged to W&B.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, List, Tuple
import io
from PIL import Image
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")


class TrainingVisualizer:
    """Generates publication-quality visualizations for W&B logging."""

    def __init__(self, output_dir: str = "./outputs/visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Configure matplotlib for high-quality output
        matplotlib.rcParams['figure.dpi'] = 300
        matplotlib.rcParams['savefig.dpi'] = 300
        matplotlib.rcParams['font.size'] = 10
        matplotlib.rcParams['axes.labelsize'] = 11
        matplotlib.rcParams['axes.titlesize'] = 12
        matplotlib.rcParams['xtick.labelsize'] = 9
        matplotlib.rcParams['ytick.labelsize'] = 9
        matplotlib.rcParams['legend.fontsize'] = 9

    def visualize_attention_on_image(
        self,
        image: torch.Tensor,
        attention_map: torch.Tensor,
        text_tokens: List[str],
        save_path: Optional[str] = None,
    ) -> Tuple[plt.Figure, Image.Image]:
        """
        Overlay attention map on image for visualization.

        Args:
            image: Image tensor [3, H, W]
            attention_map: Attention weights [num_heads, num_text_tokens, num_visual_tokens]
            text_tokens: List of text tokens
            save_path: Optional path to save figure

        Returns:
            Tuple of (matplotlib figure, PIL Image)
        """
        # Average across heads and text tokens to get spatial attention
        # attention_map: [num_heads, num_text_tokens, num_visual_tokens]
        spatial_attention = attention_map.mean(dim=(0, 1))  # [num_visual_tokens]

        # Reshape to 2D (assuming square grid of visual tokens)
        grid_size = int(np.sqrt(spatial_attention.shape[0]))
        attention_2d = spatial_attention.reshape(grid_size, grid_size).cpu().numpy()

        # Prepare image
        img_np = image.permute(1, 2, 0).cpu().numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)

        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original image
        axes[0].imshow(img_np)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # Attention heatmap
        im = axes[1].imshow(attention_2d, cmap='hot', interpolation='bilinear')
        axes[1].set_title('Attention Map')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

        # Overlay
        axes[2].imshow(img_np)
        attention_overlay = F.interpolate(
            torch.from_numpy(attention_2d).unsqueeze(0).unsqueeze(0),
            size=(img_np.shape[0], img_np.shape[1]),
            mode='bilinear',
            align_corners=False
        ).squeeze().numpy()
        axes[2].imshow(attention_overlay, cmap='hot', alpha=0.5, interpolation='bilinear')
        axes[2].set_title('Attention Overlay')
        axes[2].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)

        # Convert to PIL Image for W&B
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        buf.seek(0)
        pil_image = Image.open(buf).copy()
        buf.close()

        return fig, pil_image

    def plot_loss_decomposition(
        self,
        metrics_history: Dict[str, List[float]],
        stage_name: str,
        save_path: Optional[str] = None,
    ) -> Tuple[plt.Figure, Image.Image]:
        """
        Plot stacked loss components over time.

        Args:
            metrics_history: Dictionary of metric names to values
            stage_name: Name of training stage
            save_path: Optional path to save figure

        Returns:
            Tuple of (matplotlib figure, PIL Image)
        """
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # Filter loss components
        loss_keys = [k for k in metrics_history.keys() if 'loss' in k.lower() and k != 'loss']

        # Plot 1: Stacked area chart of loss components
        if loss_keys:
            steps = np.arange(len(metrics_history[loss_keys[0]]))
            loss_data = np.array([metrics_history[k] for k in loss_keys])

            axes[0].stackplot(steps, loss_data, labels=loss_keys, alpha=0.7)
            axes[0].set_ylabel('Loss Components')
            axes[0].set_title(f'{stage_name}: Loss Decomposition')
            axes[0].legend(loc='upper right', ncol=2)
            axes[0].grid(True, alpha=0.3)

        # Plot 2: Total loss with trend
        if 'loss' in metrics_history:
            total_loss = metrics_history['loss']
            steps = np.arange(len(total_loss))
            axes[1].plot(steps, total_loss, linewidth=2, label='Total Loss', color='#2E86AB')

            # Add moving average trend
            window = min(50, len(total_loss) // 10)
            if window > 1:
                moving_avg = np.convolve(total_loss, np.ones(window)/window, mode='valid')
                axes[1].plot(steps[window-1:], moving_avg, '--', linewidth=2,
                           label=f'Trend (MA-{window})', color='#A23B72', alpha=0.8)

            axes[1].set_xlabel('Training Step')
            axes[1].set_ylabel('Total Loss')
            axes[1].set_title('Training Loss Progression')
            axes[1].legend(loc='upper right')
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)

        # Convert to PIL Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        buf.seek(0)
        pil_image = Image.open(buf).copy()
        buf.close()

        return fig, pil_image

    def plot_gradient_distribution(
        self,
        gradients: Dict[str, torch.Tensor],
        step: int,
        save_path: Optional[str] = None,
    ) -> Tuple[plt.Figure, Image.Image]:
        """
        Plot distribution of gradients per module.

        Args:
            gradients: Dictionary of module names to gradient tensors
            step: Current training step
            save_path: Optional path to save figure

        Returns:
            Tuple of (matplotlib figure, PIL Image)
        """
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))

        # Collect statistics
        modules = []
        means = []
        stds = []
        norms = []

        for name, grad in gradients.items():
            if grad is not None:
                grad_flat = grad.flatten()
                modules.append(name)
                means.append(grad_flat.mean().item())
                stds.append(grad_flat.std().item())
                norms.append(grad.norm().item())

        x = np.arange(len(modules))

        # Plot 1: Gradient statistics
        ax1 = axes[0]
        width = 0.35
        ax1.bar(x - width/2, means, width, label='Mean', alpha=0.8)
        ax1.bar(x + width/2, stds, width, label='Std Dev', alpha=0.8)
        ax1.set_ylabel('Gradient Value')
        ax1.set_title(f'Gradient Statistics (Step {step})')
        ax1.set_xticks(x)
        ax1.set_xticklabels(modules, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')

        # Plot 2: Gradient norms (log scale)
        ax2 = axes[1]
        ax2.bar(x, norms, alpha=0.8, color='#F18F01')
        ax2.set_ylabel('Gradient Norm (log scale)')
        ax2.set_title('Gradient Norms by Module')
        ax2.set_xticks(x)
        ax2.set_xticklabels(modules, rotation=45, ha='right')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)

        # Convert to PIL Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        buf.seek(0)
        pil_image = Image.open(buf).copy()
        buf.close()

        return fig, pil_image

    def plot_confusion_matrix(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        class_names: List[str],
        stage_name: str,
        save_path: Optional[str] = None,
    ) -> Tuple[plt.Figure, Image.Image]:
        """
        Plot confusion matrix for classification tasks.

        Args:
            predictions: Predicted class indices
            labels: True class indices
            class_names: List of class names
            stage_name: Name of training stage
            save_path: Optional path to save figure

        Returns:
            Tuple of (matplotlib figure, PIL Image)
        """
        from sklearn.metrics import confusion_matrix

        # Compute confusion matrix
        cm = confusion_matrix(labels, predictions)
        cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # Plot 1: Raw counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names,
                   ax=axes[0], cbar_kws={'label': 'Count'})
        axes[0].set_title(f'{stage_name}: Confusion Matrix (Counts)')
        axes[0].set_ylabel('True Label')
        axes[0].set_xlabel('Predicted Label')

        # Plot 2: Normalized
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='RdYlGn',
                   xticklabels=class_names, yticklabels=class_names,
                   ax=axes[1], cbar_kws={'label': 'Proportion'}, vmin=0, vmax=1)
        axes[1].set_title(f'{stage_name}: Confusion Matrix (Normalized)')
        axes[1].set_ylabel('True Label')
        axes[1].set_xlabel('Predicted Label')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)

        # Convert to PIL Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        buf.seek(0)
        pil_image = Image.open(buf).copy()
        buf.close()

        return fig, pil_image

    def plot_convergence_analysis(
        self,
        metrics_history: Dict[str, List[float]],
        stage_name: str,
        save_path: Optional[str] = None,
    ) -> Tuple[plt.Figure, Image.Image]:
        """
        Plot convergence indicators and stability metrics.

        Args:
            metrics_history: Dictionary of metric names to values
            stage_name: Name of training stage
            save_path: Optional path to save figure

        Returns:
            Tuple of (matplotlib figure, PIL Image)
        """
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        # Plot 1: Loss and validation loss
        ax1 = fig.add_subplot(gs[0, :])
        if 'loss' in metrics_history:
            steps = np.arange(len(metrics_history['loss']))
            ax1.plot(steps, metrics_history['loss'], label='Training Loss', linewidth=2)
        if 'val_loss' in metrics_history:
            val_steps = np.linspace(0, len(steps)-1, len(metrics_history['val_loss']))
            ax1.plot(val_steps, metrics_history['val_loss'],
                    label='Validation Loss', linewidth=2, marker='o', markersize=4)
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Loss')
        ax1.set_title(f'{stage_name}: Loss Convergence')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Learning rate schedule
        ax2 = fig.add_subplot(gs[1, 0])
        if 'lr' in metrics_history:
            ax2.plot(metrics_history['lr'], linewidth=2, color='#E63946')
            ax2.set_xlabel('Step')
            ax2.set_ylabel('Learning Rate')
            ax2.set_title('Learning Rate Schedule')
            ax2.grid(True, alpha=0.3)

        # Plot 3: Gradient norm
        ax3 = fig.add_subplot(gs[1, 1])
        if 'grad_norm' in metrics_history:
            ax3.plot(metrics_history['grad_norm'], linewidth=2, color='#2A9D8F')
            ax3.set_xlabel('Step')
            ax3.set_ylabel('Gradient Norm')
            ax3.set_title('Gradient Norm Evolution')
            ax3.grid(True, alpha=0.3)

        # Plot 4: Accuracy metrics
        ax4 = fig.add_subplot(gs[2, 0])
        acc_keys = [k for k in metrics_history.keys() if 'acc' in k.lower()]
        for key in acc_keys[:3]:  # Limit to 3 for clarity
            ax4.plot(metrics_history[key], label=key, linewidth=2, alpha=0.8)
        if acc_keys:
            ax4.set_xlabel('Step')
            ax4.set_ylabel('Accuracy')
            ax4.set_title('Accuracy Metrics')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

        # Plot 5: Loss variance (stability indicator)
        ax5 = fig.add_subplot(gs[2, 1])
        if 'loss' in metrics_history and len(metrics_history['loss']) > 50:
            window = 50
            loss = np.array(metrics_history['loss'])
            rolling_var = np.array([
                np.var(loss[max(0, i-window):i+1])
                for i in range(len(loss))
            ])
            ax5.plot(rolling_var, linewidth=2, color='#F4A261')
            ax5.set_xlabel('Step')
            ax5.set_ylabel(f'Loss Variance (window={window})')
            ax5.set_title('Training Stability')
            ax5.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)

        # Convert to PIL Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        buf.seek(0)
        pil_image = Image.open(buf).copy()
        buf.close()
        plt.close(fig)

        return fig, pil_image

    def close(self):
        """Clean up matplotlib resources."""
        plt.close('all')

