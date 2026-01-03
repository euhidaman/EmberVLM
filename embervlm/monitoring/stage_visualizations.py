"""
Stage-Specific Visualization Module for EmberVLM

Provides comprehensive visualizations for each training stage:
- Stage 1: Visual-Language Alignment
- Stage 2: Instruction Tuning
- Stage 3: Robot Selection
- Stage 4: Chain-of-Thought Reasoning

All visualizations are designed for W&B logging and publication quality.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
import io
import logging

# Configure matplotlib for non-interactive use
import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# Import seaborn if available
try:
    import seaborn as sns
    sns.set_palette("husl")
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    logger.warning("Seaborn not available, using matplotlib only")

# Import sklearn if available
try:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    from sklearn.metrics import confusion_matrix as sklearn_cm
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logger.warning("sklearn not available, some visualizations disabled")

from PIL import Image


def _fig_to_pil(fig: plt.Figure, dpi: int = 150) -> Image.Image:
    """Convert matplotlib figure to PIL Image."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=dpi)
    buf.seek(0)
    pil_image = Image.open(buf).copy()
    buf.close()
    plt.close(fig)
    return pil_image


class Stage1Visualizer:
    """Visualizations for Stage 1: Visual-Language Alignment."""

    def __init__(self, output_dir: str = "./outputs/stage1/visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_similarity_matrix(
        self,
        image_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
        step: int,
        save: bool = True,
    ) -> Tuple[plt.Figure, Image.Image]:
        """
        Plot image-text similarity matrix (batch-level CLIP-style).

        Args:
            image_embeds: Image embeddings [B, D]
            text_embeds: Text embeddings [B, D]
            step: Training step
            save: Whether to save to disk

        Returns:
            Tuple of (figure, PIL image)
        """
        # Normalize embeddings
        image_embeds = F.normalize(image_embeds, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)

        # Compute similarity matrix
        similarity = torch.matmul(image_embeds, text_embeds.T).cpu().numpy()

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Raw similarity
        im1 = axes[0].imshow(similarity, cmap='RdYlGn', vmin=-1, vmax=1)
        axes[0].set_title('Image-Text Similarity Matrix')
        axes[0].set_xlabel('Text Index')
        axes[0].set_ylabel('Image Index')
        plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

        # Highlight diagonal (correct matches)
        batch_size = similarity.shape[0]
        for i in range(batch_size):
            axes[0].add_patch(plt.Rectangle((i-0.5, i-0.5), 1, 1,
                                            fill=False, edgecolor='blue', linewidth=2))

        # Softmax scores (retrieval probabilities)
        i2t_probs = F.softmax(torch.from_numpy(similarity), dim=1).numpy()
        im2 = axes[1].imshow(i2t_probs, cmap='Blues', vmin=0, vmax=1)
        axes[1].set_title('Image→Text Retrieval Probabilities')
        axes[1].set_xlabel('Text Index')
        axes[1].set_ylabel('Image Index')
        plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

        # Add diagonal accuracy annotation
        diagonal_acc = np.mean(np.argmax(similarity, axis=1) == np.arange(batch_size))
        fig.suptitle(f'Step {step} | Batch Retrieval Accuracy: {diagonal_acc:.1%}', fontsize=12)

        plt.tight_layout()

        if save:
            save_path = self.output_dir / f"similarity_matrix_step{step}.png"
            fig.savefig(save_path, bbox_inches='tight', dpi=300)

        return fig, _fig_to_pil(fig)

    def plot_embedding_tsne(
        self,
        image_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
        step: int,
        n_samples: int = 100,
        save: bool = True,
    ) -> Tuple[plt.Figure, Image.Image]:
        """
        Plot t-SNE visualization of image and text embeddings.

        Args:
            image_embeds: Image embeddings [N, D]
            text_embeds: Text embeddings [N, D]
            step: Training step
            n_samples: Max samples to visualize
            save: Whether to save

        Returns:
            Tuple of (figure, PIL image)
        """
        if not HAS_SKLEARN:
            logger.warning("sklearn not available for t-SNE")
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.text(0.5, 0.5, 't-SNE requires sklearn', ha='center', va='center')
            return fig, _fig_to_pil(fig)

        # Subsample if needed
        n = min(n_samples, image_embeds.size(0))
        img_emb = image_embeds[:n].cpu().numpy()
        txt_emb = text_embeds[:n].cpu().numpy()

        # Combine embeddings
        combined = np.vstack([img_emb, txt_emb])
        labels = ['Image'] * n + ['Text'] * n

        # Run t-SNE
        tsne = TSNE(n_components=2, perplexity=min(30, n-1), random_state=42)
        coords = tsne.fit_transform(combined)

        fig, ax = plt.subplots(figsize=(10, 10))

        # Plot with different colors
        img_coords = coords[:n]
        txt_coords = coords[n:]

        ax.scatter(img_coords[:, 0], img_coords[:, 1], c='#2E86AB',
                   label='Image Embeddings', alpha=0.7, s=50)
        ax.scatter(txt_coords[:, 0], txt_coords[:, 1], c='#A23B72',
                   label='Text Embeddings', alpha=0.7, s=50, marker='^')

        # Draw lines between matching pairs
        for i in range(n):
            ax.plot([img_coords[i, 0], txt_coords[i, 0]],
                   [img_coords[i, 1], txt_coords[i, 1]],
                   'gray', alpha=0.2, linewidth=0.5)

        ax.set_title(f't-SNE Visualization of Embeddings (Step {step})')
        ax.legend()
        ax.set_xlabel('t-SNE Dimension 1')
        ax.set_ylabel('t-SNE Dimension 2')

        plt.tight_layout()

        if save:
            save_path = self.output_dir / f"tsne_step{step}.png"
            fig.savefig(save_path, bbox_inches='tight', dpi=300)

        return fig, _fig_to_pil(fig)

    def plot_retrieval_examples(
        self,
        images: List[Image.Image],
        captions: List[str],
        image_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
        step: int,
        top_k: int = 3,
        n_queries: int = 4,
        save: bool = True,
    ) -> Tuple[plt.Figure, Image.Image]:
        """
        Show top-K retrieval examples (image→text and text→image).

        Args:
            images: List of PIL images
            captions: List of caption strings
            image_embeds: Image embeddings [N, D]
            text_embeds: Text embeddings [N, D]
            step: Training step
            top_k: Number of top matches to show
            n_queries: Number of query examples
            save: Whether to save

        Returns:
            Tuple of (figure, PIL image)
        """
        # Normalize and compute similarity
        image_embeds = F.normalize(image_embeds, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)
        similarity = torch.matmul(image_embeds, text_embeds.T).cpu()

        n_queries = min(n_queries, len(images))

        fig, axes = plt.subplots(n_queries, top_k + 1, figsize=(4*(top_k+1), 4*n_queries))

        for i in range(n_queries):
            # Show query image
            axes[i, 0].imshow(images[i])
            axes[i, 0].set_title(f'Query Image {i}', fontsize=10)
            axes[i, 0].axis('off')

            # Get top-K text matches
            scores, indices = similarity[i].topk(top_k)

            for j, (score, idx) in enumerate(zip(scores, indices)):
                is_correct = (idx == i)
                color = 'green' if is_correct else 'red'

                axes[i, j+1].text(0.5, 0.5,
                                  f"Rank {j+1}\nScore: {score:.3f}\n\n{captions[idx][:100]}...",
                                  ha='center', va='center', wrap=True, fontsize=8,
                                  color=color)
                axes[i, j+1].set_xlim(0, 1)
                axes[i, j+1].set_ylim(0, 1)
                axes[i, j+1].axis('off')
                axes[i, j+1].set_title('✓ Correct' if is_correct else '✗ Wrong',
                                       color=color, fontsize=9)

        fig.suptitle(f'Image→Text Retrieval Examples (Step {step})', fontsize=14)
        plt.tight_layout()

        if save:
            save_path = self.output_dir / f"retrieval_examples_step{step}.png"
            fig.savefig(save_path, bbox_inches='tight', dpi=200)

        return fig, _fig_to_pil(fig)

    def plot_cross_attention(
        self,
        attention_weights: torch.Tensor,
        image: Image.Image,
        text_tokens: List[str],
        step: int,
        save: bool = True,
    ) -> Tuple[plt.Figure, Image.Image]:
        """
        Visualize cross-attention between image regions and text tokens.

        Args:
            attention_weights: Attention [num_heads, text_len, visual_tokens]
            image: Original PIL image
            text_tokens: List of text tokens
            step: Training step
            save: Whether to save

        Returns:
            Tuple of (figure, PIL image)
        """
        # Average across heads
        attn = attention_weights.mean(dim=0).cpu().numpy()  # [text_len, visual_tokens]

        # Create figure
        n_tokens = min(10, len(text_tokens))  # Limit tokens shown
        fig, axes = plt.subplots(2, n_tokens // 2 + 1, figsize=(20, 8))
        axes = axes.flatten()

        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # Get grid size for visual tokens
        n_visual = attn.shape[1]
        grid_size = int(np.sqrt(n_visual))

        # Show attention for each token
        for i, token in enumerate(text_tokens[:n_tokens]):
            ax = axes[i + 1]

            token_attn = attn[i].reshape(grid_size, grid_size)
            token_attn_resized = np.array(
                Image.fromarray(token_attn).resize(image.size, Image.BILINEAR)
            )

            ax.imshow(image)
            ax.imshow(token_attn_resized, cmap='hot', alpha=0.5)
            ax.set_title(f'"{token}"', fontsize=9)
            ax.axis('off')

        # Hide unused axes
        for i in range(n_tokens + 1, len(axes)):
            axes[i].axis('off')

        fig.suptitle(f'Cross-Attention Visualization (Step {step})', fontsize=14)
        plt.tight_layout()

        if save:
            save_path = self.output_dir / f"cross_attention_step{step}.png"
            fig.savefig(save_path, bbox_inches='tight', dpi=200)

        return fig, _fig_to_pil(fig)


class Stage2Visualizer:
    """Visualizations for Stage 2: Instruction Tuning."""

    def __init__(self, output_dir: str = "./outputs/stage2/visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_generation_examples(
        self,
        images: List[Image.Image],
        instructions: List[str],
        generated: List[str],
        ground_truth: List[str],
        step: int,
        n_examples: int = 4,
        save: bool = True,
    ) -> Tuple[plt.Figure, Image.Image]:
        """
        Display generation examples with images.

        Args:
            images: Input images
            instructions: Instruction prompts
            generated: Generated responses
            ground_truth: Ground truth responses
            step: Training step
            n_examples: Number of examples to show
            save: Whether to save

        Returns:
            Tuple of (figure, PIL image)
        """
        n_examples = min(n_examples, len(images))

        fig, axes = plt.subplots(n_examples, 3, figsize=(18, 5*n_examples))
        if n_examples == 1:
            axes = axes.reshape(1, -1)

        for i in range(n_examples):
            # Image
            axes[i, 0].imshow(images[i])
            axes[i, 0].set_title(f'Input Image {i+1}')
            axes[i, 0].axis('off')

            # Instruction + Generated
            text = f"Instruction:\n{instructions[i][:200]}\n\n---\nGenerated:\n{generated[i][:300]}"
            axes[i, 1].text(0.05, 0.95, text, ha='left', va='top', wrap=True,
                           fontsize=9, transform=axes[i, 1].transAxes)
            axes[i, 1].set_xlim(0, 1)
            axes[i, 1].set_ylim(0, 1)
            axes[i, 1].set_title('Instruction & Generation', fontsize=10)
            axes[i, 1].axis('off')

            # Ground truth
            axes[i, 2].text(0.05, 0.95, f"Ground Truth:\n{ground_truth[i][:400]}",
                           ha='left', va='top', wrap=True, fontsize=9,
                           transform=axes[i, 2].transAxes, color='green')
            axes[i, 2].set_xlim(0, 1)
            axes[i, 2].set_ylim(0, 1)
            axes[i, 2].set_title('Ground Truth', fontsize=10, color='green')
            axes[i, 2].axis('off')

        fig.suptitle(f'Generation Examples (Step {step})', fontsize=14)
        plt.tight_layout()

        if save:
            save_path = self.output_dir / f"generation_examples_step{step}.png"
            fig.savefig(save_path, bbox_inches='tight', dpi=200)

        return fig, _fig_to_pil(fig)

    def plot_token_probability_distribution(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        step: int,
        save: bool = True,
    ) -> Tuple[plt.Figure, Image.Image]:
        """
        Plot distribution of predicted token probabilities.

        Args:
            logits: Model logits [B, T, V]
            labels: Target labels [B, T]
            step: Training step
            save: Whether to save

        Returns:
            Tuple of (figure, PIL image)
        """
        # Get probabilities for correct tokens
        probs = F.softmax(logits, dim=-1)  # [B, T, V]

        # Gather probabilities of correct tokens
        batch_size, seq_len, vocab_size = probs.shape

        # Mask out padding (-100)
        valid_mask = labels != -100

        # Flatten and gather
        flat_probs = probs.view(-1, vocab_size)
        flat_labels = labels.view(-1).clamp(min=0)

        correct_probs = flat_probs.gather(1, flat_labels.unsqueeze(1)).squeeze()
        correct_probs = correct_probs[valid_mask.view(-1)].cpu().numpy()

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram of correct token probabilities
        axes[0].hist(correct_probs, bins=50, alpha=0.7, color='#2E86AB', edgecolor='white')
        axes[0].axvline(np.mean(correct_probs), color='red', linestyle='--',
                       label=f'Mean: {np.mean(correct_probs):.3f}')
        axes[0].set_xlabel('Probability of Correct Token')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Distribution of Correct Token Probabilities')
        axes[0].legend()
        axes[0].set_xlim(0, 1)

        # Confidence categories
        bins = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        labels_cat = ['Very Low\n(0-0.1)', 'Low\n(0.1-0.3)', 'Medium\n(0.3-0.5)',
                      'High\n(0.5-0.7)', 'Very High\n(0.7-0.9)', 'Confident\n(0.9-1.0)']
        counts, _ = np.histogram(correct_probs, bins=bins)

        colors = ['#FF6B6B', '#FFA06B', '#FFD93D', '#6BCB77', '#4D96FF', '#6A5ACD']
        axes[1].bar(labels_cat, counts, color=colors, edgecolor='white')
        axes[1].set_xlabel('Confidence Category')
        axes[1].set_ylabel('Token Count')
        axes[1].set_title('Token Confidence Distribution')

        # Add percentages
        total = sum(counts)
        for i, (count, label) in enumerate(zip(counts, labels_cat)):
            axes[1].text(i, count + total*0.01, f'{count/total*100:.1f}%',
                        ha='center', fontsize=9)

        fig.suptitle(f'Token Probability Analysis (Step {step})', fontsize=14)
        plt.tight_layout()

        if save:
            save_path = self.output_dir / f"token_probs_step{step}.png"
            fig.savefig(save_path, bbox_inches='tight', dpi=200)

        return fig, _fig_to_pil(fig)

    def plot_response_length_distribution(
        self,
        generated_lengths: List[int],
        target_lengths: List[int],
        step: int,
        save: bool = True,
    ) -> Tuple[plt.Figure, Image.Image]:
        """
        Compare generated vs target response lengths.

        Args:
            generated_lengths: List of generated response lengths
            target_lengths: List of target response lengths
            step: Training step
            save: Whether to save

        Returns:
            Tuple of (figure, PIL image)
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Overlapping histograms
        bins = np.linspace(0, max(max(generated_lengths), max(target_lengths)), 30)
        axes[0].hist(target_lengths, bins=bins, alpha=0.6, label='Target', color='#2E86AB')
        axes[0].hist(generated_lengths, bins=bins, alpha=0.6, label='Generated', color='#A23B72')
        axes[0].axvline(np.mean(target_lengths), color='#2E86AB', linestyle='--',
                       label=f'Target Mean: {np.mean(target_lengths):.1f}')
        axes[0].axvline(np.mean(generated_lengths), color='#A23B72', linestyle='--',
                       label=f'Gen Mean: {np.mean(generated_lengths):.1f}')
        axes[0].set_xlabel('Response Length (tokens)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Response Length Distribution')
        axes[0].legend()

        # Scatter plot
        axes[1].scatter(target_lengths, generated_lengths, alpha=0.5, s=20)
        max_len = max(max(target_lengths), max(generated_lengths))
        axes[1].plot([0, max_len], [0, max_len], 'r--', label='Perfect Match')
        axes[1].set_xlabel('Target Length')
        axes[1].set_ylabel('Generated Length')
        axes[1].set_title('Target vs Generated Length')
        axes[1].legend()

        fig.suptitle(f'Response Length Analysis (Step {step})', fontsize=14)
        plt.tight_layout()

        if save:
            save_path = self.output_dir / f"response_lengths_step{step}.png"
            fig.savefig(save_path, bbox_inches='tight', dpi=200)

        return fig, _fig_to_pil(fig)


class Stage3Visualizer:
    """Visualizations for Stage 3: Robot Selection."""

    ROBOT_NAMES = ["Drone", "Underwater", "Humanoid", "Wheeled", "Legged"]
    ROBOT_COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

    def __init__(self, output_dir: str = "./outputs/stage3/visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.confusion_history = []

    def plot_confusion_matrix(
        self,
        predictions: Union[torch.Tensor, np.ndarray],
        labels: Union[torch.Tensor, np.ndarray],
        step: int,
        save: bool = True,
    ) -> Tuple[plt.Figure, Image.Image]:
        """
        Plot confusion matrix for robot selection.

        Args:
            predictions: Predicted robot indices
            labels: True robot indices
            step: Training step
            save: Whether to save

        Returns:
            Tuple of (figure, PIL image)
        """
        if not HAS_SKLEARN:
            logger.warning("sklearn not available for confusion matrix")
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.text(0.5, 0.5, 'Confusion matrix requires sklearn', ha='center', va='center')
            return fig, _fig_to_pil(fig)

        if torch.is_tensor(predictions):
            predictions = predictions.cpu().numpy()
        if torch.is_tensor(labels):
            labels = labels.cpu().numpy()

        cm = sklearn_cm(labels, predictions, labels=list(range(5)))
        cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)

        # Store for history tracking
        self.confusion_history.append(cm)

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # Raw counts
        if HAS_SEABORN:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=self.ROBOT_NAMES, yticklabels=self.ROBOT_NAMES,
                       ax=axes[0], cbar_kws={'label': 'Count'})
        else:
            im = axes[0].imshow(cm, cmap='Blues')
            axes[0].set_xticks(range(5))
            axes[0].set_yticks(range(5))
            axes[0].set_xticklabels(self.ROBOT_NAMES, rotation=45, ha='right')
            axes[0].set_yticklabels(self.ROBOT_NAMES)
            for i in range(5):
                for j in range(5):
                    axes[0].text(j, i, str(cm[i, j]), ha='center', va='center')
            plt.colorbar(im, ax=axes[0])

        axes[0].set_title(f'Robot Selection Confusion Matrix (Counts)')
        axes[0].set_ylabel('True Robot')
        axes[0].set_xlabel('Predicted Robot')

        # Normalized
        if HAS_SEABORN:
            sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='RdYlGn',
                       xticklabels=self.ROBOT_NAMES, yticklabels=self.ROBOT_NAMES,
                       ax=axes[1], vmin=0, vmax=1, cbar_kws={'label': 'Proportion'})
        else:
            im = axes[1].imshow(cm_norm, cmap='RdYlGn', vmin=0, vmax=1)
            axes[1].set_xticks(range(5))
            axes[1].set_yticks(range(5))
            axes[1].set_xticklabels(self.ROBOT_NAMES, rotation=45, ha='right')
            axes[1].set_yticklabels(self.ROBOT_NAMES)
            for i in range(5):
                for j in range(5):
                    axes[1].text(j, i, f'{cm_norm[i, j]:.2f}', ha='center', va='center')
            plt.colorbar(im, ax=axes[1])

        axes[1].set_title(f'Robot Selection Confusion Matrix (Normalized)')
        axes[1].set_ylabel('True Robot')
        axes[1].set_xlabel('Predicted Robot')

        # Add overall accuracy
        acc = np.trace(cm) / (np.sum(cm) + 1e-8)
        fig.suptitle(f'Step {step} | Overall Accuracy: {acc:.1%}', fontsize=14)

        plt.tight_layout()

        if save:
            save_path = self.output_dir / f"confusion_matrix_step{step}.png"
            fig.savefig(save_path, bbox_inches='tight', dpi=200)

        return fig, _fig_to_pil(fig)

    def plot_per_robot_radar(
        self,
        metrics: Dict[str, Dict[str, float]],
        step: int,
        save: bool = True,
    ) -> Tuple[plt.Figure, Image.Image]:
        """
        Create radar chart for per-robot performance.

        Args:
            metrics: Dict with keys 'precision', 'recall', 'f1' each containing
                     per-robot values
            step: Training step
            save: Whether to save

        Returns:
            Tuple of (figure, PIL image)
        """
        categories = self.ROBOT_NAMES
        n_cats = len(categories)

        # Create angles for radar chart
        angles = np.linspace(0, 2*np.pi, n_cats, endpoint=False).tolist()
        angles += angles[:1]  # Close the circle

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

        metric_names = ['precision', 'recall', 'f1']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

        for metric_name, color in zip(metric_names, colors):
            if metric_name in metrics:
                values = [metrics[metric_name].get(robot, 0) for robot in categories]
                values += values[:1]  # Close the circle

                ax.plot(angles, values, 'o-', linewidth=2, label=metric_name.upper(), color=color)
                ax.fill(angles, values, alpha=0.25, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=11)
        ax.set_ylim(0, 1)
        ax.set_title(f'Per-Robot Performance Metrics (Step {step})', size=14, y=1.1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

        plt.tight_layout()

        if save:
            save_path = self.output_dir / f"robot_radar_step{step}.png"
            fig.savefig(save_path, bbox_inches='tight', dpi=200)

        return fig, _fig_to_pil(fig)

    def plot_confidence_calibration(
        self,
        confidences: Union[torch.Tensor, np.ndarray],
        correct: Union[torch.Tensor, np.ndarray],
        step: int,
        n_bins: int = 10,
        save: bool = True,
    ) -> Tuple[plt.Figure, Image.Image]:
        """
        Plot confidence calibration diagram and ECE.

        Args:
            confidences: Model confidence scores
            correct: Boolean array of correct predictions
            step: Training step
            n_bins: Number of calibration bins
            save: Whether to save

        Returns:
            Tuple of (figure, PIL image)
        """
        if torch.is_tensor(confidences):
            confidences = confidences.cpu().numpy()
        if torch.is_tensor(correct):
            correct = correct.cpu().numpy()

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        bin_accuracies = []
        bin_confidences = []
        bin_counts = []

        for lower, upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > lower) & (confidences <= upper)
            if np.sum(in_bin) > 0:
                bin_accuracies.append(np.mean(correct[in_bin]))
                bin_confidences.append(np.mean(confidences[in_bin]))
                bin_counts.append(np.sum(in_bin))
            else:
                bin_accuracies.append(0)
                bin_confidences.append((lower + upper) / 2)
                bin_counts.append(0)

        # Calculate ECE
        ece = sum(count * abs(acc - conf) for count, acc, conf in
                  zip(bin_counts, bin_accuracies, bin_confidences)) / (sum(bin_counts) + 1e-8)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Reliability diagram
        ax1 = axes[0]
        bin_centers = (bin_lowers + bin_uppers) / 2
        ax1.bar(bin_centers, bin_accuracies, width=0.08, alpha=0.7,
               label='Accuracy', color='#2E86AB', edgecolor='white')
        ax1.plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')
        ax1.set_xlabel('Confidence')
        ax1.set_ylabel('Accuracy')
        ax1.set_title(f'Reliability Diagram | ECE: {ece:.4f}')
        ax1.legend()
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)

        # Confidence histogram
        ax2 = axes[1]
        ax2.hist(confidences, bins=20, alpha=0.7, color='#A23B72', edgecolor='white')
        ax2.axvline(np.mean(confidences), color='red', linestyle='--',
                   label=f'Mean: {np.mean(confidences):.3f}')
        ax2.set_xlabel('Confidence')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Confidence Distribution')
        ax2.legend()

        fig.suptitle(f'Calibration Analysis (Step {step})', fontsize=14)
        plt.tight_layout()

        if save:
            save_path = self.output_dir / f"calibration_step{step}.png"
            fig.savefig(save_path, bbox_inches='tight', dpi=200)

        return fig, _fig_to_pil(fig)

    def plot_reasoning_examples(
        self,
        tasks: List[str],
        reasoning_chains: List[str],
        predictions: List[str],
        ground_truth: List[str],
        correct: List[bool],
        step: int,
        n_examples: int = 6,
        save: bool = True,
    ) -> Tuple[plt.Figure, Image.Image]:
        """
        Display reasoning chain examples with correctness.

        Args:
            tasks: Task descriptions
            reasoning_chains: Generated reasoning
            predictions: Predicted robots
            ground_truth: True robots
            correct: Correctness flags
            step: Training step
            n_examples: Number to show
            save: Whether to save

        Returns:
            Tuple of (figure, PIL image)
        """
        n_examples = min(n_examples, len(tasks))

        # Show mix of correct and incorrect
        correct_idx = [i for i, c in enumerate(correct) if c]
        incorrect_idx = [i for i, c in enumerate(correct) if not c]

        # Balance examples
        n_correct = min(n_examples // 2, len(correct_idx))
        n_incorrect = min(n_examples - n_correct, len(incorrect_idx))

        selected = correct_idx[:n_correct] + incorrect_idx[:n_incorrect]

        fig, axes = plt.subplots(len(selected), 1, figsize=(16, 4*len(selected)))
        if len(selected) == 1:
            axes = [axes]

        for ax_idx, i in enumerate(selected):
            ax = axes[ax_idx]

            is_correct = correct[i]
            color = 'green' if is_correct else 'red'
            symbol = '✓' if is_correct else '✗'

            text = f"{symbol} Task: {tasks[i][:150]}...\n\n"
            text += f"Reasoning: {reasoning_chains[i][:300]}...\n\n"
            text += f"Predicted: {predictions[i]} | Ground Truth: {ground_truth[i]}"

            ax.text(0.02, 0.98, text, ha='left', va='top', wrap=True,
                   fontsize=10, transform=ax.transAxes,
                   bbox=dict(boxstyle='round', facecolor='white',
                            edgecolor=color, linewidth=2))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')

        fig.suptitle(f'Reasoning Chain Examples (Step {step})', fontsize=14)
        plt.tight_layout()

        if save:
            save_path = self.output_dir / f"reasoning_examples_step{step}.png"
            fig.savefig(save_path, bbox_inches='tight', dpi=200)

        return fig, _fig_to_pil(fig)


class Stage4Visualizer:
    """Visualizations for Stage 4: Chain-of-Thought Reasoning."""

    def __init__(self, output_dir: str = "./outputs/stage4/visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.phase1_metrics = []
        self.phase2_metrics = []

    def plot_reasoning_quality_metrics(
        self,
        coherence_scores: List[float],
        consistency_scores: List[float],
        step_counts: List[int],
        step: int,
        save: bool = True,
    ) -> Tuple[plt.Figure, Image.Image]:
        """
        Plot reasoning quality metrics dashboard.

        Args:
            coherence_scores: List of coherence scores
            consistency_scores: List of consistency scores
            step_counts: List of reasoning step counts
            step: Training step
            save: Whether to save

        Returns:
            Tuple of (figure, PIL image)
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Coherence distribution
        axes[0, 0].hist(coherence_scores, bins=20, alpha=0.7, color='#2E86AB', edgecolor='white')
        axes[0, 0].axvline(np.mean(coherence_scores), color='red', linestyle='--',
                          label=f'Mean: {np.mean(coherence_scores):.3f}')
        axes[0, 0].set_xlabel('Coherence Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Reasoning Coherence Distribution')
        axes[0, 0].legend()

        # Consistency distribution
        axes[0, 1].hist(consistency_scores, bins=20, alpha=0.7, color='#A23B72', edgecolor='white')
        axes[0, 1].axvline(np.mean(consistency_scores), color='red', linestyle='--',
                          label=f'Mean: {np.mean(consistency_scores):.3f}')
        axes[0, 1].set_xlabel('Consistency Score')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Logical Consistency Distribution')
        axes[0, 1].legend()

        # Step count distribution
        unique_steps, counts = np.unique(step_counts, return_counts=True)
        axes[1, 0].bar(unique_steps, counts, alpha=0.7, color='#4ECDC4', edgecolor='white')
        axes[1, 0].set_xlabel('Number of Reasoning Steps')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Reasoning Step Count Distribution')

        # Coherence vs Steps scatter
        axes[1, 1].scatter(step_counts, coherence_scores, alpha=0.5, s=30, c='#45B7D1')

        # Add trend line
        z = np.polyfit(step_counts, coherence_scores, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(step_counts), max(step_counts), 100)
        axes[1, 1].plot(x_line, p(x_line), 'r--', label='Trend')

        axes[1, 1].set_xlabel('Number of Reasoning Steps')
        axes[1, 1].set_ylabel('Coherence Score')
        axes[1, 1].set_title('Reasoning Steps vs Coherence')
        axes[1, 1].legend()

        fig.suptitle(f'Reasoning Quality Analysis (Step {step})', fontsize=14)
        plt.tight_layout()

        if save:
            save_path = self.output_dir / f"reasoning_quality_step{step}.png"
            fig.savefig(save_path, bbox_inches='tight', dpi=200)

        return fig, _fig_to_pil(fig)

    def plot_phase_comparison(
        self,
        phase1_history: Dict[str, List[float]],
        phase2_history: Dict[str, List[float]],
        step: int,
        save: bool = True,
    ) -> Tuple[plt.Figure, Image.Image]:
        """
        Compare Phase 1 and Phase 2 training metrics.

        Args:
            phase1_history: Phase 1 metrics history
            phase2_history: Phase 2 metrics history
            step: Current step
            save: Whether to save

        Returns:
            Tuple of (figure, PIL image)
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Loss comparison
        ax = axes[0, 0]
        if 'loss' in phase1_history:
            ax.plot(phase1_history['loss'], label='Phase 1', color='#2E86AB', linewidth=2)
        if 'loss' in phase2_history:
            offset = len(phase1_history.get('loss', []))
            x = np.arange(offset, offset + len(phase2_history['loss']))
            ax.plot(x, phase2_history['loss'], label='Phase 2', color='#A23B72', linewidth=2)
            ax.axvline(offset, color='gray', linestyle='--', alpha=0.5, label='Phase Transition')
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss by Phase')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Accuracy comparison
        ax = axes[0, 1]
        if 'robot_accuracy' in phase1_history:
            ax.plot(phase1_history['robot_accuracy'], label='Phase 1', color='#2E86AB', linewidth=2)
        if 'robot_accuracy' in phase2_history:
            offset = len(phase1_history.get('robot_accuracy', []))
            x = np.arange(offset, offset + len(phase2_history['robot_accuracy']))
            ax.plot(x, phase2_history['robot_accuracy'], label='Phase 2', color='#A23B72', linewidth=2)
            ax.axvline(offset, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Step')
        ax.set_ylabel('Accuracy')
        ax.set_title('Robot Selection Accuracy by Phase')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Reasoning loss
        ax = axes[1, 0]
        if 'consistency_loss' in phase1_history:
            ax.plot(phase1_history['consistency_loss'], label='Phase 1', color='#2E86AB', linewidth=2)
        if 'consistency_loss' in phase2_history:
            offset = len(phase1_history.get('consistency_loss', []))
            x = np.arange(offset, offset + len(phase2_history['consistency_loss']))
            ax.plot(x, phase2_history['consistency_loss'], label='Phase 2', color='#A23B72', linewidth=2)
            ax.axvline(offset, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Step')
        ax.set_ylabel('Consistency Loss')
        ax.set_title('Reasoning Consistency Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Summary bar chart
        ax = axes[1, 1]
        metrics = ['loss', 'robot_accuracy']
        phase1_final = [phase1_history.get(m, [0])[-1] if phase1_history.get(m) else 0 for m in metrics]
        phase2_final = [phase2_history.get(m, [0])[-1] if phase2_history.get(m) else 0 for m in metrics]

        x = np.arange(len(metrics))
        width = 0.35
        ax.bar(x - width/2, phase1_final, width, label='Phase 1 Final', color='#2E86AB')
        ax.bar(x + width/2, phase2_final, width, label='Phase 2 Final', color='#A23B72')
        ax.set_xticks(x)
        ax.set_xticklabels(['Loss', 'Accuracy'])
        ax.set_ylabel('Value')
        ax.set_title('Final Metrics Comparison')
        ax.legend()

        fig.suptitle(f'Phase 1 vs Phase 2 Training Comparison (Step {step})', fontsize=14)
        plt.tight_layout()

        if save:
            save_path = self.output_dir / f"phase_comparison_step{step}.png"
            fig.savefig(save_path, bbox_inches='tight', dpi=200)

        return fig, _fig_to_pil(fig)

    def plot_cot_examples(
        self,
        tasks: List[str],
        without_cot: List[Tuple[str, bool]],  # (prediction, correct)
        with_cot: List[Tuple[str, str, bool]],  # (reasoning, prediction, correct)
        step: int,
        n_examples: int = 4,
        save: bool = True,
    ) -> Tuple[plt.Figure, Image.Image]:
        """
        Compare predictions with and without Chain-of-Thought.

        Args:
            tasks: Task descriptions
            without_cot: (prediction, correct) without CoT
            with_cot: (reasoning, prediction, correct) with CoT
            step: Training step
            n_examples: Number of examples
            save: Whether to save

        Returns:
            Tuple of (figure, PIL image)
        """
        n_examples = min(n_examples, len(tasks))

        fig, axes = plt.subplots(n_examples, 2, figsize=(18, 4*n_examples))
        if n_examples == 1:
            axes = axes.reshape(1, -1)

        for i in range(n_examples):
            # Without CoT
            ax = axes[i, 0]
            pred, correct = without_cot[i]
            color = 'green' if correct else 'red'
            symbol = '✓' if correct else '✗'

            text = f"Task: {tasks[i][:150]}...\n\n"
            text += f"{symbol} Direct Prediction: {pred}"

            ax.text(0.02, 0.98, text, ha='left', va='top', wrap=True,
                   fontsize=10, transform=ax.transAxes,
                   bbox=dict(boxstyle='round', facecolor='#FFF3E0',
                            edgecolor=color, linewidth=2))
            ax.set_title('Without Chain-of-Thought', fontsize=11)
            ax.axis('off')

            # With CoT
            ax = axes[i, 1]
            reasoning, pred, correct = with_cot[i]
            color = 'green' if correct else 'red'
            symbol = '✓' if correct else '✗'

            text = f"Reasoning:\n{reasoning[:250]}...\n\n"
            text += f"{symbol} Prediction: {pred}"

            ax.text(0.02, 0.98, text, ha='left', va='top', wrap=True,
                   fontsize=10, transform=ax.transAxes,
                   bbox=dict(boxstyle='round', facecolor='#E8F5E9',
                            edgecolor=color, linewidth=2))
            ax.set_title('With Chain-of-Thought', fontsize=11)
            ax.axis('off')

        fig.suptitle(f'Chain-of-Thought Comparison (Step {step})', fontsize=14)
        plt.tight_layout()

        if save:
            save_path = self.output_dir / f"cot_comparison_step{step}.png"
            fig.savefig(save_path, bbox_inches='tight', dpi=200)

        return fig, _fig_to_pil(fig)


class CrossStageVisualizer:
    """Visualizations for cross-stage analysis."""

    def __init__(self, output_dir: str = "./outputs/visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.stage_metrics = {1: {}, 2: {}, 3: {}, 4: {}}

    def update_stage_metrics(self, stage: int, metrics: Dict[str, float]):
        """Update metrics for a stage."""
        for key, value in metrics.items():
            if key not in self.stage_metrics[stage]:
                self.stage_metrics[stage][key] = []
            self.stage_metrics[stage][key].append(value)

    def plot_stage_progression(
        self,
        eval_results: Dict[int, Dict[str, float]],
        save: bool = True,
    ) -> Tuple[plt.Figure, Image.Image]:
        """
        Plot performance progression across stages.

        Args:
            eval_results: {stage_num: {benchmark: score}}
            save: Whether to save

        Returns:
            Tuple of (figure, PIL image)
        """
        stages = sorted(eval_results.keys())
        benchmarks = list(eval_results[stages[0]].keys()) if stages else []

        fig, ax = plt.subplots(figsize=(14, 8))

        x = np.arange(len(benchmarks))
        width = 0.2

        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

        for i, stage in enumerate(stages):
            scores = [eval_results[stage].get(b, 0) for b in benchmarks]
            ax.bar(x + i*width, scores, width, label=f'Stage {stage}',
                  color=colors[i % len(colors)], alpha=0.8)

        ax.set_xlabel('Benchmark')
        ax.set_ylabel('Score')
        ax.set_title('Benchmark Performance Across Training Stages')
        ax.set_xticks(x + width * (len(stages) - 1) / 2)
        ax.set_xticklabels(benchmarks, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save:
            save_path = self.output_dir / "stage_progression.png"
            fig.savefig(save_path, bbox_inches='tight', dpi=300)

        return fig, _fig_to_pil(fig)

    def plot_training_summary(
        self,
        all_metrics: Dict[int, Dict[str, List[float]]],
        save: bool = True,
    ) -> Tuple[plt.Figure, Image.Image]:
        """
        Plot training summary across all stages.

        Args:
            all_metrics: {stage: {metric: [values]}}
            save: Whether to save

        Returns:
            Tuple of (figure, PIL image)
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        colors = {1: '#FF6B6B', 2: '#4ECDC4', 3: '#45B7D1', 4: '#96CEB4'}

        # Loss progression
        ax = axes[0, 0]
        offset = 0
        for stage in sorted(all_metrics.keys()):
            if 'loss' in all_metrics[stage]:
                values = all_metrics[stage]['loss']
                x = np.arange(offset, offset + len(values))
                ax.plot(x, values, label=f'Stage {stage}', color=colors[stage], linewidth=2)
                offset += len(values)
        ax.set_xlabel('Global Step')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss Across All Stages')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Learning rate
        ax = axes[0, 1]
        offset = 0
        for stage in sorted(all_metrics.keys()):
            if 'lr' in all_metrics[stage]:
                values = all_metrics[stage]['lr']
                x = np.arange(offset, offset + len(values))
                ax.plot(x, values, label=f'Stage {stage}', color=colors[stage], linewidth=2)
                offset += len(values)
        ax.set_xlabel('Global Step')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Per-stage final metrics
        ax = axes[1, 0]
        stage_names = [f'Stage {s}' for s in sorted(all_metrics.keys())]
        final_losses = [all_metrics[s].get('loss', [0])[-1] if all_metrics[s].get('loss') else 0
                       for s in sorted(all_metrics.keys())]

        ax.bar(stage_names, final_losses, color=[colors[s] for s in sorted(all_metrics.keys())])
        ax.set_ylabel('Final Loss')
        ax.set_title('Final Loss Per Stage')
        ax.grid(True, alpha=0.3, axis='y')

        # Training time (if available)
        ax = axes[1, 1]
        if any('training_time' in all_metrics[s] for s in all_metrics):
            times = [sum(all_metrics[s].get('training_time', [0])) for s in sorted(all_metrics.keys())]
            ax.pie(times, labels=stage_names, colors=[colors[s] for s in sorted(all_metrics.keys())],
                  autopct='%1.1f%%', startangle=90)
            ax.set_title('Training Time Distribution')
        else:
            ax.text(0.5, 0.5, 'Training time not tracked', ha='center', va='center')
            ax.axis('off')

        fig.suptitle('EmberVLM Training Summary', fontsize=16)
        plt.tight_layout()

        if save:
            save_path = self.output_dir / "training_summary.png"
            fig.savefig(save_path, bbox_inches='tight', dpi=300)

        return fig, _fig_to_pil(fig)

    def plot_carbon_footprint(
        self,
        emissions_per_stage: Dict[int, float],
        save: bool = True,
    ) -> Tuple[plt.Figure, Image.Image]:
        """
        Plot carbon emissions by stage.

        Args:
            emissions_per_stage: {stage: kg_co2}
            save: Whether to save

        Returns:
            Tuple of (figure, PIL image)
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        stages = [f'Stage {s}' for s in sorted(emissions_per_stage.keys())]
        emissions = [emissions_per_stage[s] for s in sorted(emissions_per_stage.keys())]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

        # Bar chart
        axes[0].bar(stages, emissions, color=colors[:len(stages)], edgecolor='white')
        axes[0].set_ylabel('CO₂ Emissions (kg)')
        axes[0].set_title('Carbon Footprint by Stage')
        axes[0].grid(True, alpha=0.3, axis='y')

        # Cumulative line
        cumulative = np.cumsum(emissions)
        axes[1].plot(stages, cumulative, 'o-', linewidth=2, markersize=10, color='#2E86AB')
        axes[1].fill_between(stages, 0, cumulative, alpha=0.3, color='#2E86AB')
        axes[1].set_ylabel('Cumulative CO₂ (kg)')
        axes[1].set_title(f'Cumulative Emissions: {cumulative[-1]:.3f} kg CO₂')
        axes[1].grid(True, alpha=0.3)

        fig.suptitle('EmberVLM Carbon Footprint Analysis', fontsize=14)
        plt.tight_layout()

        if save:
            save_path = self.output_dir / "carbon_footprint.png"
            fig.savefig(save_path, bbox_inches='tight', dpi=300)

        return fig, _fig_to_pil(fig)

