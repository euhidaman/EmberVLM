"""
EmberVLM Attention Visualization
Generate heatmaps showing text-image alignment for interpretability.
Inspired by LeJEPA's attention visualization.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Any
import io

import torch
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger(__name__)

# Optional imports for visualization
try:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None
    cm = None

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None


class AttentionVisualizer:
    """
    Visualize cross-modal attention between vision and language.

    Provides:
    1. Text-to-Image attention heatmaps
    2. Image-to-Text attention weights
    3. Cross-modal alignment visualization
    4. Reasoning trace visualization
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any = None,
        output_dir: str = "visualizations",
        colormap: str = "viridis"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.colormap = colormap

        if not MATPLOTLIB_AVAILABLE:
            logger.warning("matplotlib not available. Visualization will be limited.")

    def extract_attention_weights(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Extract attention weights from model forward pass.

        Returns:
            Dict with:
            - cross_attention: Vision-to-language attention [B, H, L_text, L_vision]
            - self_attention: Language self-attention [B, H, L, L]
            - vision_attention: Vision self-attention (if available)
        """
        self.model.eval()

        with torch.no_grad():
            outputs = self.model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True
            )

        result = {}

        # Cross-attention weights (from fusion module)
        if 'cross_attention_weights' in outputs:
            result['cross_attention'] = outputs['cross_attention_weights']

        # Self-attention from language model
        if 'attentions' in outputs and outputs['attentions']:
            result['self_attention'] = outputs['attentions']

        # Vision features for spatial visualization
        if 'vision_features' in outputs:
            result['vision_features'] = outputs['vision_features']

        return result

    def create_attention_heatmap(
        self,
        attention_weights: torch.Tensor,
        image: torch.Tensor,
        spatial_size: Tuple[int, int] = (7, 7),
        normalize: bool = True
    ) -> np.ndarray:
        """
        Create heatmap overlay on image.

        Args:
            attention_weights: [n_vision_tokens] attention from text to vision
            image: Original image tensor [3, H, W]
            spatial_size: Grid size for attention tokens
            normalize: Whether to normalize attention

        Returns:
            Heatmap overlaid on image as numpy array
        """
        if not MATPLOTLIB_AVAILABLE or not PIL_AVAILABLE:
            return np.zeros((224, 224, 3), dtype=np.uint8)

        # Process attention weights
        attn = attention_weights.cpu().numpy()

        if normalize:
            attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)

        # Reshape to spatial grid
        h, w = spatial_size
        attn_map = attn.reshape(h, w)

        # Upsample to image size
        image_size = image.shape[1:]  # [H, W]
        attn_map = np.array(
            Image.fromarray((attn_map * 255).astype(np.uint8)).resize(
                (image_size[1], image_size[0]),
                resample=Image.BILINEAR
            )
        ) / 255.0

        # Apply colormap
        colormap = cm.get_cmap(self.colormap)
        heatmap = colormap(attn_map)[..., :3]  # Remove alpha channel

        # Convert image to numpy [H, W, 3]
        img_np = image.cpu().numpy().transpose(1, 2, 0)

        # Denormalize if needed
        if img_np.max() <= 1:
            img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = img_np.astype(np.uint8)

        # Blend heatmap with image
        alpha = 0.6
        blended = (alpha * heatmap * 255 + (1 - alpha) * img_np).astype(np.uint8)

        return blended

    def visualize_text_to_image_attention(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_indices: Optional[List[int]] = None,
        save_path: Optional[str] = None
    ) -> Dict[str, np.ndarray]:
        """
        Visualize which image regions each text token attends to.

        Args:
            pixel_values: [B, 3, H, W] batch of images
            input_ids: [B, L] text token IDs
            attention_mask: [B, L] attention mask
            token_indices: Specific tokens to visualize (None = all)
            save_path: Path to save visualization

        Returns:
            Dict mapping token index to heatmap
        """
        # Extract attention
        attention_data = self.extract_attention_weights(
            pixel_values, input_ids, attention_mask
        )

        if 'cross_attention' not in attention_data:
            logger.warning("No cross-attention available")
            return {}

        cross_attn = attention_data['cross_attention']

        # Handle list of attention layers
        if isinstance(cross_attn, list):
            # Average across layers
            cross_attn = torch.stack(cross_attn).mean(dim=0)

        # Average across heads: [B, L_text, L_vision]
        cross_attn = cross_attn.mean(dim=1) if cross_attn.dim() == 4 else cross_attn

        results = {}

        # Use first batch item
        attn = cross_attn[0]  # [L_text, L_vision]
        image = pixel_values[0]  # [3, H, W]

        # Get number of vision tokens
        n_vision_tokens = attn.shape[-1]
        spatial_size = int(np.sqrt(n_vision_tokens))
        spatial_size = (spatial_size, spatial_size)

        # Determine which tokens to visualize
        if token_indices is None:
            token_indices = list(range(min(10, attn.shape[0])))  # First 10 tokens

        for idx in token_indices:
            if idx >= attn.shape[0]:
                continue

            token_attn = attn[idx]  # [L_vision]
            heatmap = self.create_attention_heatmap(
                token_attn, image, spatial_size
            )
            results[idx] = heatmap

        # Save if requested
        if save_path and MATPLOTLIB_AVAILABLE:
            self._save_attention_grid(results, input_ids[0], save_path)

        return results

    def visualize_reasoning_trace(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        generated_ids: torch.Tensor,
        save_path: Optional[str] = None
    ) -> List[np.ndarray]:
        """
        Visualize attention flow through reasoning steps.

        Shows how attention shifts as the model generates reasoning.
        """
        if not MATPLOTLIB_AVAILABLE:
            return []

        frames = []

        # Generate attention maps at key points during generation
        generation_steps = [0, len(generated_ids) // 4, len(generated_ids) // 2,
                          3 * len(generated_ids) // 4, len(generated_ids) - 1]

        for step in generation_steps:
            step = min(step, len(generated_ids) - 1)
            partial_ids = generated_ids[:, :step + 1]

            attention_data = self.extract_attention_weights(
                pixel_values, partial_ids
            )

            if 'cross_attention' in attention_data:
                cross_attn = attention_data['cross_attention']
                if isinstance(cross_attn, list):
                    cross_attn = torch.stack(cross_attn).mean(dim=0)

                # Get attention for last generated token
                attn = cross_attn[0, -1] if cross_attn.dim() >= 2 else cross_attn[0]

                n_vision = attn.shape[-1]
                spatial = int(np.sqrt(n_vision))

                heatmap = self.create_attention_heatmap(
                    attn.mean(dim=0) if attn.dim() > 1 else attn,
                    pixel_values[0],
                    (spatial, spatial)
                )
                frames.append(heatmap)

        if save_path and frames:
            self._save_reasoning_gif(frames, save_path)

        return frames

    def _save_attention_grid(
        self,
        heatmaps: Dict[int, np.ndarray],
        input_ids: torch.Tensor,
        save_path: str
    ):
        """Save attention heatmaps as a grid."""
        if not MATPLOTLIB_AVAILABLE:
            return

        n_plots = len(heatmaps)
        if n_plots == 0:
            return

        cols = min(5, n_plots)
        rows = (n_plots + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
        if n_plots == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for i, (idx, heatmap) in enumerate(heatmaps.items()):
            ax = axes[i]
            ax.imshow(heatmap)

            # Get token text if tokenizer available
            if self.tokenizer:
                token_text = self.tokenizer.decode([input_ids[idx].item()])
            else:
                token_text = f"Token {idx}"
            ax.set_title(token_text, fontsize=8)
            ax.axis('off')

        # Hide unused subplots
        for i in range(len(heatmaps), len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved attention grid to {save_path}")

    def _save_reasoning_gif(
        self,
        frames: List[np.ndarray],
        save_path: str,
        duration: int = 500
    ):
        """Save reasoning trace as animated GIF."""
        if not PIL_AVAILABLE or not frames:
            return

        pil_frames = [Image.fromarray(f) for f in frames]

        pil_frames[0].save(
            save_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration,
            loop=0
        )

        logger.info(f"Saved reasoning GIF to {save_path}")

    def log_to_wandb(
        self,
        heatmaps: Dict[int, np.ndarray],
        step: int,
        prefix: str = "attention"
    ):
        """Log attention visualizations to WandB."""
        try:
            import wandb

            images = {}
            for idx, heatmap in heatmaps.items():
                images[f"{prefix}/token_{idx}"] = wandb.Image(heatmap)

            wandb.log(images, step=step)

        except ImportError:
            logger.debug("WandB not available for logging")


def create_attention_visualizer(
    model: Any,
    tokenizer: Any = None,
    output_dir: str = "visualizations"
) -> AttentionVisualizer:
    """Factory function for creating attention visualizer."""
    return AttentionVisualizer(
        model=model,
        tokenizer=tokenizer,
        output_dir=output_dir
    )


if __name__ == "__main__":
    # Test attention visualization
    print("Testing Attention Visualization Module...")

    # Create mock model and data
    class MockModel:
        def __init__(self):
            pass

        def eval(self):
            pass

        def __call__(self, **kwargs):
            B = kwargs.get('pixel_values', torch.randn(1, 3, 224, 224)).shape[0]
            L = kwargs.get('input_ids', torch.randint(0, 100, (1, 32))).shape[1]

            return {
                'logits': torch.randn(B, L, 50257),
                'cross_attention_weights': [torch.randn(B, 12, L, 8).softmax(dim=-1)],
                'attentions': [torch.randn(B, 12, L, L).softmax(dim=-1)],
                'vision_features': torch.randn(B, 8, 384)
            }

    model = MockModel()
    visualizer = AttentionVisualizer(model, output_dir="test_viz")

    # Test extraction
    pixel_values = torch.randn(2, 3, 224, 224)
    input_ids = torch.randint(0, 50257, (2, 32))

    attention_data = visualizer.extract_attention_weights(pixel_values, input_ids)
    print(f"Extracted attention keys: {attention_data.keys()}")

    # Test visualization
    if MATPLOTLIB_AVAILABLE:
        heatmaps = visualizer.visualize_text_to_image_attention(
            pixel_values, input_ids,
            token_indices=[0, 1, 2],
            save_path="test_viz/test_attention.png"
        )
        print(f"Generated {len(heatmaps)} heatmaps")
    else:
        print("Matplotlib not available, skipping visualization test")

    print("Attention visualization tests complete!")

