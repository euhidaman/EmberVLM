"""
EmberVLM Vision Encoder: Frozen RepViT-XXS wrapper
Provides efficient visual feature extraction for multimodal understanding.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class RepViTBlock(nn.Module):
    """Basic RepViT block with squeeze-excite and residual connections."""

    def __init__(self, inp: int, hidden_dim: int, oup: int, kernel_size: int,
                 stride: int, use_se: bool = True, use_hs: bool = True):
        super().__init__()
        self.stride = stride
        self.identity = stride == 1 and inp == oup

        if stride == 2:
            self.token_mixer = nn.Sequential(
                # Depthwise convolution
                nn.Conv2d(inp, inp, kernel_size, stride, (kernel_size - 1) // 2,
                          groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup)
            )
            self.channel_mixer = nn.Sequential(
                nn.Conv2d(oup, 2 * oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(2 * oup),
                nn.GELU(),
                nn.Conv2d(2 * oup, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup)
            )
        else:
            self.token_mixer = nn.Sequential(
                RepVGGDW(inp),
                SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
            )
            self.channel_mixer = nn.Sequential(
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.token_mixer(x)
        out = out + self.channel_mixer(out) if self.identity else self.channel_mixer(out)
        return out


class SqueezeExcite(nn.Module):
    """Squeeze-and-Excitation module for channel attention."""

    def __init__(self, channels: int, reduction_ratio: float = 0.25):
        super().__init__()
        reduced_channels = max(1, int(channels * reduction_ratio))
        self.fc1 = nn.Conv2d(channels, reduced_channels, 1)
        self.fc2 = nn.Conv2d(reduced_channels, channels, 1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = x.mean(dim=(2, 3), keepdim=True)
        scale = self.fc2(self.act(self.fc1(scale)))
        return x * scale.sigmoid()


class RepVGGDW(nn.Module):
    """RepVGG depthwise block with reparameterization support."""

    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False),
            nn.BatchNorm2d(dim)
        )
        self.conv1 = nn.Conv2d(dim, dim, 1, 1, 0, groups=dim, bias=False)
        self.bn = nn.BatchNorm2d(dim)
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(x) + self.conv1(x) + x)


class RepViTXXS(nn.Module):
    """
    RepViT-XXS-M0.9: Ultra-lightweight vision encoder (~5M params)
    Based on RepViT architecture with smallest M0.9 configuration.

    Output: 384-dimensional features per image
    """

    # Configuration for M0.9 variant
    CFGS_M09 = [
        # kernel, expansion, channels, use_se, use_hs, stride
        [3, 2, 48, 1, 0, 1],
        [3, 2, 48, 0, 0, 1],
        [3, 2, 48, 0, 0, 1],
        [3, 2, 96, 0, 0, 2],
        [3, 2, 96, 1, 0, 1],
        [3, 2, 96, 0, 0, 1],
        [3, 2, 96, 0, 0, 1],
        [3, 2, 192, 0, 1, 2],
        [3, 2, 192, 1, 1, 1],
        [3, 2, 192, 0, 1, 1],
        [3, 2, 192, 1, 1, 1],
        [3, 2, 192, 0, 1, 1],
        [3, 2, 192, 1, 1, 1],
        [3, 2, 192, 0, 1, 1],
        [3, 2, 192, 1, 1, 1],
        [3, 2, 192, 0, 1, 1],
        [3, 2, 192, 1, 1, 1],
        [3, 2, 192, 0, 1, 1],
        [3, 2, 192, 1, 1, 1],
        [3, 2, 192, 0, 1, 1],
        [3, 2, 192, 1, 1, 1],
        [3, 2, 192, 0, 1, 1],
        [3, 2, 192, 0, 1, 1],
        [3, 2, 384, 0, 1, 2],
        [3, 2, 384, 1, 1, 1],
        [3, 2, 384, 0, 1, 1]
    ]

    def __init__(
        self,
        num_classes: int = 0,  # 0 = feature extraction mode
        output_dim: int = 384,
        num_vision_tokens: int = 8,
        pretrained: bool = True,
        freeze: bool = True
    ):
        super().__init__()

        self.output_dim = output_dim
        self.num_vision_tokens = num_vision_tokens

        # Build patch embedding
        input_channel = self.CFGS_M09[0][2]
        self.patch_embed = nn.Sequential(
            nn.Conv2d(3, input_channel // 2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel // 2),
            nn.GELU(),
            nn.Conv2d(input_channel // 2, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel)
        )

        # Build feature extraction layers
        self.features = nn.ModuleList()
        for k, t, c, use_se, use_hs, s in self.CFGS_M09:
            output_channel = self._make_divisible(c, 8)
            exp_size = self._make_divisible(input_channel * t, 8)
            self.features.append(
                RepViTBlock(input_channel, exp_size, output_channel, k, s, use_se, use_hs)
            )
            input_channel = output_channel

        self.final_channels = output_channel

        # Global pooling and projection
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Spatial pooling for vision tokens (from HxW -> num_vision_tokens)
        self.spatial_pool = nn.AdaptiveAvgPool2d(
            int(num_vision_tokens ** 0.5)  # Square grid
        )

        # Optional projection to match LM hidden size
        self.projection = nn.Identity()  # Can be replaced in fusion module

        if pretrained:
            self._load_pretrained_weights()

        if freeze:
            self._freeze_parameters()

    def _make_divisible(self, v: int, divisor: int, min_value: int = None) -> int:
        """Ensure channel count is divisible by divisor."""
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v

    def _load_pretrained_weights(self):
        """Load pretrained weights from timm or local checkpoint."""
        try:
            import timm
            # Try to load from timm
            pretrained_model = timm.create_model('repvit_m0_9', pretrained=True)
            self.load_state_dict(pretrained_model.state_dict(), strict=False)
            logger.info("Loaded pretrained RepViT-M0.9 weights from timm")
        except Exception as e:
            logger.warning(f"Could not load pretrained weights: {e}")
            logger.info("Initializing with random weights")

    def _freeze_parameters(self):
        """Freeze all parameters for feature extraction only."""
        for param in self.parameters():
            param.requires_grad = False
        logger.info("Froze all RepViT vision encoder parameters")

    def forward(
        self,
        pixel_values: torch.Tensor,
        return_spatial: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through vision encoder.

        Args:
            pixel_values: Input images [B, 3, H, W]
            return_spatial: Whether to return spatial features for attention viz

        Returns:
            vision_features: [B, num_vision_tokens, output_dim]
            spatial_features: [B, output_dim, h, w] if return_spatial else None
        """
        # Patch embedding
        x = self.patch_embed(pixel_values)

        # Feature extraction
        for block in self.features:
            x = block(x)

        spatial_features = x if return_spatial else None

        # Spatial pooling to vision tokens
        # x: [B, C, H, W] -> [B, C, sqrt(n), sqrt(n)] -> [B, n, C]
        x = self.spatial_pool(x)
        B, C, H, W = x.shape
        x = x.view(B, C, -1).permute(0, 2, 1)  # [B, n_tokens, C]

        # Ensure correct number of tokens
        assert x.shape[1] == self.num_vision_tokens, \
            f"Expected {self.num_vision_tokens} tokens, got {x.shape[1]}"

        return x, spatial_features

    def get_output_dim(self) -> int:
        """Return output feature dimension."""
        return self.final_channels

    def count_parameters(self) -> Tuple[int, int]:
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


class VisionEncoderWrapper(nn.Module):
    """
    Wrapper for vision encoder with image preprocessing.
    Handles normalization and provides clean interface for EmberVLM.
    """

    def __init__(
        self,
        encoder_type: str = "repvit_m0_9",
        num_vision_tokens: int = 8,
        pretrained: bool = True,
        freeze: bool = True,
        normalize_mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        normalize_std: Tuple[float, ...] = (0.229, 0.224, 0.225)
    ):
        super().__init__()

        self.encoder = RepViTXXS(
            num_vision_tokens=num_vision_tokens,
            pretrained=pretrained,
            freeze=freeze
        )

        # Register normalization constants as buffers
        self.register_buffer(
            'mean',
            torch.tensor(normalize_mean).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'std',
            torch.tensor(normalize_std).view(1, 3, 1, 1)
        )

        self.output_dim = self.encoder.get_output_dim()
        self.num_vision_tokens = num_vision_tokens

    def preprocess(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values to ImageNet statistics."""
        # Assume input is in [0, 1] range
        return (pixel_values - self.mean) / self.std

    def forward(
        self,
        pixel_values: torch.Tensor,
        normalize: bool = True,
        return_spatial: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Extract vision features from images.

        Args:
            pixel_values: Images [B, 3, H, W], values in [0, 1]
            normalize: Whether to apply ImageNet normalization
            return_spatial: Whether to return spatial features for visualization

        Returns:
            vision_features: [B, num_vision_tokens, output_dim]
            spatial_features: Optional spatial feature map
        """
        if normalize:
            pixel_values = self.preprocess(pixel_values)

        return self.encoder(pixel_values, return_spatial=return_spatial)


def create_vision_encoder(config: dict) -> VisionEncoderWrapper:
    """Factory function to create vision encoder from config."""
    return VisionEncoderWrapper(
        encoder_type=config.get('type', 'repvit_m0_9'),
        num_vision_tokens=config.get('num_vision_tokens', 8),
        pretrained=config.get('pretrained', True),
        freeze=config.get('frozen', True)
    )


if __name__ == "__main__":
    # Test the vision encoder
    print("Testing RepViT Vision Encoder...")

    encoder = VisionEncoderWrapper(
        num_vision_tokens=8,
        pretrained=False,  # Don't download for test
        freeze=True
    )

    # Test input
    dummy_input = torch.randn(2, 3, 224, 224)

    # Forward pass
    vision_features, spatial_features = encoder(dummy_input, return_spatial=True)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Vision features shape: {vision_features.shape}")
    print(f"Spatial features shape: {spatial_features.shape}")

    total, trainable = encoder.encoder.count_parameters()
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")

