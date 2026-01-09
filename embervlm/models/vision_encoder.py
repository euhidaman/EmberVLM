"""
RepViT Vision Encoder for EmberVLM

A lightweight vision encoder based on RepViT architecture,
optimized for edge deployment while maintaining strong visual understanding.

Also supports MobileViT-XS as an alternative vision backbone.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
from timm.models.layers import SqueezeExcite
from timm.models.vision_transformer import trunc_normal_

# Vision backbone type constants
VISION_BACKBONE_REPVIT = "repvit"
VISION_BACKBONE_MOBILEVIT_XS = "mobilevit_xs"


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """Ensure channel count is divisible by divisor."""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class Conv2d_BN(nn.Sequential):
    """Convolution with Batch Normalization."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bn_weight_init: float = 1.0,
    ):
        super().__init__()
        self.add_module(
            'c',
            nn.Conv2d(
                in_channels, out_channels, kernel_size,
                stride, padding, dilation, groups, bias=False
            )
        )
        self.add_module('bn', nn.BatchNorm2d(out_channels))
        nn.init.constant_(self.bn.weight, bn_weight_init)
        nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self) -> nn.Conv2d:
        """Fuse conv and bn for inference."""
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5

        m = nn.Conv2d(
            w.size(1) * c.groups, w.size(0), w.shape[2:],
            stride=c.stride, padding=c.padding,
            dilation=c.dilation, groups=c.groups,
            device=c.weight.device
        )
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class Residual(nn.Module):
    """Residual connection with optional dropout."""

    def __init__(self, module: nn.Module, drop: float = 0.0):
        super().__init__()
        self.m = module
        self.drop = drop

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.drop > 0:
            mask = torch.rand(x.size(0), 1, 1, 1, device=x.device)
            mask = mask.ge_(self.drop).div(1 - self.drop).detach()
            return x + self.m(x) * mask
        else:
            return x + self.m(x)

    @torch.no_grad()
    def fuse(self):
        if isinstance(self.m, Conv2d_BN):
            m = self.m.fuse()
            assert m.groups == m.in_channels
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = F.pad(identity, [1, 1, 1, 1])
            m.weight += identity.to(m.weight.device)
            return m
        elif isinstance(self.m, nn.Conv2d):
            m = self.m
            assert m.groups != m.in_channels
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = F.pad(identity, [1, 1, 1, 1])
            m.weight += identity.to(m.weight.device)
            return m
        else:
            return self


class RepVGGDW(nn.Module):
    """RepVGG-style depthwise convolution."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = Conv2d_BN(channels, channels, 3, 1, 1, groups=channels)
        self.conv1 = nn.Conv2d(channels, channels, 1, 1, 0, groups=channels)
        self.dim = channels
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn((self.conv(x) + self.conv1(x)) + x)

    @torch.no_grad()
    def fuse(self) -> nn.Conv2d:
        conv = self.conv.fuse()
        conv1 = self.conv1

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias if conv1.bias is not None else torch.zeros_like(
            conv_b)

        conv1_w = F.pad(conv1_w, [1, 1, 1, 1])
        identity = F.pad(
            torch.ones(conv1_w.shape[0], conv1_w.shape[1],
                       1, 1, device=conv1_w.device),
            [1, 1, 1, 1]
        )

        final_conv_w = conv_w + conv1_w + identity
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        bn = self.bn
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = conv.weight * w[:, None, None, None]
        b = bn.bias + (conv.bias - bn.running_mean) * \
            bn.weight / (bn.running_var + bn.eps) ** 0.5
        conv.weight.data.copy_(w)
        conv.bias.data.copy_(b)
        return conv


class RepViTBlock(nn.Module):
    """RepViT building block."""

    def __init__(
        self,
        inp: int,
        hidden_dim: int,
        oup: int,
        kernel_size: int,
        stride: int,
        use_se: bool,
        use_hs: bool,
    ):
        super().__init__()
        assert stride in [1, 2]
        self.identity = stride == 1 and inp == oup
        assert hidden_dim == 2 * inp

        if stride == 2:
            self.token_mixer = nn.Sequential(
                Conv2d_BN(inp, inp, kernel_size, stride,
                          (kernel_size - 1) // 2, groups=inp),
                SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
                Conv2d_BN(inp, oup, kernel_size=1, stride=1, padding=0)
            )
            self.channel_mixer = Residual(nn.Sequential(
                Conv2d_BN(oup, 2 * oup, 1, 1, 0),
                nn.GELU(),
                Conv2d_BN(2 * oup, oup, 1, 1, 0, bn_weight_init=0),
            ))
        else:
            assert self.identity
            self.token_mixer = nn.Sequential(
                RepVGGDW(inp),
                SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
            )
            self.channel_mixer = Residual(nn.Sequential(
                Conv2d_BN(inp, hidden_dim, 1, 1, 0),
                nn.GELU(),
                Conv2d_BN(hidden_dim, oup, 1, 1, 0, bn_weight_init=0),
            ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.channel_mixer(self.token_mixer(x))


class RepViT(nn.Module):
    """RepViT backbone network."""

    def __init__(
        self,
        cfgs: list,
        num_classes: int = 1000,
        distillation: bool = False,
    ):
        super().__init__()
        self.cfgs = cfgs

        # Build first layer (patch embedding)
        input_channel = self.cfgs[0][2]
        patch_embed = nn.Sequential(
            Conv2d_BN(3, input_channel // 2, 3, 2, 1),
            nn.GELU(),
            Conv2d_BN(input_channel // 2, input_channel, 3, 2, 1)
        )
        layers = [patch_embed]

        # Build RepViT blocks
        block = RepViTBlock
        for k, t, c, use_se, use_hs, s in self.cfgs:
            output_channel = _make_divisible(c, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            layers.append(block(input_channel, exp_size,
                          output_channel, k, s, use_se, use_hs))
            input_channel = output_channel

        self.features = nn.ModuleList(layers)
        self.num_features = output_channel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for f in self.features:
            x = f(x)
        return x

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return feature maps without classification head."""
        return self.forward(x)


def repvit_xxs(pretrained: bool = False) -> RepViT:
    """
    Construct RepViT-XXS model (extra extra small variant).
    This is optimized for edge deployment.
    """
    # Configuration for XXS variant
    cfgs = [
        # k, t, c, SE, HS, s
        [3, 2, 32, 1, 0, 1],
        [3, 2, 32, 0, 0, 1],
        [3, 2, 64, 0, 0, 2],
        [3, 2, 64, 1, 0, 1],
        [3, 2, 64, 0, 0, 1],
        [3, 2, 128, 0, 1, 2],
        [3, 2, 128, 1, 1, 1],
        [3, 2, 128, 0, 1, 1],
        [3, 2, 128, 1, 1, 1],
        [3, 2, 128, 0, 1, 1],
        [3, 2, 256, 0, 1, 2],
        [3, 2, 256, 1, 1, 1],
    ]
    model = RepViT(cfgs, num_classes=0, distillation=False)

    if pretrained:
        # Load pretrained weights from timm
        try:
            import timm
            # Use timm's repvit_m0_9 as a lightweight pretrained base
            # This is the closest match to xxs size while being available
            pretrained_model = timm.create_model(
                'repvit_m0_9.dist_450e_in1k', pretrained=True)

            # Extract compatible weights (feature extractor part)
            pretrained_dict = pretrained_model.state_dict()
            model_dict = model.state_dict()

            # Filter out incompatible keys (classifier head, size mismatches)
            compatible_dict = {}
            for k, v in pretrained_dict.items():
                if k in model_dict and v.shape == model_dict[k].shape:
                    compatible_dict[k] = v

            model_dict.update(compatible_dict)
            model.load_state_dict(model_dict, strict=False)
            print(
                f"Loaded {len(compatible_dict)}/{len(model_dict)} weights from timm/repvit_m0_9.dist_450e_in1k")
        except Exception as e:
            print(f"Warning: Could not load pretrained weights: {e}")

    return model


class RepViTEncoder(nn.Module):
    """
    RepViT Vision Encoder wrapper for EmberVLM.

    Handles image preprocessing, feature extraction, and adaptive pooling
    to produce a fixed number of visual tokens.
    """

    def __init__(
        self,
        model_name: str = "repvit_xxs",
        pretrained: bool = True,
        freeze: bool = True,
        num_visual_tokens: int = 8,
        output_dim: int = 384,
        image_size: int = 224,
    ):
        super().__init__()

        self.model_name = model_name
        self.num_visual_tokens = num_visual_tokens
        self.output_dim = output_dim
        self.image_size = image_size

        # Initialize backbone
        if model_name == "repvit_xxs":
            self.backbone = repvit_xxs(pretrained=pretrained)
            self.backbone_dim = 256  # XXS output dimension
        elif model_name.startswith("repvit_m"):
            # Use timm models directly
            import timm
            timm_model_name = f"{model_name}.dist_450e_in1k" if "dist" not in model_name else model_name
            self.backbone = timm.create_model(
                timm_model_name, pretrained=pretrained, num_classes=0)

            # Determine backbone output dimension based on model size
            # Note: These are the actual output dimensions from the timm models
            model_dims = {
                'repvit_m0_9': 384,  # Corrected: actual output is 384
                'repvit_m1_0': 384,  # Corrected: actual output is 384
                'repvit_m1_1': 384,
                'repvit_m1_5': 512,
                'repvit_m2_3': 640,
            }
            base_name = model_name.split('.')[0]
            self.backbone_dim = model_dims.get(base_name, 384)
            print(
                f"Using timm model: {timm_model_name} with output_dim={self.backbone_dim}")
        else:
            raise ValueError(
                f"Unknown model: {model_name}. Use 'repvit_xxs' or timm model like 'repvit_m0_9'")

        # Adaptive pooling to get fixed number of tokens
        pool_size = int(num_visual_tokens ** 0.5)
        if pool_size ** 2 != num_visual_tokens:
            # Non-square pooling
            self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 4))  # 8 tokens
        else:
            self.adaptive_pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))

        # Projection to output dimension
        self.projection = nn.Sequential(
            nn.Linear(self.backbone_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

        # Layer norm for features
        self.ln_vision = nn.LayerNorm(self.backbone_dim)

        # Freeze backbone if specified
        if freeze:
            self._freeze_backbone()

        # Initialize projection
        self._init_weights()

    def _freeze_backbone(self):
        """Freeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

    def _init_weights(self):
        """Initialize projection weights."""
        for module in self.projection.modules():
            if isinstance(module, nn.Linear):
                trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)

    def forward(
        self,
        pixel_values: torch.Tensor,
        return_dict: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through vision encoder.

        Args:
            pixel_values: Input images [B, C, H, W]
            return_dict: Whether to return a dictionary

        Returns:
            Dictionary containing:
                - visual_tokens: Visual embeddings [B, num_tokens, output_dim]
                - pooled_output: Global pooled features [B, output_dim]
        """
        batch_size = pixel_values.size(0)

        # Extract features from backbone
        with torch.set_grad_enabled(not self.backbone.training):
            features = self.backbone.forward_features(pixel_values)

        # features shape: [B, C, H, W]
        # Adaptive pooling first (before normalization)
        pooled_features = self.adaptive_pool(features)  # [B, C, h, w]

        # Reshape to sequence of tokens for normalization
        B, C, h, w = pooled_features.shape
        visual_tokens = pooled_features.permute(
            0, 2, 3, 1).reshape(B, h * w, C)

        # Apply layer norm in the correct dimension
        visual_tokens = self.ln_vision(visual_tokens)  # [B, h*w, C]

        # Project to output dimension
        visual_tokens = self.projection(visual_tokens)

        # Global pooled output (from original features, not pooled)
        pooled_output = F.adaptive_avg_pool2d(features, 1).flatten(1)  # [B, C]
        pooled_output = self.projection[0](pooled_output)

        if return_dict:
            return {
                'visual_tokens': visual_tokens,
                'pooled_output': pooled_output,
            }
        return visual_tokens

    def get_num_visual_tokens(self) -> int:
        """Return number of visual tokens."""
        return self.num_visual_tokens

    def get_output_dim(self) -> int:
        """Return output dimension."""
        return self.output_dim

    @property
    def device(self) -> torch.device:
        """Get device of model parameters."""
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        """Get dtype of model parameters."""
        return next(self.parameters()).dtype


class MobileViTEncoder(nn.Module):
    """
    MobileViT Vision Encoder wrapper for EmberVLM.

    Uses official Apple MobileViT-XS (~2.3M params) from HuggingFace as an alternative to RepViT.
    Model: apple/mobilevit-x-small
    Handles image preprocessing, feature extraction, and adaptive pooling
    to produce a fixed number of visual tokens.

    MobileViT-XS output dimension is 384, matching RepViT and the default
    fusion module configuration.
    """

    def __init__(
        self,
        model_name: str = "apple/mobilevit-x-small",
        pretrained: bool = True,
        freeze: bool = True,
        num_visual_tokens: int = 8,
        output_dim: int = 384,
        image_size: int = 256,  # MobileViT default is 256
    ):
        super().__init__()

        try:
            from transformers import AutoModel, AutoImageProcessor
            HF_AVAILABLE = True
        except ImportError:
            HF_AVAILABLE = False
            AutoModel = None
            AutoImageProcessor = None

        if not HF_AVAILABLE:
            raise ImportError(
                "transformers library is required for MobileViTEncoder. "
                "Install with: pip install transformers"
            )

        self.model_name = model_name
        self.num_visual_tokens = num_visual_tokens
        self.output_dim = output_dim
        self.image_size = image_size

        # Load official Apple MobileViT from HuggingFace
        # Default: apple/mobilevit-x-small (~2.3M params)
        print(
            f"Loading official Apple MobileViT from HuggingFace: {model_name}...")

        if pretrained:
            self.backbone = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=False,
            )
        else:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_name)
            self.backbone = AutoModel.from_config(config)

        # MobileViT-XS output dimension is 384
        # Last layer hidden size
        self.backbone_dim = self.backbone.config.hidden_sizes[-1]
        print(f"Loaded MobileViT with output_dim={self.backbone_dim}")

        # Adaptive pooling to get fixed number of tokens
        pool_size = int(num_visual_tokens ** 0.5)
        if pool_size ** 2 != num_visual_tokens:
            # Non-square pooling
            self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 4))  # 8 tokens
        else:
            self.adaptive_pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))

        # Projection to output dimension (if backbone_dim != output_dim)
        if self.backbone_dim != output_dim:
            self.projection = nn.Sequential(
                nn.Linear(self.backbone_dim, output_dim),
                nn.LayerNorm(output_dim),
            )
        else:
            self.projection = nn.Sequential(
                nn.Linear(self.backbone_dim, output_dim),
                nn.LayerNorm(output_dim),
            )

        # Layer norm for features
        self.ln_vision = nn.LayerNorm(self.backbone_dim)

        # Freeze backbone if specified
        if freeze:
            self._freeze_backbone()

        # Initialize projection
        self._init_weights()

    def _freeze_backbone(self):
        """Freeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

    def _init_weights(self):
        """Initialize projection weights."""
        for module in self.projection.modules():
            if isinstance(module, nn.Linear):
                trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)

    def forward(
        self,
        pixel_values: torch.Tensor,
        return_dict: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through vision encoder.

        Args:
            pixel_values: Input images [B, C, H, W]
            return_dict: Whether to return a dictionary

        Returns:
            Dictionary containing:
                - visual_tokens: Visual embeddings [B, num_tokens, output_dim]
                - pooled_output: Global pooled features [B, output_dim]
        """
        batch_size = pixel_values.size(0)

        # Extract features from HuggingFace MobileViT backbone
        with torch.set_grad_enabled(not self.backbone.training):
            outputs = self.backbone(
                pixel_values, output_hidden_states=True, return_dict=True)
            # Get last hidden state from the convolutional layers
            # HuggingFace MobileViT returns last_hidden_state in [B, C, H, W] format
            features = outputs.last_hidden_state

        # features shape: [B, C, H, W]
        # Adaptive pooling
        pooled_features = self.adaptive_pool(features)  # [B, C, h, w]

        # Reshape to sequence of tokens
        B, C, h, w = pooled_features.shape
        visual_tokens = pooled_features.permute(
            0, 2, 3, 1).reshape(B, h * w, C)

        # Apply layer norm
        visual_tokens = self.ln_vision(visual_tokens)  # [B, h*w, C]

        # Project to output dimension
        visual_tokens = self.projection(visual_tokens)

        # Global pooled output (from original features)
        pooled_output = F.adaptive_avg_pool2d(features, 1).flatten(1)  # [B, C]
        pooled_output = self.projection[0](pooled_output)

        if return_dict:
            return {
                'visual_tokens': visual_tokens,
                'pooled_output': pooled_output,
            }
        return visual_tokens

    def get_num_visual_tokens(self) -> int:
        """Return number of visual tokens."""
        return self.num_visual_tokens

    def get_output_dim(self) -> int:
        """Return output dimension."""
        return self.output_dim

    @property
    def device(self) -> torch.device:
        """Get device of model parameters."""
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        """Get dtype of model parameters."""
        return next(self.parameters()).dtype


def create_vision_encoder(
    backbone_type: str = VISION_BACKBONE_REPVIT,
    model_name: Optional[str] = None,
    pretrained: bool = True,
    freeze: bool = True,
    num_visual_tokens: int = 8,
    output_dim: int = 384,
    image_size: int = 224,
) -> nn.Module:
    """
    Factory function to create vision encoder.

    Args:
        backbone_type: Vision backbone type ('repvit' or 'mobilevit_xs')
        model_name: Specific model name (overrides backbone_type defaults)
        pretrained: Whether to load pretrained weights
        freeze: Whether to freeze backbone parameters
        num_visual_tokens: Number of visual tokens to output
        output_dim: Output dimension for visual tokens
        image_size: Input image size

    Returns:
        Vision encoder module (RepViTEncoder or MobileViTEncoder)
    """
    if backbone_type == VISION_BACKBONE_MOBILEVIT_XS or (model_name and 'mobilevit' in model_name):
        actual_model_name = model_name if model_name else "apple/mobilevit-x-small"
        # MobileViT default image size is 256
        actual_image_size = image_size if image_size != 224 else 256
        return MobileViTEncoder(
            model_name=actual_model_name,
            pretrained=pretrained,
            freeze=freeze,
            num_visual_tokens=num_visual_tokens,
            output_dim=output_dim,
            image_size=actual_image_size,
        )
    else:
        # Default to RepViT
        actual_model_name = model_name if model_name else "repvit_m0_9"
        return RepViTEncoder(
            model_name=actual_model_name,
            pretrained=pretrained,
            freeze=freeze,
            num_visual_tokens=num_visual_tokens,
            output_dim=output_dim,
            image_size=image_size,
        )


# Image preprocessing utilities
class ImagePreprocessor:
    """Preprocessor for vision encoder input."""

    def __init__(
        self,
        image_size: int = 224,
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225),
    ):
        self.image_size = image_size
        self.mean = mean
        self.std = std

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """
        Preprocess images for vision encoder.

        Args:
            images: Raw images [B, C, H, W] in range [0, 1]

        Returns:
            Normalized images ready for encoder
        """
        # Resize if needed
        if images.shape[-2:] != (self.image_size, self.image_size):
            images = F.interpolate(
                images,
                size=(self.image_size, self.image_size),
                mode='bilinear',
                align_corners=False,
            )

        # Normalize
        mean = torch.tensor(
            self.mean, device=images.device, dtype=images.dtype)
        std = torch.tensor(self.std, device=images.device, dtype=images.dtype)
        images = (images - mean[None, :, None, None]) / \
            std[None, :, None, None]

        return images
