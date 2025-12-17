"""
EmberVLM GGUF Conversion
Convert PyTorch model to llama.cpp compatible GGUF format.
"""

import os
import logging
import json
import struct
from pathlib import Path
from typing import Dict, Optional, Any, List, Tuple
from enum import IntEnum
from dataclasses import dataclass
import tempfile

import torch
import torch.nn as nn
import numpy as np

logger = logging.getLogger(__name__)


class GGMLType(IntEnum):
    """GGML tensor types for quantization."""
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    Q8_K = 15


@dataclass
class GGUFTensor:
    """Tensor metadata for GGUF format."""
    name: str
    shape: Tuple[int, ...]
    dtype: GGMLType
    data: np.ndarray


class GGUFWriter:
    """
    Write models in GGUF format for llama.cpp.
    """

    GGUF_MAGIC = 0x46554747  # "GGUF" in little endian
    GGUF_VERSION = 3

    def __init__(self, output_path: str):
        self.output_path = Path(output_path)
        self.metadata: Dict[str, Any] = {}
        self.tensors: List[GGUFTensor] = []

    def add_metadata(self, key: str, value: Any):
        """Add metadata entry."""
        self.metadata[key] = value

    def add_tensor(self, name: str, tensor: torch.Tensor, quantize: bool = True):
        """Add tensor to GGUF file."""
        # Convert to numpy
        data = tensor.cpu().detach().numpy()

        # Determine dtype
        if quantize and len(data.shape) >= 2:
            # Quantize to Q4_0 for large tensors
            data, dtype = self._quantize_q4_0(data)
        else:
            dtype = GGMLType.F32 if data.dtype == np.float32 else GGMLType.F16

        self.tensors.append(GGUFTensor(
            name=name,
            shape=tensor.shape,
            dtype=dtype,
            data=data
        ))

    def _quantize_q4_0(self, data: np.ndarray) -> Tuple[np.ndarray, GGMLType]:
        """
        Quantize tensor to Q4_0 format (4-bit quantization).
        Group size: 32 elements
        """
        original_shape = data.shape

        # Reshape to groups of 32
        if len(original_shape) == 1:
            data = data.reshape(-1, 32)
        else:
            # Flatten and reshape
            flat = data.flatten()
            # Pad to multiple of 32
            pad_size = (32 - len(flat) % 32) % 32
            if pad_size > 0:
                flat = np.pad(flat, (0, pad_size), mode='constant')
            data = flat.reshape(-1, 32)

        # For each group: compute scale and quantize
        n_groups = data.shape[0]
        quantized = np.zeros((n_groups, 16 + 2), dtype=np.uint8)  # 16 bytes for 32 4-bit values + 2 for scale

        for i in range(n_groups):
            group = data[i].astype(np.float32)

            # Compute scale
            abs_max = np.max(np.abs(group))
            scale = abs_max / 7.0 if abs_max > 0 else 1.0

            # Quantize to [-8, 7]
            quantized_group = np.clip(np.round(group / scale), -8, 7).astype(np.int8)

            # Pack pairs of 4-bit values into bytes
            for j in range(16):
                low = (quantized_group[2*j] + 8) & 0x0F
                high = (quantized_group[2*j + 1] + 8) & 0x0F
                quantized[i, j] = low | (high << 4)

            # Store scale as half precision
            scale_half = np.float16(scale)
            quantized[i, 16:18] = np.frombuffer(scale_half.tobytes(), dtype=np.uint8)

        return quantized.flatten(), GGMLType.Q4_0

    def write(self):
        """Write GGUF file."""
        with open(self.output_path, 'wb') as f:
            # Write header
            self._write_header(f)

            # Write metadata
            self._write_metadata(f)

            # Write tensor info
            self._write_tensor_info(f)

            # Write tensor data
            self._write_tensor_data(f)

        logger.info(f"Wrote GGUF file: {self.output_path}")

    def _write_header(self, f):
        """Write GGUF header."""
        f.write(struct.pack('<I', self.GGUF_MAGIC))  # Magic
        f.write(struct.pack('<I', self.GGUF_VERSION))  # Version
        f.write(struct.pack('<Q', len(self.tensors)))  # Tensor count
        f.write(struct.pack('<Q', len(self.metadata)))  # Metadata count

    def _write_string(self, f, s: str):
        """Write length-prefixed string."""
        data = s.encode('utf-8')
        f.write(struct.pack('<Q', len(data)))
        f.write(data)

    def _write_metadata(self, f):
        """Write metadata entries."""
        for key, value in self.metadata.items():
            self._write_string(f, key)

            if isinstance(value, str):
                f.write(struct.pack('<I', 8))  # String type
                self._write_string(f, value)
            elif isinstance(value, int):
                f.write(struct.pack('<I', 4))  # Uint32 type
                f.write(struct.pack('<I', value))
            elif isinstance(value, float):
                f.write(struct.pack('<I', 6))  # Float32 type
                f.write(struct.pack('<f', value))
            elif isinstance(value, list):
                f.write(struct.pack('<I', 9))  # Array type
                # Write array type and length
                f.write(struct.pack('<I', 4))  # Uint32 elements
                f.write(struct.pack('<Q', len(value)))
                for v in value:
                    f.write(struct.pack('<I', v))

    def _write_tensor_info(self, f):
        """Write tensor metadata."""
        for tensor in self.tensors:
            self._write_string(f, tensor.name)

            # Number of dimensions
            f.write(struct.pack('<I', len(tensor.shape)))

            # Dimensions
            for dim in tensor.shape:
                f.write(struct.pack('<Q', dim))

            # Type
            f.write(struct.pack('<I', tensor.dtype))

            # Offset (will be filled in later)
            f.write(struct.pack('<Q', 0))

    def _write_tensor_data(self, f):
        """Write tensor data."""
        # Align to 32 bytes
        current_pos = f.tell()
        alignment = 32
        padding = (alignment - current_pos % alignment) % alignment
        f.write(b'\x00' * padding)

        for tensor in self.tensors:
            f.write(tensor.data.tobytes())


class EmberVLMToGGUF:
    """
    Convert EmberVLM model to GGUF format.
    """

    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        self.model = model
        self.config = config

    def convert(
        self,
        output_path: str,
        quantize: bool = True,
        quantization_bits: int = 4
    ) -> str:
        """
        Convert model to GGUF format.

        Args:
            output_path: Output file path
            quantize: Whether to quantize weights
            quantization_bits: Quantization bits (4 or 8)

        Returns:
            Path to output file
        """
        writer = GGUFWriter(output_path)

        # Add metadata
        self._add_metadata(writer)

        # Add tensors
        self._add_tensors(writer, quantize)

        # Write file
        writer.write()

        # Get file size
        size_mb = Path(output_path).stat().st_size / (1024 * 1024)
        logger.info(f"GGUF file size: {size_mb:.2f} MB")

        return output_path

    def _add_metadata(self, writer: GGUFWriter):
        """Add model metadata."""
        writer.add_metadata("general.architecture", "embervlm")
        writer.add_metadata("general.name", "EmberVLM")
        writer.add_metadata("general.quantization_version", 2)

        # Model config
        writer.add_metadata("embervlm.context_length",
                          self.config.get('max_position_embeddings', 512))
        writer.add_metadata("embervlm.embedding_length",
                          self.config.get('hidden_size', 768))
        writer.add_metadata("embervlm.block_count",
                          self.config.get('num_layers', 6))
        writer.add_metadata("embervlm.attention.head_count",
                          self.config.get('num_attention_heads', 12))
        writer.add_metadata("embervlm.vision.num_tokens",
                          self.config.get('num_vision_tokens', 8))
        writer.add_metadata("embervlm.vocab_size",
                          self.config.get('vocab_size', 50257))

    def _add_tensors(self, writer: GGUFWriter, quantize: bool):
        """Add model tensors."""
        state_dict = self.model.state_dict()

        for name, tensor in state_dict.items():
            # Skip non-weight tensors
            if 'running_mean' in name or 'running_var' in name or 'num_batches' in name:
                continue

            # Convert name to GGUF format
            gguf_name = self._convert_tensor_name(name)

            # Determine if this tensor should be quantized
            should_quantize = quantize and self._should_quantize(name, tensor)

            writer.add_tensor(gguf_name, tensor, quantize=should_quantize)

    def _convert_tensor_name(self, name: str) -> str:
        """Convert PyTorch tensor name to GGUF format."""
        # Replace dots with underscores for compatibility
        name = name.replace('.', '_')

        # Map common patterns
        mappings = {
            'embed_tokens_weight': 'token_embd.weight',
            'lm_head_weight': 'output.weight',
            'norm_weight': 'output_norm.weight',
        }

        for pattern, replacement in mappings.items():
            if pattern in name:
                return replacement

        return name

    def _should_quantize(self, name: str, tensor: torch.Tensor) -> bool:
        """Determine if tensor should be quantized."""
        # Don't quantize small tensors
        if tensor.numel() < 1024:
            return False

        # Don't quantize normalization weights
        if 'norm' in name.lower() or 'ln' in name.lower():
            return False

        # Don't quantize biases
        if 'bias' in name.lower():
            return False

        # Quantize everything else
        return True


def convert_to_gguf(
    model: nn.Module,
    output_path: str,
    config: Optional[Dict] = None,
    quantize: bool = True
) -> str:
    """
    Convert EmberVLM model to GGUF format.

    Args:
        model: PyTorch model
        output_path: Output file path
        config: Model configuration
        quantize: Whether to quantize

    Returns:
        Path to output GGUF file
    """
    config = config or {}
    converter = EmberVLMToGGUF(model, config)
    return converter.convert(output_path, quantize=quantize)


if __name__ == "__main__":
    # Test GGUF conversion
    print("Testing GGUF Conversion...")

    # Create simple test model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(1000, 64)
            self.linear = nn.Linear(64, 64)
            self.norm = nn.LayerNorm(64)

    model = SimpleModel()

    # Test conversion
    with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as f:
        output_path = f.name

    convert_to_gguf(
        model=model,
        output_path=output_path,
        config={'hidden_size': 64, 'num_layers': 1},
        quantize=True
    )

    # Check file size
    size = Path(output_path).stat().st_size
    print(f"Output size: {size} bytes")

    # Cleanup
    os.unlink(output_path)

    print("GGUF conversion tests complete!")

