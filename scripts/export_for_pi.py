#!/usr/bin/env python3
"""
EmberVLM Export Script
Export trained model for Raspberry Pi deployment.

Usage:
    python export_for_pi.py --checkpoint outputs/checkpoint-1000 --output deployment/
"""

import argparse
import logging
import json
import sys
from pathlib import Path

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def export_model(
    checkpoint_path: str,
    output_dir: str,
    quantize: bool = True,
    target_size_mb: float = 100
):
    """
    Export EmberVLM for Raspberry Pi deployment.

    Args:
        checkpoint_path: Path to trained checkpoint
        output_dir: Output directory for exported files
        quantize: Whether to apply quantization
        target_size_mb: Target model size in MB
    """
    logger.info("=" * 50)
    logger.info("EmberVLM Export for Raspberry Pi")
    logger.info("=" * 50)

    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    logger.info(f"Loading model from {checkpoint_path}...")

    try:
        from models import EmberVLM
        model = EmberVLM.from_pretrained(str(checkpoint_path))
    except Exception as e:
        logger.error(f"Could not load model: {e}")
        logger.info("Creating default model for export demo...")

        from models import EmberVLM, EmberVLMConfig
        config = EmberVLMConfig(vision_pretrained=False)
        model = EmberVLM(config)

    # Get original size
    original_size = sum(p.numel() * p.element_size() for p in model.parameters())
    logger.info(f"Original model size: {original_size / 1024 / 1024:.2f} MB")

    # Apply optimizations
    if quantize:
        logger.info("Applying quantization...")
        from quantization import create_pi_optimizer

        optimizer = create_pi_optimizer(
            target_size_mb=target_size_mb,
            target_memory_mb=400,
            target_latency_ms=500
        )

        results = optimizer.optimize_model(
            model=model,
            output_dir=str(output_dir),
            apply_pruning=True,
            apply_quantization=True
        )

        logger.info(f"Optimized model size: {results.get('quantized_size_mb', 0):.2f} MB")
    else:
        # Save without optimization
        torch.save(model.state_dict(), output_dir / "embervlm_pi.pt")

    # Convert to GGUF
    logger.info("Converting to GGUF format...")
    from quantization import convert_to_gguf

    gguf_path = output_dir / "embervlm.gguf"
    convert_to_gguf(
        model=model,
        output_path=str(gguf_path),
        config=model.config.to_dict().get('model', {}),
        quantize=quantize
    )

    gguf_size = gguf_path.stat().st_size / 1024 / 1024
    logger.info(f"GGUF file size: {gguf_size:.2f} MB")

    # Generate deployment config
    config = {
        "model_name": "EmberVLM-Pi",
        "version": "0.1.0",
        "quantized": quantize,
        "original_size_mb": original_size / 1024 / 1024,
        "gguf_size_mb": gguf_size,
        "inference": {
            "max_tokens": 100,
            "context_length": 512,
            "temperature": 0.7
        },
        "hardware": {
            "target": "Raspberry Pi Zero",
            "max_memory_mb": 400
        }
    }

    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    # Copy inference script
    import shutil
    inference_src = Path(__file__).parent.parent / "deployment" / "pi_inference.py"
    if inference_src.exists():
        shutil.copy(inference_src, output_dir / "pi_inference.py")

    logger.info("=" * 50)
    logger.info("Export complete!")
    logger.info("=" * 50)
    logger.info(f"\nOutput files:")
    logger.info(f"  Model: {output_dir / 'embervlm.gguf'}")
    logger.info(f"  Config: {output_dir / 'config.json'}")
    logger.info(f"  Inference: {output_dir / 'pi_inference.py'}")
    logger.info(f"\nTo deploy on Raspberry Pi:")
    logger.info(f"  1. Copy the '{output_dir}' folder to your Pi")
    logger.info(f"  2. Install dependencies: pip install numpy pillow")
    logger.info(f"  3. Run: python pi_inference.py --model embervlm.gguf --interactive")


def main():
    parser = argparse.ArgumentParser(description='Export EmberVLM for Raspberry Pi')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='outputs/embervlm/final',
        help='Path to trained checkpoint'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='deployment_package',
        help='Output directory'
    )
    parser.add_argument(
        '--no-quantize',
        action='store_true',
        help='Skip quantization'
    )
    parser.add_argument(
        '--target-size',
        type=float,
        default=100,
        help='Target model size in MB'
    )

    args = parser.parse_args()

    export_model(
        checkpoint_path=args.checkpoint,
        output_dir=args.output,
        quantize=not args.no_quantize,
        target_size_mb=args.target_size
    )


if __name__ == "__main__":
    main()

