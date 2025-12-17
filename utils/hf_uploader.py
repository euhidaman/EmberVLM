"""
EmberVLM HuggingFace Hub Uploader
Checkpoint management with automatic HuggingFace Hub uploads.
"""

import os
import logging
import json
import shutil
from pathlib import Path
from typing import Dict, Optional, Any, List
from datetime import datetime
import tempfile

logger = logging.getLogger(__name__)

# Try to import huggingface_hub
try:
    from huggingface_hub import HfApi, Repository, create_repo, upload_folder
    from huggingface_hub.utils import RepositoryNotFoundError
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    HfApi = None


class HuggingFaceUploader:
    """
    Upload checkpoints to HuggingFace Hub with overwrite policy.
    """

    def __init__(
        self,
        repo_id: str,
        token: Optional[str] = None,
        private: bool = False,
        overwrite: bool = True,
        local_cache_dir: str = "hf_cache"
    ):
        self.repo_id = repo_id
        self.token = token or os.environ.get("HF_TOKEN")
        self.private = private
        self.overwrite = overwrite
        self.local_cache_dir = Path(local_cache_dir)
        self.local_cache_dir.mkdir(parents=True, exist_ok=True)

        self.api = None
        self._initialized = False

        if HF_AVAILABLE:
            self.api = HfApi(token=self.token)
        else:
            logger.warning("huggingface_hub not available. Install with: pip install huggingface_hub")

    def init_repo(self) -> bool:
        """Initialize or create HuggingFace repository."""
        if not self.api:
            return False

        try:
            # Check if repo exists
            self.api.repo_info(repo_id=self.repo_id, repo_type="model")
            logger.info(f"Repository {self.repo_id} exists")
            self._initialized = True
            return True

        except RepositoryNotFoundError:
            # Create new repo
            try:
                create_repo(
                    repo_id=self.repo_id,
                    token=self.token,
                    private=self.private,
                    repo_type="model"
                )
                logger.info(f"Created repository: {self.repo_id}")
                self._initialized = True
                return True
            except Exception as e:
                logger.error(f"Could not create repository: {e}")
                return False

        except Exception as e:
            logger.error(f"Could not access repository: {e}")
            return False

    def upload_checkpoint(
        self,
        checkpoint_dir: str,
        step: int,
        metrics: Optional[Dict[str, float]] = None,
        model_info: Optional[Dict] = None,
        carbon_info: Optional[Dict] = None,
        commit_message: Optional[str] = None
    ) -> bool:
        """
        Upload checkpoint to HuggingFace Hub.

        Args:
            checkpoint_dir: Local directory with checkpoint files
            step: Training step number
            metrics: Training metrics to include in model card
            model_info: Model information for model card
            carbon_info: Carbon emissions information
            commit_message: Custom commit message

        Returns:
            True if upload successful
        """
        if not self._initialized:
            if not self.init_repo():
                return False

        checkpoint_path = Path(checkpoint_dir)
        if not checkpoint_path.exists():
            logger.error(f"Checkpoint directory not found: {checkpoint_dir}")
            return False

        # Create temporary directory for upload
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Copy checkpoint files
            for item in checkpoint_path.iterdir():
                if item.is_file():
                    shutil.copy2(item, tmp_path / item.name)
                elif item.is_dir():
                    shutil.copytree(item, tmp_path / item.name)

            # Generate model card
            model_card = self._generate_model_card(
                step=step,
                metrics=metrics or {},
                model_info=model_info or {},
                carbon_info=carbon_info or {}
            )

            # Save model card
            with open(tmp_path / "README.md", 'w') as f:
                f.write(model_card)

            # Save config.json for HuggingFace
            config_json = {
                'model_type': 'embervlm',
                'architectures': ['EmberVLM'],
                'step': step,
                **model_info
            }
            with open(tmp_path / "config.json", 'w') as f:
                json.dump(config_json, f, indent=2)

            # Upload to Hub
            commit_msg = commit_message or f"Checkpoint at step {step}"

            try:
                if self.overwrite:
                    # Delete existing files first (overwrite policy)
                    try:
                        self.api.delete_folder(
                            repo_id=self.repo_id,
                            path_in_repo="",
                            repo_type="model",
                            token=self.token
                        )
                    except Exception:
                        pass  # Folder might not exist

                # Upload new checkpoint
                upload_folder(
                    repo_id=self.repo_id,
                    folder_path=str(tmp_path),
                    commit_message=commit_msg,
                    token=self.token
                )

                logger.info(f"Uploaded checkpoint to {self.repo_id}")
                return True

            except Exception as e:
                logger.error(f"Upload failed: {e}")
                return False

    def _generate_model_card(
        self,
        step: int,
        metrics: Dict[str, float],
        model_info: Dict,
        carbon_info: Dict
    ) -> str:
        """Generate README model card."""

        card = f"""---
license: apache-2.0
tags:
- vision-language
- multimodal
- robot-fleet
- embervlm
- tiny-model
datasets:
- custom
metrics:
- accuracy
---

# EmberVLM - Checkpoint Step {step}

Ultra-efficient multimodal Vision-Language Model for robot fleet reasoning.

## Model Description

EmberVLM is a tiny (~35M parameters) multimodal model designed for:
- **Robot fleet selection** - Choose optimal robots for tasks
- **Incident response** - Plan responses to emergency scenarios
- **Visual understanding** - Analyze images for robotic applications
- **Action planning** - Generate step-by-step action sequences

### Architecture

| Component | Parameters | Status |
|-----------|------------|--------|
| Vision Encoder (RepViT-XXS) | ~5M | Frozen |
| Language Model (TinyLLM) | ~30M | Partial Fine-tuning |
| Fusion Module | ~500K | Trainable |
| **Total** | **~35M** | - |

## Training Information

- **Current Step**: {step:,}
- **Training Loss**: {metrics.get('train_loss', 'N/A')}
- **Validation Loss**: {metrics.get('val_loss', 'N/A')}

### Performance Metrics

| Metric | Value |
|--------|-------|
| Robot Selection Accuracy | {metrics.get('robot_accuracy', 'N/A')} |
| VQA Accuracy | {metrics.get('vqa_accuracy', 'N/A')} |
| Captioning (CIDEr) | {metrics.get('cider', 'N/A')} |

## Environmental Impact

- **CO₂ Emitted**: {carbon_info.get('emissions_kg', 0):.4f} kg
- **Energy Used**: {carbon_info.get('energy_kwh', 0):.4f} kWh
- **Training Duration**: {carbon_info.get('duration_hours', 0):.2f} hours

This is equivalent to:
- 🚗 {carbon_info.get('emissions_kg', 0) / 0.12:.1f} km driven by car
- 📱 {carbon_info.get('emissions_kg', 0) / 0.0085:.0f} smartphone charges

## Usage

```python
from embervlm import EmberVLM

model = EmberVLM.from_pretrained("{self.repo_id}")
model.eval()

# Robot selection
output = model.generate(
    pixel_values=image,
    input_ids=tokenizer.encode("Select robot for: building inspection"),
    max_new_tokens=100
)
```

## Deployment

Optimized for Raspberry Pi Zero deployment:
- Model size: <100MB (quantized)
- Memory usage: <400MB RAM
- Inference: <500ms per response

## Citation

```bibtex
@misc{{embervlm2024,
  title={{EmberVLM: Tiny Multimodal Robot Fleet Reasoning}},
  year={{2024}},
  note={{Checkpoint at step {step}}}
}}
```

## License

Apache 2.0
"""
        return card

    def upload_gguf(
        self,
        gguf_path: str,
        step: int,
        commit_message: Optional[str] = None
    ) -> bool:
        """Upload quantized GGUF file."""
        if not self._initialized:
            if not self.init_repo():
                return False

        gguf_file = Path(gguf_path)
        if not gguf_file.exists():
            logger.error(f"GGUF file not found: {gguf_path}")
            return False

        try:
            self.api.upload_file(
                path_or_fileobj=str(gguf_file),
                path_in_repo=f"embervlm-step-{step}.gguf",
                repo_id=self.repo_id,
                token=self.token,
                commit_message=commit_message or f"GGUF at step {step}"
            )
            logger.info(f"Uploaded GGUF to {self.repo_id}")
            return True

        except Exception as e:
            logger.error(f"GGUF upload failed: {e}")
            return False

    def get_repo_url(self) -> str:
        """Get HuggingFace repo URL."""
        return f"https://huggingface.co/{self.repo_id}"


def create_hf_uploader(
    repo_id: str,
    token: Optional[str] = None,
    **kwargs
) -> HuggingFaceUploader:
    """Factory function for HuggingFace uploader."""
    return HuggingFaceUploader(
        repo_id=repo_id,
        token=token,
        **kwargs
    )


if __name__ == "__main__":
    # Test HuggingFace uploader
    print("Testing HuggingFace Uploader...")

    uploader = HuggingFaceUploader(
        repo_id="test/embervlm-test",
        token=None,  # Would need actual token
        overwrite=True
    )

    # Test model card generation
    model_card = uploader._generate_model_card(
        step=1000,
        metrics={'train_loss': 0.5, 'robot_accuracy': 0.85},
        model_info={'total_params': 35000000},
        carbon_info={'emissions_kg': 0.5, 'energy_kwh': 2.0, 'duration_hours': 1.5}
    )

    print("Generated model card:")
    print(model_card[:500] + "...")

    print("HuggingFace uploader tests complete!")

