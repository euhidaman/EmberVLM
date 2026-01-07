"""
Data Loaders for EmberVLM Training

Provides data loaders for different training stages.
Memory-safe implementation with distributed training support.
"""

import os
import gc
import json
import random
import warnings
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Callable

import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.distributed as dist
from PIL import Image, ImageFile
import torchvision.transforms as transforms

# Enable loading of truncated images - prevents crashes from corrupted files
ImageFile.LOAD_TRUNCATED_IMAGES = True
# Limit PIL's decompression bomb prevention (for very large images)
Image.MAX_IMAGE_PIXELS = 178956970  # ~13K x 13K

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    ARROW_AVAILABLE = True
except ImportError:
    ARROW_AVAILABLE = False


# ============================================================================
# MEMORY SAFETY CONFIGURATION
# ============================================================================
# Hard limits to prevent OOM crashes on shared servers
# These values are tuned for 2x A100 (80GB) with ~500GB system RAM
# Adjust lower if you experience memory issues

# Maximum samples per individual dataset source
MAX_SAMPLES_PER_DATASET = 150000  # 150k per dataset source

# Maximum total samples for Stage 1 alignment (Vision-Language)
# For a good VLM, we need diverse data. 1M samples is reasonable for 37M param model
MAX_STAGE1_TOTAL_SAMPLES = 1000000  # 1M total for stage 1

# Maximum total samples for Stage 2 instruction tuning
MAX_STAGE2_TOTAL_SAMPLES = 200000  # 200k for stage 2 (LLaVA has ~158k)

# Maximum CC3M samples to load
# CC3M is valuable for vision-language alignment, but 3M is too much
# 150k gives good coverage without memory explosion
MAX_CC3M_SAMPLES = 150000  # 150k from CC3M (it's ~3M total)

# Maximum GQA samples per JSON file
# GQA teaches scene reasoning - important for robot selection
MAX_GQA_SAMPLES_PER_FILE = 100000  # 100k per GQA file

# Maximum GQA files to process
# Use balanced files + train for best coverage
MAX_GQA_FILES = 5  # Process 5 GQA files (balanced train/val/test + 2 more)

# Safe number of DataLoader workers (prevents CPU oversubscription)
SAFE_NUM_WORKERS = 2  # 2 workers per GPU is usually safe

# Whether to use lazy/streaming dataset (memory efficient but slower)
USE_LAZY_LOADING = True


def _get_rank() -> int:
    """Get current distributed rank (0 if not distributed)."""
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def _get_world_size() -> int:
    """Get world size (1 if not distributed)."""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def _is_main_process() -> bool:
    """Check if this is rank 0 (main process)."""
    return _get_rank() == 0


def _barrier():
    """Synchronize all distributed processes."""
    if dist.is_initialized():
        dist.barrier()


def _broadcast_object(obj, src=0):
    """Broadcast a Python object from src rank to all ranks."""
    if not dist.is_initialized() or _get_world_size() == 1:
        return obj

    object_list = [obj] if _get_rank() == src else [None]
    dist.broadcast_object_list(object_list, src=src)
    return object_list[0]


def _get_memory_usage_mb() -> float:
    """Get current process memory usage in MB."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except:
        return 0.0


def _log_memory_warning(logger, context: str):
    """Log a warning if memory usage is high."""
    mem_mb = _get_memory_usage_mb()
    if mem_mb > 50000:  # > 50GB
        logger.warning(f"⚠️ HIGH MEMORY USAGE: {mem_mb:.0f} MB during {context}")
    elif mem_mb > 30000:  # > 30GB
        logger.info(f"Memory usage: {mem_mb:.0f} MB during {context}")


def _safe_load_image(image_input, transform, image_size: int, logger=None) -> Optional[torch.Tensor]:
    """
    Safely load and transform an image with robust error handling.

    Returns None on failure instead of crashing.
    """
    try:
        # Suppress PIL warnings during loading
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)

            if isinstance(image_input, Image.Image):
                # Already a PIL Image
                if image_input.mode != 'RGB':
                    image = image_input.convert('RGB')
                else:
                    image = image_input.copy()  # Copy to avoid modifying original
            elif isinstance(image_input, bytes):
                # Raw bytes
                import io
                image = Image.open(io.BytesIO(image_input)).convert('RGB')
            elif isinstance(image_input, (str, Path)):
                # File path
                image = Image.open(str(image_input)).convert('RGB')
            else:
                return None

            # Verify image is valid by loading data
            image.load()

            # Apply transform
            return transform(image)

    except Exception as e:
        if logger:
            logger.debug(f"Failed to load image: {e}")
        return None


class BaseVLMDataset(Dataset):
    """Base dataset class for vision-language data."""

    def __init__(
        self,
        data_dir: str,
        tokenizer: Any,
        split: str = 'train',
        max_length: int = 512,
        image_size: int = 224,
        transform: Optional[Callable] = None,
    ):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.split = split
        self.max_length = max_length
        self.image_size = image_size

        # Image transform
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
        else:
            self.transform = transform

        # Load data
        self.samples = self._load_data()

    def _load_data(self) -> List[Dict[str, Any]]:
        """Load data samples. Override in subclasses."""
        raise NotImplementedError

    def _load_image(self, image_input) -> torch.Tensor:
        """Load and transform image from path or PIL Image safely."""
        result = _safe_load_image(image_input, self.transform, self.image_size)
        if result is not None:
            return result
        # Return black image on error
        return torch.zeros(3, self.image_size, self.image_size)

    def _tokenize(
        self,
        text: str,
        add_labels: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Tokenize text."""
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        result = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
        }

        if add_labels:
            labels = encoding['input_ids'].squeeze(0).clone()
            labels[labels == self.tokenizer.pad_token_id] = -100
            result['labels'] = labels

        return result

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        raise NotImplementedError


class AlignmentDataset(BaseVLMDataset):
    """Dataset for Stage 1 visual-language alignment - supports multiple dataset formats."""

    def __init__(self, *args, **kwargs):
        # Store CC3M dataset reference for lazy loading
        self._cc3m_dataset = None
        self._cc3m_indices = None
        super().__init__(*args, **kwargs)

    def _load_cc3m_hf(self, logger) -> List[Dict[str, Any]]:
        """
        Load CC3M data using HuggingFace datasets library.

        MEMORY SAFE: Only loads metadata, not actual images.
        Images are loaded lazily at __getitem__ time.
        """
        samples = []

        # Only rank 0 should log detailed info
        is_main = _is_main_process()

        # CC3M is stored in HuggingFace WebDataset format
        cc3m_dir = self.data_dir / 'cc3m'

        if not cc3m_dir.exists():
            if is_main:
                logger.debug("CC3M directory not found")
            return samples

        try:
            # Try to import datasets library
            try:
                from datasets import load_dataset
            except ImportError:
                if is_main:
                    logger.warning("HuggingFace datasets library not available - skipping CC3M")
                return samples

            # Check if dataset is already cached locally
            cached_dataset_path = cc3m_dir / 'dataset' / 'pixparse___cc3m-wds' / 'default' / '0.0.0'

            dataset = None
            if cached_dataset_path.exists():
                hash_dirs = [d for d in cached_dataset_path.iterdir() if d.is_dir() and len(d.name) == 40]
                if hash_dirs:
                    local_dataset_path = hash_dirs[0]
                    if is_main:
                        logger.info(f"Loading CC3M dataset from cached path: {local_dataset_path}")

                    try:
                        dataset = load_dataset(
                            "arrow",
                            data_files={
                                "train": str(local_dataset_path / "cc3m-wds-train-*.arrow")
                            },
                            split="train"
                        )
                        if is_main:
                            logger.info(f"✓ Successfully loaded {len(dataset):,} samples from cached CC3M dataset")
                    except Exception as e:
                        if is_main:
                            logger.warning(f"Failed to load from cached path: {e}")
                        dataset = None

            if dataset is None:
                if is_main:
                    logger.info("Loading CC3M dataset from HuggingFace...")
                dataset = load_dataset(
                    "pixparse/cc3m-wds",
                    cache_dir=str(cc3m_dir / 'dataset'),
                    split="train"
                )

            if is_main:
                logger.info(f"Found {len(dataset):,} samples in CC3M dataset")

            # MEMORY SAFETY: Use much smaller sample limit
            samples_limit = min(MAX_CC3M_SAMPLES, len(dataset))

            if is_main:
                logger.info(f"⚠️ MEMORY SAFE: Limiting CC3M to {samples_limit:,} samples (from {len(dataset):,})")

            # Sample indices uniformly
            import numpy as np
            if len(dataset) > samples_limit:
                np.random.seed(42)  # Reproducible sampling
                indices = np.random.choice(len(dataset), samples_limit, replace=False)
                indices = sorted(indices)  # Sort for sequential access
            else:
                indices = list(range(len(dataset)))

            # Store dataset reference for lazy loading
            self._cc3m_dataset = dataset
            self._cc3m_indices = indices

            # Create lightweight sample references (NO image data stored!)
            for i, idx in enumerate(indices):
                samples.append({
                    'type': 'cc3m',
                    'cc3m_idx': idx,
                    'caption': None,  # Will be loaded lazily
                })

                # Progress logging every 10k
                if is_main and (i + 1) % 10000 == 0:
                    logger.info(f"  Indexed {i + 1:,} CC3M samples...")

            if is_main:
                logger.info(f"✓ Indexed {len(samples):,} CC3M samples (lazy loading enabled)")
                _log_memory_warning(logger, "CC3M indexing")

        except Exception as e:
            if is_main:
                logger.warning(f"Failed to load CC3M dataset: {e}")

        return samples

    def _get_cc3m_sample(self, cc3m_idx: int) -> Dict[str, Any]:
        """Lazily load a CC3M sample by index."""
        if self._cc3m_dataset is None:
            return None

        try:
            item = self._cc3m_dataset[cc3m_idx]
            image = item.get('jpg') or item.get('image')
            caption = item.get('txt') or item.get('caption') or item.get('text', '')

            if image is None or not caption:
                return None

            return {
                'image': image,
                'caption': str(caption).strip(),
            }
        except Exception:
            return None

    def _load_refcoco_arrow(self, logger) -> List[Dict[str, Any]]:
        """Load RefCOCO/RefCOCO+/RefCOCOg data from Arrow files."""
        samples = []

        if not ARROW_AVAILABLE or not PANDAS_AVAILABLE:
            return samples

        # Check for RefCOCO variants - they're in nested directories
        refcoco_dirs = [
            ('refcoco', self.data_dir / 'refcoco'),
            ('refcoco_plus', self.data_dir / 'refcoco_plus'),
            ('refcocog', self.data_dir / 'refcocog'),
        ]

        # Find COCO images directory
        coco_img_dirs = []
        coco_dir = self.data_dir / 'coco'
        if coco_dir.exists():
            for pattern in ['train2014', 'val2014', 'train2017', 'val2017']:
                found = list(coco_dir.glob(pattern))
                coco_img_dirs.extend(found)

        if not coco_img_dirs:
            logger.debug("COCO image directories not found for RefCOCO")
            return samples

        for dataset_name, refcoco_dir in refcoco_dirs:
            if not refcoco_dir.exists():
                continue

            # Arrow files are directly in the refcoco directory (no nested subdirs in your structure)
            arrow_files = sorted(refcoco_dir.glob('*.arrow'))

            if not arrow_files:
                logger.debug(f"No Arrow files found in {dataset_name}")
                continue

            logger.info(f"Loading {dataset_name} from {len(arrow_files)} Arrow files...")

            for arrow_file in arrow_files:
                # Skip dataset_info.json or lock files
                if 'dataset_info' in arrow_file.name or '.lock' in arrow_file.name:
                    continue

                try:
                    table = pa.ipc.open_file(str(arrow_file)).read_all()
                    df = table.to_pandas()

                    for idx in range(len(df)):
                        try:
                            row = df.iloc[idx]

                            # Extract referring expression
                            caption = None
                            for col in ['sent', 'caption', 'sentence', 'text']:
                                if col in df.columns and pd.notna(row[col]):
                                    caption = str(row[col]).strip()
                                    break

                            # Extract image ID
                            image_id = None
                            for col in ['image_id', 'imageId', 'image']:
                                if col in df.columns and pd.notna(row[col]):
                                    image_id = row[col]
                                    break

                            if caption and image_id:
                                # Try to find image - RefCOCO uses COCO 2014 images
                                image_filename = f"COCO_train2014_{int(image_id):012d}.jpg"
                                image_filename_val = f"COCO_val2014_{int(image_id):012d}.jpg"

                                for img_dir in coco_img_dirs:
                                    for fname in [image_filename, image_filename_val]:
                                        candidate = img_dir / fname
                                        if candidate.exists():
                                            samples.append({
                                                'image': str(candidate),
                                                'caption': caption,
                                            })
                                            break
                                    else:
                                        continue
                                    break

                        except Exception as e:
                            logger.debug(f"Failed to process row: {e}")
                            continue

                except Exception as e:
                    logger.debug(f"Failed to load {arrow_file.name}: {e}")
                    continue

            if samples:
                logger.info(f"✓ Loaded {len(samples):,} samples from {dataset_name}")

        return samples

    def _load_data(self) -> List[Dict[str, Any]]:
        """
        Load image-text pairs from multiple dataset formats.

        MEMORY SAFE: Implements hard limits on samples and efficient loading.
        Only rank 0 does heavy I/O, then broadcasts metadata to other ranks.
        """
        import logging
        logger = logging.getLogger(__name__)

        is_main = _is_main_process()
        samples = []

        # Track samples by source for limiting
        samples_by_source = {}
        gqa_files_processed = 0

        if is_main:
            logger.info("="*60)
            logger.info("MEMORY-SAFE DATASET LOADING")
            logger.info(f"  Max samples per dataset: {MAX_SAMPLES_PER_DATASET:,}")
            logger.info(f"  Max total Stage 1 samples: {MAX_STAGE1_TOTAL_SAMPLES:,}")
            logger.info(f"  Max CC3M samples: {MAX_CC3M_SAMPLES:,}")
            logger.info(f"  Max GQA files: {MAX_GQA_FILES}")
            logger.info("="*60)

        # Load CC3M from HuggingFace format (memory safe - lazy loading)
        cc3m_samples = self._load_cc3m_hf(logger)
        samples.extend(cc3m_samples)
        samples_by_source['cc3m'] = len(cc3m_samples)

        # Load RefCOCO variants from Arrow files (limited)
        refcoco_samples = self._load_refcoco_arrow(logger)
        # Limit RefCOCO samples
        if len(refcoco_samples) > MAX_SAMPLES_PER_DATASET:
            refcoco_samples = refcoco_samples[:MAX_SAMPLES_PER_DATASET]
            if is_main:
                logger.info(f"⚠️ Limited RefCOCO to {MAX_SAMPLES_PER_DATASET:,} samples")
        samples.extend(refcoco_samples)
        samples_by_source['refcoco'] = len(refcoco_samples)

        # Recursively search for JSON files in subdirectories
        json_files = list(self.data_dir.rglob('*.json'))

        if not json_files and is_main:
            logger.warning(f"No JSON files found in {self.data_dir} or its subdirectories")

        if is_main:
            logger.info(f"Found {len(json_files)} JSON files to process")

        # Skip metadata files and non-VL datasets
        skip_patterns = [
            'download_summary', 'dataset_info', 'instances_', 'person_keypoints_',
            '__MACOSX', '.lock', '_builder.lock', '_incomplete', 'dataset.json',
            'readme.txt', 'LICENCE.txt', 'loadDataset.py', '.py', '.csv', '.download_attempted',
        ]

        # Also skip annotation-only files that don't contain usable data
        skip_exact = [
            'v2_mscoco_train2014_annotations.json',
            'v2_mscoco_val2014_annotations.json',
            'mscoco_train2014_annotations.json',
            'mscoco_val2014_annotations.json',
        ]

        # Prioritize certain files (balanced/smaller datasets first)
        priority_patterns = ['balanced', 'val', 'train']

        def get_priority(f):
            name = f.name.lower()
            for i, p in enumerate(priority_patterns):
                if p in name:
                    return i
            return len(priority_patterns)

        json_files = sorted(json_files, key=get_priority)

        for json_file in json_files:
            # Check if we've hit the total sample limit
            if len(samples) >= MAX_STAGE1_TOTAL_SAMPLES:
                if is_main:
                    logger.info(f"⚠️ Reached max total samples ({MAX_STAGE1_TOTAL_SAMPLES:,}), stopping dataset loading")
                break

            # Skip metadata files
            file_name_lower = json_file.name.lower()
            if any(pattern in file_name_lower for pattern in skip_patterns):
                continue

            # Skip exact matches of problematic files
            if json_file.name in skip_exact:
                if is_main:
                    logger.debug(f"Skipping annotation-only file: {json_file.name}")
                continue

            # Skip files in __MACOSX directories
            if '__MACOSX' in str(json_file):
                continue

            # Check if this is a GQA file and if we've processed enough
            is_gqa = 'gqa' in str(json_file).lower()
            if is_gqa:
                if gqa_files_processed >= MAX_GQA_FILES:
                    if is_main:
                        logger.debug(f"Skipping GQA file (already processed {MAX_GQA_FILES} files): {json_file.name}")
                    continue

            if is_main:
                logger.info(f"Processing file: {json_file}")

            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                if is_main:
                    logger.warning(f"Failed to load {json_file}: {e}")
                continue

            # Get the parent directory for resolving image paths
            json_parent = json_file.parent
            before_count = len(samples)
            file_sample_count = 0
            max_samples_this_file = MAX_SAMPLES_PER_DATASET

            if isinstance(data, list):
                # Handle list format (LLaVA, CC3M, etc.)
                for item in data:
                    # Check per-file limit
                    if file_sample_count >= max_samples_this_file:
                        break

                    text = None
                    image_path = None

                    # Extract text (caption, question+answer, instruction, conversations, etc.)
                    if 'conversations' in item:
                        # LLaVA format: conversations = [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]
                        convos = item['conversations']
                        if len(convos) >= 2:
                            human_text = convos[0].get('value', '').replace('<image>\n', '').replace('<image>', '').strip()
                            gpt_text = convos[1].get('value', '').strip()
                            if human_text and gpt_text:
                                text = f"Question: {human_text} Answer: {gpt_text}"
                    elif 'caption' in item:
                        text = item['caption']
                    elif 'text' in item:
                        text = item['text']
                    elif 'question' in item and 'answer' in item:
                        text = f"Question: {item['question']} Answer: {item['answer']}"
                    elif 'question' in item:
                        text = item['question']

                    # Extract image path
                    if 'image' in item:
                        image_path = item['image']
                        # LLaVA often uses COCO image IDs like "coco/train2017/000000123456.jpg"
                        if not os.path.isabs(image_path):
                            # Try multiple potential base dirs
                            potential_paths = [
                                json_parent / image_path,
                                self.data_dir / image_path,
                                self.data_dir / 'coco' / image_path.split('/')[-1] if '/' in image_path else None,
                            ]
                            for p in potential_paths:
                                if p and p.exists():
                                    image_path = str(p)
                                    break
                            else:
                                image_path = str(json_parent / image_path)  # Fallback
                    elif 'image_id' in item:
                        image_path = f"{item['image_id']}.jpg"
                        if not os.path.isabs(image_path):
                            image_path = str(json_parent / image_path)
                    elif 'file_name' in item:
                        image_path = item['file_name']
                        if not os.path.isabs(image_path):
                            image_path = str(json_parent / image_path)

                    if text and image_path:
                        samples.append({
                            'image': image_path,
                            'caption': text,
                        })
                        file_sample_count += 1

            elif isinstance(data, dict):
                # ===== COCO Captions Format =====
                if 'annotations' in data and 'images' in data and any('caption' in ann for ann in data['annotations'][:5]):
                    logger.info(f"Detected COCO Captions format: {len(data['images'])} images, {len(data['annotations'])} annotations")

                    image_map = {img['id']: img['file_name'] for img in data['images']}

                    # Find image directories
                    image_dirs = []
                    for search_dir in [json_parent, json_parent.parent]:
                        for pattern in ['train2017', 'val2017', 'train2014', 'val2014', 'images']:
                            found_dirs = list(search_dir.glob(pattern))
                            image_dirs.extend(found_dirs)

                    if not image_dirs:
                        if is_main:
                            logger.warning(f"No image directories found for {json_file}")
                    else:
                        if is_main:
                            logger.info(f"Found image directories: {[str(d) for d in image_dirs]}")

                        for ann in data['annotations']:
                            # Check per-file limit
                            if file_sample_count >= max_samples_this_file:
                                break

                            image_id = ann.get('image_id')
                            caption = ann.get('caption', '')

                            if caption and image_id in image_map:
                                image_filename = image_map[image_id]
                                image_path = None
                                for img_dir in image_dirs:
                                    candidate = img_dir / image_filename
                                    if candidate.exists():
                                        image_path = str(candidate)
                                        break

                                if image_path:
                                    samples.append({
                                        'image': image_path,
                                        'caption': caption,
                                    })
                                    file_sample_count += 1

                # ===== VQA Format (questions + annotations) =====
                elif 'questions' in data or ('annotations' in data and any('question' in ann or 'answer' in ann for ann in data.get('annotations', [])[:5])):
                    if is_main:
                        logger.info(f"Detected VQA format")

                    # VQA v2 format has separate questions and annotations
                    questions = data.get('questions', data.get('annotations', []))

                    # Find image directory
                    image_dirs = []
                    for search_dir in [json_parent, json_parent.parent]:
                        for pattern in ['train2014', 'val2014', 'test2015', 'train2017', 'val2017', 'images']:
                            found_dirs = list(search_dir.glob(pattern))
                            image_dirs.extend(found_dirs)

                    if not image_dirs:
                        # Try COCO directory structure
                        coco_dir = self.data_dir / 'coco'
                        if coco_dir.exists():
                            for pattern in ['train2014', 'val2014', 'train2017', 'val2017']:
                                found_dirs = list(coco_dir.glob(pattern))
                                image_dirs.extend(found_dirs)

                    for item in questions:
                        # Check per-file limit
                        if file_sample_count >= max_samples_this_file:
                            break

                        question = item.get('question', '')
                        answer = item.get('answer', item.get('multiple_choice_answer', ''))
                        image_id = item.get('image_id', '')

                        if question and image_id:
                            # Construct text as Q&A pair or just question
                            if answer:
                                text = f"Question: {question} Answer: {answer}"
                            else:
                                text = f"Question: {question}"

                            # Try multiple COCO image naming conventions
                            possible_filenames = [
                                f"COCO_train2014_{image_id:012d}.jpg",
                                f"COCO_val2014_{image_id:012d}.jpg",
                                f"COCO_train2017_{image_id:012d}.jpg",
                                f"COCO_val2017_{image_id:012d}.jpg",
                                f"{image_id:012d}.jpg",  # Sometimes just the ID
                            ]
                            if 'image_name' in item:
                                possible_filenames.insert(0, item['image_name'])

                            found_image = False
                            if image_dirs:
                                for img_dir in image_dirs:
                                    if found_image:
                                        break
                                    for image_filename in possible_filenames:
                                        candidate = img_dir / image_filename
                                        if candidate.exists():
                                            samples.append({
                                                'image': str(candidate),
                                                'caption': text,
                                            })
                                            file_sample_count += 1
                                            found_image = True
                                            break

                # ===== GQA Format =====
                elif isinstance(data, dict) and len(data) > 0:
                    # Check if it's GQA format: dict of dicts with question/answer structure
                    first_values = list(data.values())[:5]
                    is_gqa_format = all(isinstance(v, dict) and any(k in v for k in ['question', 'answer', 'imageId']) for v in first_values if isinstance(v, dict))

                    if is_gqa_format:
                        # GQA is dict of dicts: {question_id: {question, answer, imageId, ...}}
                        if is_main:
                            logger.info(f"Detected GQA format: {len(data)} questions")

                        # Find GQA images directory
                        image_dirs = []
                        for search_dir in [json_parent, json_parent.parent]:
                            for pattern in ['images', 'allImages']:
                                found_dirs = list(search_dir.glob(pattern))
                                image_dirs.extend(found_dirs)

                        # MEMORY SAFETY: Use much smaller limit per GQA file
                        gqa_limit = min(MAX_GQA_SAMPLES_PER_FILE, max_samples_this_file - file_sample_count, len(data))

                        if is_main:
                            logger.info(f"⚠️ GQA limit for this file: {gqa_limit:,} (from {len(data):,})")

                        for qid, item in list(data.items())[:gqa_limit]:
                            if file_sample_count >= max_samples_this_file:
                                break

                            if not isinstance(item, dict):
                                continue

                            question = item.get('question', '')
                            answer = item.get('answer', '')
                            image_id = item.get('imageId', '')

                            if question and image_id:
                                text = f"Question: {question} Answer: {answer}" if answer else f"Question: {question}"
                                image_filename = f"{image_id}.jpg"

                                if image_dirs:
                                    for img_dir in image_dirs:
                                        candidate = img_dir / image_filename
                                        if candidate.exists():
                                            samples.append({
                                                'image': str(candidate),
                                                'caption': text,
                                            })
                                            file_sample_count += 1
                                            break

                        # Mark that we've processed a GQA file
                        if is_gqa:
                            gqa_files_processed += 1
                            if is_main:
                                logger.info(f"GQA files processed: {gqa_files_processed}/{MAX_GQA_FILES}")

                # ===== RefCOCO Format =====
                elif 'refs' in data or any('ref_id' in str(k) for k in list(data.keys())[:5]):
                    if is_main:
                        logger.info(f"Detected RefCOCO format")

                    refs = data.get('refs', data.get('annotations', []))

                    # Find image directory (RefCOCO uses COCO images)
                    image_dirs = []
                    coco_dir = self.data_dir / 'coco'
                    if coco_dir.exists():
                        for pattern in ['train2014', 'train2017', 'val2014', 'val2017']:
                            found_dirs = list(coco_dir.glob(pattern))
                            image_dirs.extend(found_dirs)

                    for ref in refs:
                        if file_sample_count >= max_samples_this_file:
                            break

                        # RefCOCO has referring expressions
                        sentences = ref.get('sentences', [])
                        image_id = ref.get('image_id', '')

                        for sent in sentences:
                            if file_sample_count >= max_samples_this_file:
                                break
                            text = sent.get('sent', sent.get('raw', ''))
                            if text and image_id:
                                image_filename = f"COCO_train2014_{image_id:012d}.jpg"

                                if image_dirs:
                                    for img_dir in image_dirs:
                                        candidate = img_dir / image_filename
                                        if candidate.exists():
                                            samples.append({
                                                'image': str(candidate),
                                                'caption': text,
                                            })
                                            file_sample_count += 1
                                            break

                # ===== OCR-VQA Format =====
                # Check if it's actually OCR-VQA data (has questions, not just metadata)
                elif 'data' in data and isinstance(data['data'], list) and len(data['data']) > 0:
                    # Verify first item has OCR-VQA structure
                    first_item = data['data'][0] if isinstance(data['data'], list) else {}
                    if 'question' in first_item or 'imageURL' in first_item:
                        if is_main:
                            logger.info(f"Detected OCR-VQA format")

                        items = data['data']
                        ocr_limit = min(max_samples_this_file, len(items))

                        for item in items[:ocr_limit]:
                            if file_sample_count >= max_samples_this_file:
                                break

                            if not isinstance(item, dict):
                                continue

                            question = item.get('question', item.get('text', ''))
                            answer = item.get('answer', '')
                            if 'answers' in item:
                                if isinstance(item['answers'], list) and len(item['answers']) > 0:
                                    answer = item['answers'][0]
                                elif isinstance(item['answers'], dict) and 'answer' in item['answers']:
                                    answer = item['answers']['answer']

                            image_id = item.get('imageURL', item.get('image_id', item.get('image', '')))

                            if question and image_id:
                                text = f"Question: {question} Answer: {answer}" if answer else f"Question: {question}"

                                # Try to find image
                                if isinstance(image_id, str) and ('/' in image_id or '.' in image_id):
                                    image_path = image_id
                                else:
                                    image_path = f"{image_id}.jpg"

                                if not os.path.isabs(image_path):
                                    image_path = str(json_parent / image_path)

                                samples.append({
                                    'image': image_path,
                                    'caption': text,
                                })
                                file_sample_count += 1

            # Log how many samples were added from this file
            after_count = len(samples)
            added_count = after_count - before_count

            if added_count > 0:
                if is_main:
                    logger.info(f"✓ Loaded {added_count:,} samples from {json_file.name}")
            else:
                if is_main:
                    logger.debug(f"No samples loaded from {json_file.name}")

            # Free memory after processing each file
            del data
            gc.collect()

        # Filter by split if needed
        if self.split == 'train':
            samples = samples[:int(len(samples) * 0.9)]
        else:
            samples = samples[int(len(samples) * 0.9):]

        if is_main:
            logger.info("="*80)
            logger.info(f"FINAL DATASET: Loaded {len(samples):,} total samples for {self.split} split")
            logger.info(f"Memory usage: {_get_memory_usage_mb():.0f} MB")
            logger.info("="*80)

        return samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Handle CC3M lazy loading
        if sample.get('type') == 'cc3m':
            cc3m_data = self._get_cc3m_sample(sample['cc3m_idx'])
            if cc3m_data:
                pixel_values = self._load_image(cc3m_data['image'])
                caption = cc3m_data['caption']
            else:
                # Fallback to black image
                pixel_values = torch.zeros(3, self.image_size, self.image_size)
                caption = ""
        else:
            # Load image from path
            pixel_values = self._load_image(sample['image'])
            caption = sample.get('caption', '')

        # Tokenize caption
        text_data = self._tokenize(caption)

        return {
            'pixel_values': pixel_values,
            **text_data,
        }


class InstructionDataset(BaseVLMDataset):
    """Dataset for Stage 2 instruction tuning."""

    def _load_data(self) -> List[Dict[str, Any]]:
        """Load instruction data with memory safety limits."""
        import logging
        logger = logging.getLogger(__name__)

        is_main = _is_main_process()
        samples = []

        # Recursively search for JSON files
        json_files = list(self.data_dir.rglob('*.json'))

        if not json_files:
            if is_main:
                logger.warning(f"No JSON files found in {self.data_dir} or its subdirectories")
            return samples

        if is_main:
            logger.info(f"Found {len(json_files)} JSON files to process for instruction tuning")
            logger.info(f"⚠️ MEMORY SAFE: Max samples = {MAX_STAGE2_TOTAL_SAMPLES:,}")

        for json_file in json_files:
            # Check total sample limit
            if len(samples) >= MAX_STAGE2_TOTAL_SAMPLES:
                if is_main:
                    logger.info(f"⚠️ Reached max Stage 2 samples ({MAX_STAGE2_TOTAL_SAMPLES:,})")
                break

            if is_main:
                logger.info(f"Processing instruction file: {json_file}")

            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                if is_main:
                    logger.warning(f"Failed to load {json_file}: {e}")
                continue

            json_parent = json_file.parent
            file_samples = 0
            max_per_file = min(MAX_SAMPLES_PER_DATASET, MAX_STAGE2_TOTAL_SAMPLES - len(samples))

            if isinstance(data, list):
                for item in data:
                    if file_samples >= max_per_file:
                        break

                    if 'image' in item:
                        image_path = item['image']
                        # Resolve relative image paths
                        if not os.path.isabs(image_path):
                            # For LLaVA, images might be in a separate directory
                            image_path = str(json_parent / image_path)

                        # Try to load instruction-format data
                        if 'conversations' in item:
                            # LLaVA format with conversations
                            instruction = ""
                            response = ""
                            for conv in item['conversations']:
                                if conv.get('from') == 'human':
                                    instruction = conv.get('value', '')
                                elif conv.get('from') == 'gpt':
                                    response = conv.get('value', '')
                            if instruction and response:
                                samples.append({
                                    'image': image_path,
                                    'instruction': instruction,
                                    'response': response,
                                })
                                file_samples += 1
                        elif 'instruction' in item or 'question' in item:
                            instruction = item.get('instruction', item.get('question', ''))
                            response = item.get('response', item.get('answer', ''))
                            if instruction and response:
                                samples.append({
                                    'image': image_path,
                                    'instruction': instruction,
                                    'response': response,
                                })
                                file_samples += 1
                        # Fallback to caption data for instruction tuning
                        elif 'caption' in item or 'text' in item:
                            caption = item.get('caption', item.get('text', ''))
                            if caption:
                                # Convert caption to instruction format
                                samples.append({
                                    'image': image_path,
                                    'instruction': 'Describe this image.',
                                    'response': caption,
                                })
                                file_samples += 1

            logger.info(f"Loaded {file_samples} instruction samples from {json_file.name}")

        # Split
        if self.split == 'train':
            samples = samples[:int(len(samples) * 0.9)]
        else:
            samples = samples[int(len(samples) * 0.9):]

        logger.info(f"Loaded {len(samples)} samples for {self.split} split from {self.data_dir}")
        return samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        pixel_values = self._load_image(sample['image'])

        # Format as instruction-response
        text = f"Instruction: {sample['instruction']}\nResponse: {sample['response']}"
        text_data = self._tokenize(text)

        return {
            'pixel_values': pixel_values,
            **text_data,
        }


class ReasoningDataset(BaseVLMDataset):
    """
    Dataset for Stage 4 reasoning training (DeepSeek-R1 style).

    Supports both legacy format (reasoning_chain list) and new XML format.
    """

    def _load_data(self) -> List[Dict[str, Any]]:
        """Load reasoning data with chains."""
        import logging
        logger = logging.getLogger(__name__)

        samples = []

        # Recursively search for JSON files
        json_files = list(self.data_dir.rglob('*.json'))

        if not json_files:
            logger.warning(f"No JSON files found in {self.data_dir} or its subdirectories")
            return samples

        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load {json_file}: {e}")
                continue

            json_parent = json_file.parent

            if isinstance(data, list):
                for item in data:
                    sample = {
                        'instruction': item.get('instruction', ''),
                        'reasoning_chain': item.get('reasoning_chain', []),
                        'response': item.get('response', ''),
                        'robot_target': item.get('selected_robot', item.get('robot_target', item.get('answer'))),
                        'task': item.get('task', ''),
                        'format': item.get('format', 'legacy'),  # 'xml' or 'legacy'
                    }

                    # Handle image if present
                    if 'image' in item:
                        image_path = item['image']
                        if not os.path.isabs(image_path):
                            image_path = str(json_parent / image_path)
                        sample['image'] = image_path

                    # Handle chat-style prompt (DeepSeek-R1 style)
                    if 'prompt' in item:
                        sample['prompt'] = item['prompt']

                    samples.append(sample)

        # Split
        if self.split == 'train':
            samples = samples[:int(len(samples) * 0.9)]
        else:
            samples = samples[int(len(samples) * 0.9):]

        logger.info(f"Loaded {len(samples)} samples for {self.split} split from {self.data_dir}")
        return samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Load image if present
        if 'image' in sample and sample['image']:
            pixel_values = self._load_image(sample['image'])
        else:
            # Create placeholder image for text-only samples
            pixel_values = torch.zeros(3, self.image_size, self.image_size)

        # Check if this is XML format (DeepSeek-R1 style)
        is_xml_format = sample.get('format') == 'xml' or '<reasoning>' in sample.get('response', '')

        if is_xml_format:
            # XML format: response already contains <reasoning>...</reasoning><answer>...</answer>
            text = f"{sample['instruction']}\n\n{sample['response']}"
        else:
            # Legacy format: Build text from reasoning chain list
            reasoning_text = ""
            if sample['reasoning_chain']:
                reasoning_steps = "\n".join([
                    f"Step {i+1}: {step}"
                    for i, step in enumerate(sample['reasoning_chain'])
                ])
                reasoning_text = f"<reasoning>\n{reasoning_steps}\n</reasoning>\n<answer>\n{sample.get('robot_target', '')}\n</answer>"
                text = f"{sample['instruction']}\n\n{reasoning_text}"
            else:
                text = f"{sample['instruction']}\n{sample['response']}"

        text_data = self._tokenize(text)

        result = {
            'pixel_values': pixel_values,
            **text_data,
        }

        # Add robot target if available
        robot_mapping = {
            'Drone': 0, 'drone': 0,
            'Humanoid': 1, 'humanoid': 1,
            'Wheeled': 2, 'Robot with Wheels': 2, 'robot with wheels': 2, 'wheeled robot': 2,
            'Legged': 3, 'Robot with Legs': 3, 'robot with legs': 3, 'legged robot': 3,
            'Underwater': 4, 'Underwater Robot': 4, 'underwater robot': 4,
        }

        robot_target = sample.get('robot_target')
        if robot_target is not None:
            if isinstance(robot_target, str):
                # Try to normalize the robot name
                robot_target_lower = robot_target.lower().strip()
                robot_target = robot_mapping.get(robot_target, robot_mapping.get(robot_target_lower, 0))
            result['robot_target'] = torch.tensor(robot_target, dtype=torch.long)
            result['robot_target_names'] = sample.get('robot_target', '')  # Keep string name for rewards

        return result


def get_alignment_dataloader(
    data_dir: str,
    tokenizer: Any,
    batch_size: int = 32,
    split: str = 'train',
    distributed: bool = False,
    num_workers: int = None,  # Will use SAFE_NUM_WORKERS if not specified
    **kwargs,
) -> DataLoader:
    """
    Create dataloader for alignment stage.

    MEMORY SAFE: Uses limited num_workers and persistent_workers to prevent
    memory duplication and CPU oversubscription.
    """
    import logging
    logger = logging.getLogger(__name__)

    # Use safe default for num_workers
    if num_workers is None:
        num_workers = SAFE_NUM_WORKERS
    else:
        # Cap at safe maximum
        num_workers = min(num_workers, SAFE_NUM_WORKERS)

    if _is_main_process():
        logger.info(f"Creating alignment dataloader with {num_workers} workers")

    dataset = AlignmentDataset(
        data_dir=data_dir,
        tokenizer=tokenizer,
        split=split,
        **kwargs,
    )

    sampler = None
    if distributed and split == 'train' and dist.is_initialized():
        sampler = DistributedSampler(dataset, shuffle=True)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None and split == 'train'),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train'),
        persistent_workers=(num_workers > 0),  # Keep workers alive between batches
        prefetch_factor=2 if num_workers > 0 else None,  # Limit prefetching
    )


def get_instruction_dataloader(
    data_dir: str,
    tokenizer: Any,
    batch_size: int = 32,
    split: str = 'train',
    distributed: bool = False,
    num_workers: int = None,
    **kwargs,
) -> DataLoader:
    """
    Create dataloader for instruction tuning.

    MEMORY SAFE: Uses limited num_workers.
    """
    import logging
    logger = logging.getLogger(__name__)

    if num_workers is None:
        num_workers = SAFE_NUM_WORKERS
    else:
        num_workers = min(num_workers, SAFE_NUM_WORKERS)

    if _is_main_process():
        logger.info(f"Creating instruction dataloader with {num_workers} workers")

    dataset = InstructionDataset(
        data_dir=data_dir,
        tokenizer=tokenizer,
        split=split,
        **kwargs,
    )

    sampler = None
    if distributed and split == 'train' and dist.is_initialized():
        sampler = DistributedSampler(dataset, shuffle=True)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None and split == 'train'),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train'),
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )


def get_reasoning_dataloader(
    data_dir: str,
    tokenizer: Any,
    batch_size: int = 32,
    split: str = 'train',
    distributed: bool = False,
    num_workers: int = None,
    **kwargs,
) -> DataLoader:
    """
    Create dataloader for reasoning training.

    MEMORY SAFE: Uses limited num_workers.
    """
    import logging
    logger = logging.getLogger(__name__)

    if num_workers is None:
        num_workers = SAFE_NUM_WORKERS
    else:
        num_workers = min(num_workers, SAFE_NUM_WORKERS)

    if _is_main_process():
        logger.info(f"Creating reasoning dataloader with {num_workers} workers")

    dataset = ReasoningDataset(
        data_dir=data_dir,
        tokenizer=tokenizer,
        split=split,
        **kwargs,
    )

    sampler = None
    if distributed and split == 'train' and dist.is_initialized():
        sampler = DistributedSampler(dataset, shuffle=True)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None and split == 'train'),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train'),
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Custom collate function with error handling."""
    result = {}

    for key in batch[0].keys():
        values = [item[key] for item in batch if key in item]

        if len(values) == 0:
            continue

        if isinstance(values[0], torch.Tensor):
            try:
                result[key] = torch.stack(values)
            except Exception:
                # If stacking fails, skip this key
                pass
        else:
            result[key] = values

    return result

