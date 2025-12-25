"""
Data Loaders for EmberVLM Training

Provides data loaders for different training stages.
"""

import os
import json
import random
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Callable

import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.distributed as dist
from PIL import Image
import torchvision.transforms as transforms

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
        """Load and transform image from path or PIL Image."""
        try:
            # Check if input is already a PIL Image (e.g., from CC3M)
            if isinstance(image_input, Image.Image):
                # Ensure image is in RGB mode and properly loaded
                if image_input.mode != 'RGB':
                    image = image_input.convert('RGB')
                else:
                    image = image_input
                # Ensure image data is loaded
                image.load()
            else:
                # It's a path string
                image_path = image_input
                if not os.path.isabs(image_path):
                    image_path = self.data_dir / image_path
                image = Image.open(image_path).convert('RGB')

            return self.transform(image)
        except Exception as e:
            import logging
            logging.getLogger(__name__).debug(f"Failed to load image: {e}")
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

    def _load_cc3m_hf(self, logger) -> List[Dict[str, Any]]:
        """Load CC3M data using HuggingFace datasets library."""
        samples = []

        # CC3M is stored in HuggingFace WebDataset format
        cc3m_dir = self.data_dir / 'cc3m'

        if not cc3m_dir.exists():
            logger.debug("CC3M directory not found")
            return samples

        try:
            # Try to import datasets library
            try:
                from datasets import load_dataset
            except ImportError:
                logger.warning("HuggingFace datasets library not available - skipping CC3M. Install with: pip install datasets")
                return samples

            logger.info("Loading CC3M dataset from HuggingFace format...")

            # Load using HuggingFace datasets with the local cache
            # The dataset is already downloaded to cc3m_dir
            try:
                dataset = load_dataset(
                    "pixparse/cc3m-wds",
                    cache_dir=str(cc3m_dir),
                    split="train",
                    trust_remote_code=True
                )

                logger.info(f"Found {len(dataset):,} samples in CC3M dataset")

                # Limit to avoid OOM - sample 500k from the ~3M dataset
                samples_limit = 500000

                # Sample indices uniformly across the dataset
                if len(dataset) > samples_limit:
                    import numpy as np
                    indices = np.linspace(0, len(dataset) - 1, samples_limit, dtype=int)
                    dataset = dataset.select(indices)
                    logger.info(f"Sampled {samples_limit:,} from CC3M dataset")

                # Process samples with progress tracking
                successful = 0
                failed = 0

                for i, item in enumerate(dataset):
                    try:
                        # Extract image - CC3M stores as PIL Image in 'jpg' field
                        image = item.get('jpg') or item.get('image')

                        # Extract caption
                        caption = item.get('txt') or item.get('caption') or item.get('text', '')

                        if image is None or not caption:
                            failed += 1
                            continue

                        # Ensure image is PIL Image and in RGB mode
                        if not isinstance(image, Image.Image):
                            # Try to decode if it's bytes
                            if isinstance(image, bytes):
                                import io
                                image = Image.open(io.BytesIO(image))
                            else:
                                failed += 1
                                continue

                        # Convert to RGB and verify
                        if image.mode != 'RGB':
                            image = image.convert('RGB')
                        image.load()  # Force load to verify image is valid

                        caption = str(caption).strip()
                        if not caption:
                            failed += 1
                            continue

                        samples.append({
                            'image': image,
                            'caption': caption,
                        })
                        successful += 1

                        # Log progress every 50k samples
                        if successful > 0 and successful % 50000 == 0:
                            logger.info(f"  Loaded {successful:,} CC3M samples so far...")

                    except Exception as e:
                        failed += 1
                        if failed <= 10:  # Only log first 10 failures
                            logger.debug(f"Failed to process CC3M sample {i}: {e}")
                        continue

                if samples:
                    logger.info(f"✓ Loaded {len(samples):,} samples from CC3M (successful: {successful:,}, failed: {failed:,})")
                else:
                    logger.warning(f"✗ No samples loaded from CC3M dataset (failed: {failed:,})")

            except Exception as e:
                logger.warning(f"Failed to load CC3M with load_dataset: {e}")
                logger.info("CC3M dataset should be downloaded to data/base_vlm/cc3m/")
                return samples

        except Exception as e:
            logger.warning(f"Failed to load CC3M dataset: {e}")

        return samples

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
        """Load image-text pairs from multiple dataset formats."""
        import logging
        logger = logging.getLogger(__name__)

        samples = []

        # Load CC3M from HuggingFace format
        cc3m_samples = self._load_cc3m_hf(logger)
        samples.extend(cc3m_samples)

        # Load RefCOCO variants from Arrow files
        refcoco_samples = self._load_refcoco_arrow(logger)
        samples.extend(refcoco_samples)

        # Recursively search for JSON files in subdirectories
        json_files = list(self.data_dir.rglob('*.json'))

        if not json_files:
            logger.warning(f"No JSON files found in {self.data_dir} or its subdirectories")
            return samples

        logger.info(f"Found {len(json_files)} JSON files to process")

        # Skip metadata files and non-VL datasets based on actual file structure
        skip_patterns = [
            'download_summary',     # Download metadata
            'dataset_info',         # HuggingFace dataset metadata (refcoco/refcoco+/refcocog dirs)
            'instances_',           # COCO object detection (not VL)
            'person_keypoints_',    # COCO pose estimation (not VL)
            '__MACOSX',             # Mac OS metadata folder
            '.lock',                # Lock files
            '_builder.lock',        # HF builder locks
            '_incomplete',          # Incomplete downloads
            'dataset.json',         # OCR-VQA has empty dataset.json file
            'readme.txt',           # Documentation files
            'LICENCE.txt',          # License files
            'loadDataset.py',       # Python scripts
            '.py',                  # Any Python files
            '.csv',                 # CSV files (not primary data format)
            '.download_attempted',  # Download markers
        ]

        for json_file in json_files:
            # Skip metadata files
            file_name_lower = json_file.name.lower()
            if any(pattern in file_name_lower for pattern in skip_patterns):
                logger.debug(f"Skipping metadata/non-data file: {json_file.name}")
                continue

            # Skip files in __MACOSX directories
            if '__MACOSX' in str(json_file):
                logger.debug(f"Skipping Mac metadata file: {json_file}")
                continue

            logger.info(f"Processing file: {json_file}")

            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load {json_file}: {e}")
                continue

            # Get the parent directory for resolving image paths
            json_parent = json_file.parent
            before_count = len(samples)

            if isinstance(data, list):
                # Handle list format (LLaVA, CC3M, etc.)
                for item in data:
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
                        logger.warning(f"No image directories found for {json_file}")
                    else:
                        logger.info(f"Found image directories: {[str(d) for d in image_dirs]}")

                        for ann in data['annotations']:
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

                # ===== VQA Format (questions + annotations) =====
                elif 'questions' in data or ('annotations' in data and any('question' in ann or 'answer' in ann for ann in data.get('annotations', [])[:5])):
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
                                            found_image = True
                                            break

                # ===== GQA Format =====
                elif isinstance(data, dict) and len(data) > 0:
                    # Check if it's GQA format: dict of dicts with question/answer structure
                    first_values = list(data.values())[:5]
                    is_gqa = all(isinstance(v, dict) and any(k in v for k in ['question', 'answer', 'imageId']) for v in first_values if isinstance(v, dict))

                    if is_gqa:
                        # GQA is dict of dicts: {question_id: {question, answer, imageId, ...}}
                        logger.info(f"Detected GQA format: {len(data)} questions")

                        # Find GQA images directory
                        image_dirs = []
                        for search_dir in [json_parent, json_parent.parent]:
                            for pattern in ['images', 'allImages']:
                                found_dirs = list(search_dir.glob(pattern))
                                image_dirs.extend(found_dirs)

                        # Process GQA questions (limit to 500K per file to avoid OOM)
                        # This includes train/val/test/challenge - we use ALL data
                        gqa_limit = min(500000, len(data))
                        gqa_samples_added = 0

                        for qid, item in list(data.items())[:gqa_limit]:
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
                                            gqa_samples_added += 1
                                            break

                        if gqa_samples_added > 0:
                            logger.info(f"Loaded {gqa_samples_added} GQA samples from {json_file.name}")

                # ===== RefCOCO Format =====
                elif 'refs' in data or any('ref_id' in str(k) for k in list(data.keys())[:5]):
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
                        # RefCOCO has referring expressions
                        sentences = ref.get('sentences', [])
                        image_id = ref.get('image_id', '')

                        for sent in sentences:
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
                                            break

                # ===== OCR-VQA Format =====
                # Check if it's actually OCR-VQA data (has questions, not just metadata)
                elif 'data' in data and isinstance(data['data'], list) and len(data['data']) > 0:
                    # Verify first item has OCR-VQA structure
                    first_item = data['data'][0] if isinstance(data['data'], list) else {}
                    if 'question' in first_item or 'imageURL' in first_item:
                        logger.info(f"Detected OCR-VQA format")

                        items = data['data']

                        for item in items[:50000]:  # Limit for memory
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

            added_count = len(samples) - before_count
            # Log how many samples were added from this file
            after_count = len(samples)
            added_count = after_count - before_count

            if added_count > 0:
                logger.info(f"✓ Loaded {added_count:,} samples from {json_file.name}")
            else:
                logger.warning(f"✗ No samples loaded from {json_file.name}")

        # Filter by split if needed
        if self.split == 'train':
            samples = samples[:int(len(samples) * 0.9)]
        else:
            samples = samples[int(len(samples) * 0.9):]

        logger.info(f"="*80)
        logger.info(f"FINAL DATASET: Loaded {len(samples):,} total samples for {self.split} split")
        logger.info(f"Data sources: COCO (captions), VQA v2, OK-VQA, A-OKVQA, GQA (all splits),")
        logger.info(f"              LLaVA-Instruct, RefCOCO/+/g, OCR-VQA, CC3M, LAION-COCO")
        logger.info(f"="*80)
        return samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Load image
        pixel_values = self._load_image(sample['image'])

        # Tokenize caption
        text_data = self._tokenize(sample['caption'])

        return {
            'pixel_values': pixel_values,
            **text_data,
        }


class InstructionDataset(BaseVLMDataset):
    """Dataset for Stage 2 instruction tuning."""

    def _load_data(self) -> List[Dict[str, Any]]:
        """Load instruction data."""
        import logging
        logger = logging.getLogger(__name__)

        samples = []

        # Recursively search for JSON files
        json_files = list(self.data_dir.rglob('*.json'))

        if not json_files:
            logger.warning(f"No JSON files found in {self.data_dir} or its subdirectories")
            return samples

        logger.info(f"Found {len(json_files)} JSON files to process for instruction tuning")

        for json_file in json_files:
            logger.info(f"Processing instruction file: {json_file}")

            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load {json_file}: {e}")
                continue

            json_parent = json_file.parent
            file_samples = 0

            if isinstance(data, list):
                for item in data:
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
    """Dataset for Stage 4 reasoning training."""

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
                    if 'image' in item:
                        image_path = item['image']
                        if not os.path.isabs(image_path):
                            image_path = str(json_parent / image_path)
                        samples.append({
                            'image': image_path,
                            'instruction': item.get('instruction', ''),
                            'reasoning_chain': item.get('reasoning_chain', []),
                            'response': item.get('response', ''),
                            'robot_target': item.get('selected_robot', item.get('robot_target')),
                        })

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

        # Format with reasoning chain
        reasoning_text = ""
        if sample['reasoning_chain']:
            reasoning_steps = "\n".join([
                f"Step {i+1}: {step}"
                for i, step in enumerate(sample['reasoning_chain'])
            ])
            reasoning_text = f"<|reasoning_start|>\n{reasoning_steps}\n<|reasoning_end|>\n"

        text = f"{sample['instruction']}\n{reasoning_text}{sample['response']}"
        text_data = self._tokenize(text)

        result = {
            'pixel_values': pixel_values,
            **text_data,
        }

        # Add robot target if available
        robot_mapping = {
            'Drone': 0, 'Humanoid': 1, 'Wheeled': 2,
            'Legged': 3, 'Underwater': 4
        }

        robot_target = sample.get('robot_target')
        if robot_target is not None:
            if isinstance(robot_target, str):
                robot_target = robot_mapping.get(robot_target, 0)
            result['robot_target'] = torch.tensor(robot_target, dtype=torch.long)

        return result


def get_alignment_dataloader(
    data_dir: str,
    tokenizer: Any,
    batch_size: int = 32,
    split: str = 'train',
    distributed: bool = False,
    num_workers: int = 4,
    **kwargs,
) -> DataLoader:
    """Create dataloader for alignment stage."""
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
    )


def get_instruction_dataloader(
    data_dir: str,
    tokenizer: Any,
    batch_size: int = 32,
    split: str = 'train',
    distributed: bool = False,
    num_workers: int = 4,
    **kwargs,
) -> DataLoader:
    """Create dataloader for instruction tuning."""
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
    )


def get_reasoning_dataloader(
    data_dir: str,
    tokenizer: Any,
    batch_size: int = 32,
    split: str = 'train',
    distributed: bool = False,
    num_workers: int = 4,
    **kwargs,
) -> DataLoader:
    """Create dataloader for reasoning training."""
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
    )


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Custom collate function."""
    result = {}

    for key in batch[0].keys():
        values = [item[key] for item in batch]

        if isinstance(values[0], torch.Tensor):
            result[key] = torch.stack(values)
        else:
            result[key] = values

    return result

