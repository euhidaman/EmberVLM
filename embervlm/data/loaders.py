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

    def _load_image(self, image_path: str) -> torch.Tensor:
        """Load and transform image."""
        try:
            if not os.path.isabs(image_path):
                image_path = self.data_dir / image_path

            image = Image.open(image_path).convert('RGB')
            return self.transform(image)
        except Exception as e:
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
    """Dataset for Stage 1 visual-language alignment."""

    def _load_data(self) -> List[Dict[str, Any]]:
        """Load image-caption pairs."""
        import logging
        logger = logging.getLogger(__name__)

        samples = []

        # Recursively search for JSON files in subdirectories
        json_files = list(self.data_dir.rglob('*.json'))

        if not json_files:
            logger.warning(f"No JSON files found in {self.data_dir} or its subdirectories")
            return samples

        logger.info(f"Found {len(json_files)} JSON files to process")

        for json_file in json_files:
            # Skip non-caption files (instances, keypoints, etc.)
            if 'captions' not in json_file.name.lower():
                continue

            logger.info(f"Processing caption file: {json_file}")

            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load {json_file}: {e}")
                continue

            # Get the parent directory for resolving image paths
            json_parent = json_file.parent

            if isinstance(data, list):
                for item in data:
                    if 'image' in item and ('caption' in item or 'text' in item):
                        image_path = item['image']
                        # If image path is relative, resolve it relative to JSON location
                        if not os.path.isabs(image_path):
                            image_path = str(json_parent / image_path)
                        samples.append({
                            'image': image_path,
                            'caption': item.get('caption', item.get('text', '')),
                        })
            elif isinstance(data, dict):
                # Handle COCO-style format
                if 'annotations' in data and 'images' in data:
                    logger.info(f"Processing COCO format: {len(data['images'])} images, {len(data['annotations'])} annotations")

                    # Build image id to filename mapping
                    image_map = {img['id']: img['file_name'] for img in data['images']}

                    # Determine image directory (handle both 2014 and 2017 formats)
                    image_dirs = []
                    # Look in parent and parent's parent directories
                    for search_dir in [json_parent, json_parent.parent]:
                        for pattern in ['train2017', 'val2017', 'train2014', 'val2014', 'images']:
                            found_dirs = list(search_dir.glob(pattern))
                            image_dirs.extend(found_dirs)

                    if not image_dirs:
                        logger.warning(f"No image directories found for {json_file}")
                        continue

                    logger.info(f"Found image directories: {[str(d) for d in image_dirs]}")

                    for ann in data['annotations']:
                        image_id = ann.get('image_id')
                        caption = ann.get('caption', '')

                        if not caption:
                            continue

                        if image_id in image_map:
                            image_filename = image_map[image_id]
                            # Try to find image in any of the image directories
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

                    logger.info(f"Loaded {len(samples)} caption samples from {json_file.name}")

                # Handle simple dict format
                elif 'annotations' in data:
                    for ann in data['annotations']:
                        caption = ann.get('caption', '')
                        if not caption:
                            continue
                        image_path = ann.get('image', ann.get('file_name', ''))
                        if not os.path.isabs(image_path):
                            image_path = str(json_parent / image_path)
                        samples.append({
                            'image': image_path,
                            'caption': caption,
                        })

        # Filter by split if needed
        if self.split == 'train':
            samples = samples[:int(len(samples) * 0.9)]
        else:
            samples = samples[int(len(samples) * 0.9):]

        logger.info(f"Loaded {len(samples)} samples for {self.split} split from {self.data_dir}")
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

