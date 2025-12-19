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
        samples = []

        # Check for different data formats
        json_files = list(self.data_dir.glob('*.json'))

        for json_file in json_files:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if isinstance(data, list):
                for item in data:
                    if 'image' in item and ('caption' in item or 'text' in item):
                        samples.append({
                            'image': item['image'],
                            'caption': item.get('caption', item.get('text', '')),
                        })
            elif isinstance(data, dict):
                # Handle different JSON structures
                if 'annotations' in data:
                    for ann in data['annotations']:
                        samples.append({
                            'image': ann.get('image', ann.get('file_name', '')),
                            'caption': ann.get('caption', ''),
                        })

        # Filter by split if needed
        if self.split == 'train':
            samples = samples[:int(len(samples) * 0.9)]
        else:
            samples = samples[int(len(samples) * 0.9):]

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
        samples = []

        json_files = list(self.data_dir.glob('*.json'))

        for json_file in json_files:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if isinstance(data, list):
                for item in data:
                    if 'image' in item:
                        samples.append({
                            'image': item['image'],
                            'instruction': item.get('instruction', item.get('question', '')),
                            'response': item.get('response', item.get('answer', '')),
                        })

        # Split
        if self.split == 'train':
            samples = samples[:int(len(samples) * 0.9)]
        else:
            samples = samples[int(len(samples) * 0.9):]

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
        samples = []

        json_files = list(self.data_dir.glob('*.json'))

        for json_file in json_files:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if isinstance(data, list):
                for item in data:
                    if 'image' in item:
                        samples.append({
                            'image': item['image'],
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

