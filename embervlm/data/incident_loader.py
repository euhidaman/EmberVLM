"""
Incident Dataset Loader for EmberVLM

Loads and processes data from the Incidents1M dataset
for training incident understanding capabilities.
"""

import os
import json
import random
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable

import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from PIL import Image
import torchvision.transforms as transforms


# Incident categories
INCIDENT_CATEGORIES = [
    "earthquake", "fire", "flood", "hurricane", "landslide",
    "tornado", "tsunami", "volcanic_eruption", "wildfire",
    "building_collapse", "car_accident", "explosion", "industrial_accident",
    "oil_spill", "plane_crash", "ship_accident", "train_accident",
    "protest", "riot", "war_zone", "other"
]

PLACE_CATEGORIES = [
    "urban", "rural", "coastal", "forest", "desert", "mountain",
    "industrial", "residential", "commercial", "highway", "water_body"
]


class IncidentDataset(Dataset):
    """
    Dataset for Incidents1M data.

    Supports multiple data formats:
    - eccv_train.json / eccv_val.json: Basic incident annotations
    - multi_label_train.json / multi_label_val.json: Multi-label classifications
    """

    def __init__(
        self,
        data_dir: str,
        tokenizer: Any,
        split: str = 'train',
        max_length: int = 768,
        image_size: int = 224,
        transform: Optional[Callable] = None,
        use_multi_label: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.split = split
        self.max_length = max_length
        self.image_size = image_size
        self.use_multi_label = use_multi_label

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
        """Load incident data from JSON files."""
        samples = []

        # Determine which files to load
        if self.split == 'train':
            base_files = ['eccv_train.json', 'multi_label_train.json']
        else:
            base_files = ['eccv_val.json', 'multi_label_val.json']

        for filename in base_files:
            filepath = self.data_dir / filename
            if filepath.exists():
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Handle different formats
                if isinstance(data, dict):
                    # ECCV format: {image_id: annotations}
                    for image_id, ann in data.items():
                        sample = self._parse_sample(image_id, ann)
                        if sample:
                            samples.append(sample)
                elif isinstance(data, list):
                    # List format
                    for item in data:
                        sample = self._parse_list_item(item)
                        if sample:
                            samples.append(sample)

        return samples

    def _parse_sample(self, image_id: str, ann: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse sample from ECCV format."""
        try:
            # Get image path
            image_path = ann.get('image', ann.get('file_name', f'{image_id}.jpg'))

            # Get incident types
            incidents = ann.get('incidents', [])
            if isinstance(incidents, dict):
                incidents = [k for k, v in incidents.items() if v]

            # Get place types
            places = ann.get('places', [])
            if isinstance(places, dict):
                places = [k for k, v in places.items() if v]

            return {
                'image': image_path,
                'image_id': image_id,
                'incidents': incidents,
                'places': places,
                'description': ann.get('description', ''),
            }
        except Exception:
            return None

    def _parse_list_item(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse sample from list format."""
        try:
            return {
                'image': item.get('image', item.get('file_name', '')),
                'image_id': item.get('id', item.get('image_id', '')),
                'incidents': item.get('incidents', item.get('labels', [])),
                'places': item.get('places', []),
                'description': item.get('description', item.get('caption', '')),
            }
        except Exception:
            return None

    def _load_image(self, image_path: str) -> torch.Tensor:
        """Load and transform image."""
        try:
            if not os.path.isabs(image_path):
                # Try different image directories
                possible_paths = [
                    self.data_dir / 'images' / image_path,
                    self.data_dir / image_path,
                    self.data_dir.parent / 'images' / image_path,
                ]

                for path in possible_paths:
                    if path.exists():
                        image_path = path
                        break

            image = Image.open(image_path).convert('RGB')
            return self.transform(image)
        except Exception:
            return torch.zeros(3, self.image_size, self.image_size)

    def _create_incident_prompt(self, sample: Dict[str, Any]) -> str:
        """Create training prompt for incident understanding."""
        prompts = [
            "Describe the incident in this image.",
            "What emergency situation is shown in this image?",
            "Analyze this scene and identify any incidents.",
            "What type of disaster or emergency is depicted?",
        ]

        instruction = random.choice(prompts)

        # Create response
        incidents = sample.get('incidents', [])
        places = sample.get('places', [])
        description = sample.get('description', '')

        if incidents:
            incident_text = ', '.join(incidents)
            response = f"This image shows: {incident_text}."
        else:
            response = "No specific incident is clearly identifiable."

        if places:
            place_text = ', '.join(places)
            response += f" The scene appears to be in a {place_text} setting."

        if description:
            response += f" {description}"

        return f"Instruction: {instruction}\nResponse: {response}"

    def _create_multi_label_tensor(
        self,
        labels: List[str],
        categories: List[str]
    ) -> torch.Tensor:
        """Create multi-label tensor."""
        tensor = torch.zeros(len(categories))
        for label in labels:
            if label in categories:
                tensor[categories.index(label)] = 1.0
        return tensor

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Load image
        pixel_values = self._load_image(sample['image'])

        # Create text
        text = self._create_incident_prompt(sample)

        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        # Labels for LM
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        result = {
            'pixel_values': pixel_values,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }

        # Add multi-label classification targets
        if self.use_multi_label:
            result['incident_labels'] = self._create_multi_label_tensor(
                sample.get('incidents', []),
                INCIDENT_CATEGORIES
            )
            result['place_labels'] = self._create_multi_label_tensor(
                sample.get('places', []),
                PLACE_CATEGORIES
            )

        return result


def get_incident_dataloader(
    data_dir: str,
    tokenizer: Any,
    batch_size: int = 32,
    split: str = 'train',
    distributed: bool = False,
    num_workers: int = 4,
    **kwargs,
) -> DataLoader:
    """Create dataloader for incident data."""
    dataset = IncidentDataset(
        data_dir=data_dir,
        tokenizer=tokenizer,
        split=split,
        **kwargs,
    )

    sampler = None
    if distributed and split == 'train':
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

