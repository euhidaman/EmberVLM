#!/usr/bin/env python3
"""
Download Incidents Dataset Images with Comprehensive Logging & WandB Integration
Downloads images for the ECCV Incidents Dataset and filters annotations.

The Incidents Dataset is from:
- Paper: "Incidents: A Large-Scale Dataset for Multi-Label Classification" (ECCV 2020)
- Repository: https://github.com/ethanweber/IncidentsDataset

Usage:
    python download_incidents_images.py --output-dir ./data/incidents_images
    python download_incidents_images.py --output-dir ./data/incidents_images --max-images 10000
    python download_incidents_images.py --annotations-dir incidents-dataset --check-only

Features:
    - Comprehensive logging (console + file)
    - WandB integration for remote monitoring
    - Annotation filtering (excludes missing images)
    - Detailed status reports with totals
    - Resume support (skips existing images)
"""

import os
import sys
import json
import argparse
import logging
import hashlib
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import threading

# Try to import optional dependencies
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    tqdm = None

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

# Constants
WANDB_PROJECT = "EmberVLM-incidents-download"
LOG_FILE = "incidents_download.log"
STATUS_FILE = "incidents_download_status.json"
FILTERED_SUFFIX = "_filtered"


@dataclass
class DownloadStats:
    """Track download statistics."""
    total_annotations: int = 0
    total_images_found: int = 0
    total_images_missing: int = 0
    total_images_downloaded: int = 0
    total_images_failed: int = 0
    total_images_skipped: int = 0  # Already exists
    total_images_no_url: int = 0
    total_bytes_downloaded: int = 0
    annotations_kept: int = 0
    annotations_dropped: int = 0
    start_time: str = ""
    end_time: str = ""
    duration_seconds: float = 0.0

    # Per-file stats
    file_stats: Dict[str, Dict] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


class IncidentsDownloader:
    """
    Download and manage Incidents Dataset images with comprehensive logging.
    """

    def __init__(
        self,
        annotations_dir: str = "incidents-dataset",
        output_dir: str = "data/incidents_images",
        log_level: str = "INFO",
        use_wandb: bool = True,
        num_workers: int = 8,
        timeout: int = 30
    ):
        self.annotations_dir = Path(annotations_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_workers = num_workers
        self.timeout = timeout

        # Statistics
        self.stats = DownloadStats()
        self.stats.start_time = datetime.now().isoformat()

        # Image status tracking
        self.image_status: Dict[str, Dict] = {}  # image_id -> {status, path, url, error}

        # Thread lock for stats updates
        self._lock = threading.Lock()

        # Setup logging
        self.logger = self._setup_logging(log_level)

        # Initialize WandB
        self.wandb_run = None
        if use_wandb and WANDB_AVAILABLE:
            self._init_wandb()
        elif use_wandb and not WANDB_AVAILABLE:
            self.logger.warning("WandB not available. Install with: pip install wandb")

        self.logger.info("=" * 60)
        self.logger.info("Incidents Dataset Downloader Initialized")
        self.logger.info("=" * 60)
        self.logger.info(f"Annotations directory: {self.annotations_dir.absolute()}")
        self.logger.info(f"Output directory: {self.output_dir.absolute()}")
        self.logger.info(f"Workers: {self.num_workers}, Timeout: {self.timeout}s")

    def _setup_logging(self, log_level: str) -> logging.Logger:
        """Setup comprehensive logging."""
        logger = logging.getLogger("IncidentsDownloader")
        logger.setLevel(getattr(logging, log_level.upper()))
        logger.handlers.clear()

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_format = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)

        # File handler
        log_path = self.output_dir / LOG_FILE
        file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s [%(levelname)s] [%(funcName)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

        return logger

    def _init_wandb(self):
        """Initialize WandB for remote monitoring."""
        try:
            run_name = f"incidents-download-{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            self.wandb_run = wandb.init(
                project=WANDB_PROJECT,
                name=run_name,
                config={
                    'annotations_dir': str(self.annotations_dir),
                    'output_dir': str(self.output_dir),
                    'num_workers': self.num_workers,
                    'timeout': self.timeout
                },
                tags=['incidents', 'download', 'embervlm']
            )

            self.logger.info(f"WandB initialized: {run_name}")
            self.logger.info(f"View logs at: {self.wandb_run.url}")

        except Exception as e:
            self.logger.error(f"Failed to initialize WandB: {e}")
            self.wandb_run = None

    def _log_to_wandb(self, metrics: Dict, step: Optional[int] = None):
        """Log metrics to WandB."""
        if self.wandb_run:
            try:
                wandb.log(metrics, step=step)
            except Exception as e:
                self.logger.debug(f"WandB logging failed: {e}")

    def load_annotations(self, filename: str) -> Tuple[Dict, str]:
        """
        Load annotations from a JSON file.

        Returns:
            Tuple of (data dict, format type)
        """
        filepath = self.annotations_dir / filename
        if not filepath.exists():
            self.logger.warning(f"File not found: {filepath}")
            return {}, "unknown"

        self.logger.info(f"Loading {filename}...")

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Detect format
            if isinstance(data, dict):
                format_type = "dict"
                self.logger.info(f"  Format: dict with {len(data)} entries")
            elif isinstance(data, list):
                format_type = "list"
                self.logger.info(f"  Format: list with {len(data)} entries")
            else:
                format_type = "unknown"
                self.logger.warning(f"  Format: unknown ({type(data)})")

            return data, format_type

        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error in {filename}: {e}")
            return {}, "error"
        except Exception as e:
            self.logger.error(f"Error loading {filename}: {e}")
            return {}, "error"

    def extract_image_info(self, data: Any, format_type: str) -> Dict[str, Dict]:
        """
        Extract image information from annotations.

        Returns:
            Dict mapping image_id to {url, annotations, ...}
        """
        image_info = {}

        if format_type == "dict":
            for key, value in data.items():
                image_id = str(key)
                info = {'id': image_id, 'annotations': value}

                # Extract URL from various possible fields
                if isinstance(value, dict):
                    url = (value.get('url') or value.get('image_url') or
                           value.get('flickr_url') or value.get('image'))
                    if url and isinstance(url, str) and url.startswith('http'):
                        info['url'] = url
                elif isinstance(value, str) and value.startswith('http'):
                    info['url'] = value

                image_info[image_id] = info

        elif format_type == "list":
            for i, item in enumerate(data):
                if isinstance(item, dict):
                    image_id = str(item.get('id') or item.get('image_id') or
                                   item.get('filename') or i)
                    info = {'id': image_id, 'annotations': item}

                    url = (item.get('url') or item.get('image_url') or
                           item.get('flickr_url') or item.get('image'))
                    if url and isinstance(url, str) and url.startswith('http'):
                        info['url'] = url

                    image_info[image_id] = info

        return image_info

    def check_existing_images(self, image_info: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Check which images already exist locally.

        Updates image_info with 'exists' and 'local_path' fields.
        """
        self.logger.info("Checking for existing images...")

        existing_count = 0
        extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp']

        for image_id, info in image_info.items():
            found = False

            for ext in extensions:
                local_path = self.output_dir / f"{image_id}{ext}"
                if local_path.exists():
                    info['exists'] = True
                    info['local_path'] = str(local_path)
                    info['status'] = 'exists'
                    existing_count += 1
                    found = True
                    break

            if not found:
                info['exists'] = False
                info['local_path'] = None

        self.logger.info(f"  Found {existing_count} existing images")
        return image_info

    def download_image(self, image_id: str, url: str, timeout: int = None) -> Dict:
        """
        Download a single image.

        Returns:
            Dict with status, path, error, size
        """
        timeout = timeout or self.timeout
        result = {
            'image_id': image_id,
            'url': url,
            'status': 'pending',
            'path': None,
            'error': None,
            'size': 0
        }

        # Determine file extension
        ext = '.jpg'
        url_lower = url.lower()
        if '.png' in url_lower:
            ext = '.png'
        elif '.gif' in url_lower:
            ext = '.gif'
        elif '.webp' in url_lower:
            ext = '.webp'

        output_path = self.output_dir / f"{image_id}{ext}"

        # Skip if exists
        if output_path.exists():
            result['status'] = 'skipped'
            result['path'] = str(output_path)
            result['size'] = output_path.stat().st_size
            return result

        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            request = urllib.request.Request(url, headers=headers)

            with urllib.request.urlopen(request, timeout=timeout) as response:
                content = response.read()

            with open(output_path, 'wb') as f:
                f.write(content)

            result['status'] = 'success'
            result['path'] = str(output_path)
            result['size'] = len(content)

        except urllib.error.HTTPError as e:
            result['status'] = 'failed'
            result['error'] = f"HTTP {e.code}: {e.reason}"
        except urllib.error.URLError as e:
            result['status'] = 'failed'
            result['error'] = f"URL Error: {e.reason}"
        except TimeoutError:
            result['status'] = 'failed'
            result['error'] = "Timeout"
        except Exception as e:
            result['status'] = 'failed'
            result['error'] = str(e)

        return result

    def download_images_parallel(
        self,
        image_info: Dict[str, Dict],
        max_images: Optional[int] = None
    ) -> Dict[str, Dict]:
        """
        Download images in parallel with progress tracking.
        """
        # Filter to images with URLs that don't exist
        to_download = [
            (img_id, info['url'])
            for img_id, info in image_info.items()
            if info.get('url') and not info.get('exists')
        ]

        if max_images:
            to_download = to_download[:max_images]

        total = len(to_download)
        self.logger.info(f"Downloading {total} images with {self.num_workers} workers...")

        if total == 0:
            self.logger.info("No images to download")
            return {}

        results = {}
        completed = 0
        success_count = 0
        failed_count = 0
        skipped_count = 0
        total_bytes = 0

        start_time = time.time()

        # Progress bar if available
        pbar = None
        if TQDM_AVAILABLE:
            pbar = tqdm(total=total, desc="Downloading", unit="img")

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {}

            for image_id, url in to_download:
                future = executor.submit(self.download_image, image_id, url)
                futures[future] = image_id

            for future in as_completed(futures):
                image_id = futures[future]

                try:
                    result = future.result()
                    results[image_id] = result

                    # Update stats
                    with self._lock:
                        completed += 1

                        if result['status'] == 'success':
                            success_count += 1
                            total_bytes += result['size']
                            self.stats.total_images_downloaded += 1
                            self.stats.total_bytes_downloaded += result['size']
                        elif result['status'] == 'skipped':
                            skipped_count += 1
                            self.stats.total_images_skipped += 1
                        else:
                            failed_count += 1
                            self.stats.total_images_failed += 1
                            self.logger.debug(f"Failed: {image_id} - {result['error']}")

                    # Update progress
                    if pbar:
                        pbar.update(1)
                        pbar.set_postfix({
                            'ok': success_count,
                            'fail': failed_count,
                            'skip': skipped_count
                        })

                    # Log to WandB periodically
                    if completed % 100 == 0:
                        elapsed = time.time() - start_time
                        speed = total_bytes / elapsed / 1024 if elapsed > 0 else 0

                        self._log_to_wandb({
                            'download/completed': completed,
                            'download/total': total,
                            'download/success': success_count,
                            'download/failed': failed_count,
                            'download/skipped': skipped_count,
                            'download/progress_pct': completed / total * 100,
                            'download/speed_kbps': speed,
                            'download/bytes_total': total_bytes
                        }, step=completed)

                except Exception as e:
                    self.logger.error(f"Error processing {image_id}: {e}")
                    results[image_id] = {
                        'image_id': image_id,
                        'status': 'error',
                        'error': str(e)
                    }
                    failed_count += 1

        if pbar:
            pbar.close()

        # Final stats
        elapsed = time.time() - start_time
        speed = total_bytes / elapsed / 1024 if elapsed > 0 else 0

        self.logger.info("-" * 40)
        self.logger.info("Download Batch Complete:")
        self.logger.info(f"  Completed: {completed}/{total}")
        self.logger.info(f"  Success: {success_count}")
        self.logger.info(f"  Failed: {failed_count}")
        self.logger.info(f"  Skipped (existing): {skipped_count}")
        self.logger.info(f"  Total downloaded: {total_bytes / 1024 / 1024:.2f} MB")
        self.logger.info(f"  Speed: {speed:.2f} KB/s")
        self.logger.info(f"  Duration: {elapsed:.1f}s")

        return results

    def filter_annotations(
        self,
        data: Any,
        format_type: str,
        image_info: Dict[str, Dict],
        download_results: Dict[str, Dict]
    ) -> Tuple[Any, int, int]:
        """
        Filter annotations to only include entries with valid images.

        Returns:
            Tuple of (filtered_data, kept_count, dropped_count)
        """
        kept = 0
        dropped = 0

        # Merge download results into image_info
        for img_id, result in download_results.items():
            if img_id in image_info:
                image_info[img_id]['download_result'] = result

        # Determine which images are valid
        valid_images = set()
        for img_id, info in image_info.items():
            is_valid = False

            # Image exists locally
            if info.get('exists'):
                is_valid = True
            # Download succeeded
            elif info.get('download_result', {}).get('status') == 'success':
                is_valid = True
            # Download was skipped (already exists)
            elif info.get('download_result', {}).get('status') == 'skipped':
                is_valid = True

            if is_valid:
                valid_images.add(img_id)

        self.logger.info(f"Valid images: {len(valid_images)}")

        # Filter based on format
        if format_type == "dict":
            filtered_data = {}
            for key, value in data.items():
                if str(key) in valid_images:
                    filtered_data[key] = value
                    kept += 1
                else:
                    dropped += 1

        elif format_type == "list":
            filtered_data = []
            for i, item in enumerate(data):
                if isinstance(item, dict):
                    img_id = str(item.get('id') or item.get('image_id') or
                                 item.get('filename') or i)
                else:
                    img_id = str(i)

                if img_id in valid_images:
                    filtered_data.append(item)
                    kept += 1
                else:
                    dropped += 1
        else:
            filtered_data = data
            kept = len(valid_images)

        return filtered_data, kept, dropped

    def save_filtered_annotations(
        self,
        filename: str,
        filtered_data: Any,
        kept: int,
        dropped: int
    ):
        """Save filtered annotations to a new file."""
        # Create filtered filename
        name_parts = filename.rsplit('.', 1)
        if len(name_parts) == 2:
            filtered_filename = f"{name_parts[0]}{FILTERED_SUFFIX}.{name_parts[1]}"
        else:
            filtered_filename = f"{filename}{FILTERED_SUFFIX}"

        output_path = self.output_dir / filtered_filename

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(filtered_data, f, indent=2)

        self.logger.info(f"Saved filtered annotations: {output_path}")
        self.logger.info(f"  Kept: {kept}, Dropped: {dropped}")

        return str(output_path)

    def process_annotation_file(
        self,
        filename: str,
        max_images: Optional[int] = None,
        download: bool = True
    ) -> Dict:
        """
        Process a single annotation file.

        Returns:
            Dict with processing results
        """
        self.logger.info("=" * 60)
        self.logger.info(f"Processing: {filename}")
        self.logger.info("=" * 60)

        result = {
            'filename': filename,
            'status': 'pending',
            'total_annotations': 0,
            'images_with_url': 0,
            'images_no_url': 0,
            'images_existing': 0,
            'images_downloaded': 0,
            'images_failed': 0,
            'annotations_kept': 0,
            'annotations_dropped': 0,
            'filtered_file': None
        }

        # Load annotations
        data, format_type = self.load_annotations(filename)
        if format_type == "error" or not data:
            result['status'] = 'error'
            return result

        # Extract image info
        image_info = self.extract_image_info(data, format_type)
        result['total_annotations'] = len(image_info)
        self.stats.total_annotations += len(image_info)

        # Count URLs
        with_url = sum(1 for info in image_info.values() if info.get('url'))
        without_url = len(image_info) - with_url
        result['images_with_url'] = with_url
        result['images_no_url'] = without_url
        self.stats.total_images_no_url += without_url

        self.logger.info(f"  Total entries: {len(image_info)}")
        self.logger.info(f"  With URL: {with_url}")
        self.logger.info(f"  Without URL: {without_url}")

        # Check existing images
        image_info = self.check_existing_images(image_info)
        existing = sum(1 for info in image_info.values() if info.get('exists'))
        result['images_existing'] = existing
        self.stats.total_images_found += existing

        self.logger.info(f"  Already downloaded: {existing}")

        # Download images
        download_results = {}
        if download and with_url > 0:
            download_results = self.download_images_parallel(image_info, max_images)

            # Update result counts
            success = sum(1 for r in download_results.values() if r['status'] == 'success')
            failed = sum(1 for r in download_results.values() if r['status'] == 'failed')
            result['images_downloaded'] = success
            result['images_failed'] = failed

        # Filter annotations
        filtered_data, kept, dropped = self.filter_annotations(
            data, format_type, image_info, download_results
        )

        result['annotations_kept'] = kept
        result['annotations_dropped'] = dropped
        self.stats.annotations_kept += kept
        self.stats.annotations_dropped += dropped

        # Save filtered annotations
        if kept > 0:
            filtered_file = self.save_filtered_annotations(
                filename, filtered_data, kept, dropped
            )
            result['filtered_file'] = filtered_file

        result['status'] = 'complete'

        # Store in file stats
        self.stats.file_stats[filename] = result

        # Log to WandB
        self._log_to_wandb({
            f'file/{filename}/total': result['total_annotations'],
            f'file/{filename}/kept': kept,
            f'file/{filename}/dropped': dropped,
            f'file/{filename}/downloaded': result['images_downloaded'],
            f'file/{filename}/failed': result['images_failed']
        })

        return result

    def process_all_files(
        self,
        max_images_per_file: Optional[int] = None,
        download: bool = True
    ):
        """Process all annotation files."""
        json_files = [
            'eccv_train.json',
            'eccv_val.json',
            'multi_label_train.json',
            'multi_label_val.json'
        ]

        # Check which files exist
        available_files = []
        for filename in json_files:
            if (self.annotations_dir / filename).exists():
                available_files.append(filename)
            else:
                self.logger.warning(f"File not found: {filename}")

        self.logger.info(f"Found {len(available_files)} annotation files")

        # Process each file
        all_results = {}
        for filename in available_files:
            result = self.process_annotation_file(
                filename,
                max_images=max_images_per_file,
                download=download
            )
            all_results[filename] = result

        return all_results

    def print_final_summary(self):
        """Print comprehensive final summary with totals."""
        self.stats.end_time = datetime.now().isoformat()

        # Calculate duration
        start = datetime.fromisoformat(self.stats.start_time)
        end = datetime.fromisoformat(self.stats.end_time)
        self.stats.duration_seconds = (end - start).total_seconds()

        # Calculate totals
        total_images_processed = (
            self.stats.total_images_downloaded +
            self.stats.total_images_failed +
            self.stats.total_images_skipped +
            self.stats.total_images_found
        )

        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info("FINAL SUMMARY - INCIDENTS DATASET DOWNLOAD")
        self.logger.info("=" * 70)
        self.logger.info("")

        # Annotation Summary
        self.logger.info("ANNOTATIONS:")
        self.logger.info(f"  Total annotations scanned:    {self.stats.total_annotations:>10,}")
        self.logger.info(f"  Annotations kept (valid img): {self.stats.annotations_kept:>10,}")
        self.logger.info(f"  Annotations dropped (no img): {self.stats.annotations_dropped:>10,}")
        self.logger.info("")

        # Image Summary
        self.logger.info("IMAGES:")
        self.logger.info(f"  Already existing locally:     {self.stats.total_images_found:>10,}")
        self.logger.info(f"  Successfully downloaded:      {self.stats.total_images_downloaded:>10,}")
        self.logger.info(f"  Skipped (already exists):     {self.stats.total_images_skipped:>10,}")
        self.logger.info(f"  Failed to download:           {self.stats.total_images_failed:>10,}")
        self.logger.info(f"  No URL available:             {self.stats.total_images_no_url:>10,}")
        self.logger.info(f"  ----------------------------------------")
        self.logger.info(f"  TOTAL VALID IMAGES:           {self.stats.annotations_kept:>10,}")
        self.logger.info(f"  TOTAL MISSING IMAGES:         {self.stats.annotations_dropped:>10,}")
        self.logger.info("")

        # Data Transfer
        mb_downloaded = self.stats.total_bytes_downloaded / 1024 / 1024
        self.logger.info("DATA TRANSFER:")
        self.logger.info(f"  Total downloaded:             {mb_downloaded:>10.2f} MB")
        self.logger.info(f"  Duration:                     {self.stats.duration_seconds:>10.1f} seconds")
        if self.stats.duration_seconds > 0:
            speed = mb_downloaded / self.stats.duration_seconds
            self.logger.info(f"  Average speed:                {speed:>10.2f} MB/s")
        self.logger.info("")

        # Per-file breakdown
        if self.stats.file_stats:
            self.logger.info("PER-FILE BREAKDOWN:")
            self.logger.info("-" * 70)
            self.logger.info(f"{'File':<35} {'Total':>8} {'Kept':>8} {'Dropped':>8} {'DL':>6}")
            self.logger.info("-" * 70)

            for filename, stats in self.stats.file_stats.items():
                self.logger.info(
                    f"{filename:<35} "
                    f"{stats['total_annotations']:>8,} "
                    f"{stats['annotations_kept']:>8,} "
                    f"{stats['annotations_dropped']:>8,} "
                    f"{stats['images_downloaded']:>6,}"
                )

            self.logger.info("-" * 70)
            self.logger.info(
                f"{'TOTAL':<35} "
                f"{self.stats.total_annotations:>8,} "
                f"{self.stats.annotations_kept:>8,} "
                f"{self.stats.annotations_dropped:>8,} "
                f"{self.stats.total_images_downloaded:>6,}"
            )

        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info(f"Output directory: {self.output_dir.absolute()}")
        self.logger.info(f"Filtered annotation files saved with '{FILTERED_SUFFIX}' suffix")
        self.logger.info("=" * 70)

        # Log final stats to WandB
        self._log_to_wandb({
            'final/total_annotations': self.stats.total_annotations,
            'final/annotations_kept': self.stats.annotations_kept,
            'final/annotations_dropped': self.stats.annotations_dropped,
            'final/images_downloaded': self.stats.total_images_downloaded,
            'final/images_failed': self.stats.total_images_failed,
            'final/images_existing': self.stats.total_images_found,
            'final/bytes_downloaded': self.stats.total_bytes_downloaded,
            'final/duration_seconds': self.stats.duration_seconds
        })

        # Save status file
        status_path = self.output_dir / STATUS_FILE
        with open(status_path, 'w') as f:
            json.dump(self.stats.to_dict(), f, indent=2)
        self.logger.info(f"Status saved to: {status_path}")

    def finish(self):
        """Cleanup and finalize."""
        if self.wandb_run:
            wandb.finish()
            self.logger.info("WandB run finished")


def main():
    parser = argparse.ArgumentParser(
        description='Download Incidents Dataset Images with Logging',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--annotations-dir',
        type=str,
        default='incidents-dataset',
        help='Directory containing annotation JSON files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/incidents_images',
        help='Directory to save downloaded images and filtered annotations'
    )
    parser.add_argument(
        '--max-images',
        type=int,
        default=None,
        help='Maximum images to download per file (default: all)'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=8,
        help='Number of parallel download workers'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=30,
        help='Download timeout in seconds'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    parser.add_argument(
        '--no-wandb',
        action='store_true',
        help='Disable WandB logging'
    )
    parser.add_argument(
        '--check-only',
        action='store_true',
        help='Only check annotations, do not download'
    )

    args = parser.parse_args()

    # Initialize downloader
    downloader = IncidentsDownloader(
        annotations_dir=args.annotations_dir,
        output_dir=args.output_dir,
        log_level=args.log_level,
        use_wandb=not args.no_wandb,
        num_workers=args.num_workers,
        timeout=args.timeout
    )

    try:
        # Process all files
        downloader.process_all_files(
            max_images_per_file=args.max_images,
            download=not args.check_only
        )

        # Print final summary
        downloader.print_final_summary()

    except KeyboardInterrupt:
        downloader.logger.warning("Download interrupted by user")
        downloader.print_final_summary()

    except Exception as e:
        downloader.logger.error(f"Error during download: {e}")
        raise

    finally:
        downloader.finish()


if __name__ == "__main__":
    main()

