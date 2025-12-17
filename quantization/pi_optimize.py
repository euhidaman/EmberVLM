"""
EmberVLM Raspberry Pi Optimization
Optimize model for deployment on Raspberry Pi Zero.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Optional, Any, Tuple
import json
import time

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class PiOptimizer:
    """
    Optimize EmberVLM for Raspberry Pi Zero deployment.

    Targets:
    - Memory: <400MB RAM
    - Size: <100MB on disk
    - Latency: <500ms per inference
    - Power: <2W
    """

    def __init__(
        self,
        target_memory_mb: float = 400,
        target_size_mb: float = 100,
        target_latency_ms: float = 500
    ):
        self.target_memory_mb = target_memory_mb
        self.target_size_mb = target_size_mb
        self.target_latency_ms = target_latency_ms

    def optimize_model(
        self,
        model: nn.Module,
        output_dir: str,
        apply_pruning: bool = True,
        apply_quantization: bool = True,
        compile_for_arm: bool = True
    ) -> Dict[str, Any]:
        """
        Apply all optimizations for Pi deployment.

        Args:
            model: PyTorch model
            output_dir: Output directory
            apply_pruning: Apply weight pruning
            apply_quantization: Apply quantization
            compile_for_arm: Prepare for ARM compilation

        Returns:
            Optimization results and metrics
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results = {}

        # 1. Model analysis
        original_size = self._get_model_size(model)
        results['original_size_mb'] = original_size / (1024 * 1024)
        logger.info(f"Original model size: {results['original_size_mb']:.2f} MB")

        # 2. Pruning
        if apply_pruning:
            model = self._apply_pruning(model, sparsity=0.4)
            pruned_size = self._get_model_size(model)
            results['pruned_size_mb'] = pruned_size / (1024 * 1024)
            logger.info(f"After pruning: {results['pruned_size_mb']:.2f} MB")

        # 3. Quantization
        if apply_quantization:
            quantized_model = self._apply_quantization(model)
            quantized_size = self._get_model_size(quantized_model)
            results['quantized_size_mb'] = quantized_size / (1024 * 1024)
            logger.info(f"After quantization: {results['quantized_size_mb']:.2f} MB")
            model = quantized_model

        # 4. Save optimized model
        model_path = output_path / "embervlm_pi.pt"
        torch.save(model.state_dict(), model_path)

        # 5. Generate deployment config
        config = self._generate_pi_config(model, results)
        config_path = output_path / "pi_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        # 6. Generate inference code
        inference_code = self._generate_inference_code()
        code_path = output_path / "pi_inference.py"
        with open(code_path, 'w') as f:
            f.write(inference_code)

        results['model_path'] = str(model_path)
        results['config_path'] = str(config_path)
        results['inference_code_path'] = str(code_path)

        # Check against targets
        final_size = results.get('quantized_size_mb', results.get('pruned_size_mb', results['original_size_mb']))
        results['meets_size_target'] = final_size < self.target_size_mb

        logger.info(f"Optimization complete. Final size: {final_size:.2f} MB")

        return results

    def _get_model_size(self, model: nn.Module) -> int:
        """Get model size in bytes."""
        size = 0
        for param in model.parameters():
            size += param.numel() * param.element_size()
        for buffer in model.buffers():
            size += buffer.numel() * buffer.element_size()
        return size

    def _apply_pruning(self, model: nn.Module, sparsity: float = 0.4) -> nn.Module:
        """
        Apply magnitude-based weight pruning.

        Args:
            model: Model to prune
            sparsity: Target sparsity (0.4 = remove 40% of weights)
        """
        try:
            import torch.nn.utils.prune as prune

            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    prune.l1_unstructured(module, name='weight', amount=sparsity)
                    prune.remove(module, 'weight')
                elif isinstance(module, nn.Conv2d):
                    prune.l1_unstructured(module, name='weight', amount=sparsity)
                    prune.remove(module, 'weight')

            logger.info(f"Applied {sparsity*100:.0f}% pruning")

        except Exception as e:
            logger.warning(f"Pruning failed: {e}")

        return model

    def _apply_quantization(self, model: nn.Module) -> nn.Module:
        """
        Apply dynamic quantization for CPU inference.
        """
        try:
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {nn.Linear, nn.Embedding},
                dtype=torch.qint8
            )
            logger.info("Applied dynamic quantization (INT8)")
            return quantized_model

        except Exception as e:
            logger.warning(f"Quantization failed: {e}")
            return model

    def _generate_pi_config(self, model: nn.Module, results: Dict) -> Dict:
        """Generate configuration for Pi deployment."""
        return {
            'model_name': 'EmberVLM-Pi',
            'optimization': {
                'pruning': True,
                'quantization': 'INT8',
                'sparsity': 0.4
            },
            'memory': {
                'target_mb': self.target_memory_mb,
                'model_size_mb': results.get('quantized_size_mb', 0)
            },
            'inference': {
                'max_batch_size': 1,
                'max_tokens': 100,
                'context_length': 512,
                'kv_cache_size': 128
            },
            'hardware': {
                'target': 'Raspberry Pi Zero',
                'cpu': 'ARM1176JZF-S',
                'ram_mb': 512,
                'recommended_swap_mb': 1024
            },
            'image_processing': {
                'input_size': 224,
                'normalize_mean': [0.485, 0.456, 0.406],
                'normalize_std': [0.229, 0.224, 0.225]
            }
        }

    def _generate_inference_code(self) -> str:
        """Generate single-file inference code for Pi."""
        return '''#!/usr/bin/env python3
"""
EmberVLM Raspberry Pi Inference
Single-file deployment for robot fleet reasoning.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Optional, List, Tuple

import torch
import torch.nn.functional as F
import numpy as np

# Try to import PIL for image processing
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class EmberVLMPi:
    """
    EmberVLM inference runtime for Raspberry Pi.
    
    Features:
    - Single-image inference
    - Robot fleet selection
    - Incident response planning
    - Minimal memory footprint
    """
    
    def __init__(
        self,
        model_path: str,
        config_path: Optional[str] = None,
        device: str = "cpu"
    ):
        self.device = torch.device(device)
        
        # Load config
        self.config = self._load_config(config_path)
        
        # Load model
        print(f"Loading model from {model_path}...")
        start_time = time.time()
        self.model = self._load_model(model_path)
        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.2f}s")
        
        # Initialize tokenizer
        self.tokenizer = self._init_tokenizer()
        
        # KV cache for efficient generation
        self.kv_cache_size = self.config.get('inference', {}).get('kv_cache_size', 128)
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load deployment config."""
        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                return json.load(f)
        return {
            'inference': {
                'max_tokens': 100,
                'context_length': 512
            },
            'image_processing': {
                'input_size': 224,
                'normalize_mean': [0.485, 0.456, 0.406],
                'normalize_std': [0.229, 0.224, 0.225]
            }
        }
        
    def _load_model(self, model_path: str):
        """Load quantized model."""
        # This would load the actual EmberVLM model
        # For deployment, we use the quantized weights
        state_dict = torch.load(model_path, map_location=self.device)
        
        # Placeholder - in real deployment, reconstruct model architecture
        # and load weights
        return state_dict
        
    def _init_tokenizer(self):
        """Initialize simple tokenizer."""
        # In real deployment, use proper tokenizer
        # For Pi, might use simple byte-level encoding
        return None
        
    def preprocess_image(
        self,
        image_path: str
    ) -> torch.Tensor:
        """
        Preprocess image for model input.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed image tensor [1, 3, 224, 224]
        """
        if not PIL_AVAILABLE:
            raise ImportError("PIL required for image processing")
            
        # Load and resize
        img_size = self.config['image_processing']['input_size']
        image = Image.open(image_path).convert('RGB')
        image = image.resize((img_size, img_size), Image.BILINEAR)
        
        # Convert to tensor
        img_array = np.array(image).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
        
        # Normalize
        mean = torch.tensor(self.config['image_processing']['normalize_mean']).view(3, 1, 1)
        std = torch.tensor(self.config['image_processing']['normalize_std']).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std
        
        return img_tensor.unsqueeze(0)
        
    def select_robot(
        self,
        image_path: str,
        task_description: str,
        robot_fleet: Optional[List[str]] = None
    ) -> Dict[str, any]:
        """
        Select optimal robot for a task.
        
        Args:
            image_path: Path to scene image
            task_description: Description of the task
            robot_fleet: Available robots (default: standard fleet)
            
        Returns:
            Dict with selected robot, confidence, and reasoning
        """
        start_time = time.time()
        
        if robot_fleet is None:
            robot_fleet = ["Drone", "Humanoid", "Robot with Legs", 
                         "Robot with Wheels", "Underwater Robot"]
                         
        # Preprocess image
        image = self.preprocess_image(image_path)
        
        # Create prompt
        prompt = self._create_robot_selection_prompt(task_description, robot_fleet)
        
        # Generate response
        response = self._generate(image, prompt)
        
        # Parse response
        result = self._parse_robot_selection(response, robot_fleet)
        
        inference_time = time.time() - start_time
        result['inference_time_ms'] = inference_time * 1000
        
        return result
        
    def plan_incident_response(
        self,
        image_path: str,
        incident_type: str,
        description: str
    ) -> Dict[str, any]:
        """
        Plan response to an incident.
        
        Args:
            image_path: Path to incident image
            incident_type: Type of incident
            description: Incident description
            
        Returns:
            Dict with action plan and robot assignments
        """
        start_time = time.time()
        
        # Preprocess image
        image = self.preprocess_image(image_path)
        
        # Create prompt
        prompt = self._create_incident_prompt(incident_type, description)
        
        # Generate response
        response = self._generate(image, prompt)
        
        # Parse response
        result = self._parse_incident_response(response)
        
        inference_time = time.time() - start_time
        result['inference_time_ms'] = inference_time * 1000
        
        return result
        
    def _create_robot_selection_prompt(
        self,
        task: str,
        robots: List[str]
    ) -> str:
        """Create prompt for robot selection."""
        robot_str = ", ".join(robots)
        return f"""Select the best robot for: {task}
Available robots: {robot_str}
Provide your reasoning and selection."""

    def _create_incident_prompt(
        self,
        incident_type: str,
        description: str
    ) -> str:
        """Create prompt for incident response."""
        return f"""Incident: {incident_type}
Description: {description}
Provide step-by-step response plan."""

    def _generate(
        self,
        image: torch.Tensor,
        prompt: str,
        max_tokens: int = 100
    ) -> str:
        """Generate text response."""
        # Placeholder - actual implementation would run model inference
        # This is where llama.cpp or direct PyTorch inference would happen
        return "[Generated response placeholder]"
        
    def _parse_robot_selection(
        self,
        response: str,
        available_robots: List[str]
    ) -> Dict:
        """Parse robot selection from response."""
        # Simple parsing - look for robot names in response
        selected = []
        for robot in available_robots:
            if robot.lower() in response.lower():
                selected.append(robot)
                
        return {
            'selected_robots': selected or [available_robots[0]],
            'confidence': 0.8,
            'reasoning': response,
            'action_plan': ['Analyze task', 'Deploy robot', 'Monitor progress']
        }
        
    def _parse_incident_response(self, response: str) -> Dict:
        """Parse incident response plan."""
        return {
            'action_plan': [
                'Assess situation',
                'Deploy primary responder',
                'Coordinate support',
                'Monitor and adjust'
            ],
            'primary_robot': 'Drone',
            'support_robots': ['Robot with Legs'],
            'reasoning': response
        }


def main():
    """Main entry point for Pi inference."""
    import argparse
    
    parser = argparse.ArgumentParser(description='EmberVLM Pi Inference')
    parser.add_argument('--model', required=True, help='Path to model file')
    parser.add_argument('--config', help='Path to config file')
    parser.add_argument('--image', required=True, help='Path to image')
    parser.add_argument('--task', required=True, help='Task description')
    parser.add_argument('--mode', choices=['robot', 'incident'], default='robot')
    
    args = parser.parse_args()
    
    # Initialize model
    model = EmberVLMPi(
        model_path=args.model,
        config_path=args.config
    )
    
    # Run inference
    if args.mode == 'robot':
        result = model.select_robot(args.image, args.task)
    else:
        result = model.plan_incident_response(args.image, 'general', args.task)
        
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
'''


def create_pi_optimizer(**kwargs) -> PiOptimizer:
    """Factory function for Pi optimizer."""
    return PiOptimizer(**kwargs)


if __name__ == "__main__":
    # Test Pi optimization
    print("Testing Pi Optimization...")

    optimizer = PiOptimizer(
        target_memory_mb=400,
        target_size_mb=100,
        target_latency_ms=500
    )

    # Create simple test model
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(1000, 256)
            self.linear1 = nn.Linear(256, 512)
            self.linear2 = nn.Linear(512, 256)
            self.norm = nn.LayerNorm(256)

        def forward(self, x):
            x = self.embed(x)
            x = F.relu(self.linear1(x))
            x = self.linear2(x)
            return self.norm(x)

    model = TestModel()

    # Test optimization
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        results = optimizer.optimize_model(
            model=model,
            output_dir=tmp_dir,
            apply_pruning=True,
            apply_quantization=True
        )

        print(f"Optimization results: {json.dumps(results, indent=2)}")

    print("Pi optimization tests complete!")

