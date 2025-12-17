#!/usr/bin/env python3
"""
EmberVLM Raspberry Pi Inference Runtime
Single-file deployment for robot fleet reasoning on edge devices.

Usage:
    python pi_inference.py --model model.gguf --image scene.jpg --task "Inspect building"

    # Interactive mode:
    python pi_inference.py --model model.gguf --interactive
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Optional, List, Any
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Core imports
import numpy as np

# Try to import optional dependencies
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Using numpy-only inference.")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("PIL not available. Image loading disabled.")


@dataclass
class RobotInfo:
    """Robot capability information."""
    name: str
    capabilities: List[str]
    limitations: List[str]
    environments: List[str]

    @classmethod
    def from_dict(cls, data: Dict) -> 'RobotInfo':
        return cls(
            name=data['name'],
            capabilities=data.get('capabilities', []),
            limitations=data.get('limitations', []),
            environments=data.get('environments', [])
        )


# Standard robot fleet
ROBOT_FLEET = [
    RobotInfo(
        name="Drone",
        capabilities=["fastest", "aerial navigation", "surveillance", "lightweight transport", "aerial inspection"],
        limitations=["limited payload", "loud", "weather dependent", "battery life"],
        environments=["outdoor", "large indoor spaces", "hard to reach areas"]
    ),
    RobotInfo(
        name="Humanoid",
        capabilities=["manipulation", "walking", "human interaction", "complex tasks", "tool use"],
        limitations=["slow movement", "balance issues", "high power consumption"],
        environments=["indoor", "human environments", "stairs", "complex terrain"]
    ),
    RobotInfo(
        name="Robot with Legs",
        capabilities=["rough terrain navigation", "stability", "load carrying", "inspection"],
        limitations=["limited manipulation", "height restrictions"],
        environments=["outdoor", "uneven terrain", "stairs", "industrial sites", "search and rescue"]
    ),
    RobotInfo(
        name="Robot with Wheels",
        capabilities=["fast movement", "good payload", "stable platform", "efficient"],
        limitations=["flat surfaces only", "limited climbing", "obstacle avoidance"],
        environments=["indoor", "warehouse", "flat outdoor areas", "roads"]
    ),
    RobotInfo(
        name="Underwater Robot",
        capabilities=["underwater navigation", "deep sea exploration", "marine inspection", "underwater manipulation"],
        limitations=["water environments only", "communication limitations", "pressure constraints"],
        environments=["underwater", "marine", "pools", "pipes", "ocean exploration"]
    )
]


class SimpleTokenizer:
    """Simple byte-level tokenizer for edge deployment."""

    def __init__(self, vocab_size: int = 256):
        self.vocab_size = vocab_size

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        return list(text.encode('utf-8'))

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        return bytes(token_ids).decode('utf-8', errors='ignore')


class ImageProcessor:
    """Simple image processor for edge deployment."""

    def __init__(
        self,
        target_size: int = 224,
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225]
    ):
        self.target_size = target_size
        self.mean = np.array(mean).reshape(1, 1, 3)
        self.std = np.array(std).reshape(1, 1, 3)

    def process(self, image_path: str) -> np.ndarray:
        """
        Process image for model input.

        Returns:
            Normalized image array [H, W, 3]
        """
        if not PIL_AVAILABLE:
            raise ImportError("PIL required for image processing")

        # Load image
        image = Image.open(image_path).convert('RGB')

        # Resize
        image = image.resize((self.target_size, self.target_size), Image.BILINEAR)

        # Convert to numpy and normalize
        img_array = np.array(image).astype(np.float32) / 255.0
        img_array = (img_array - self.mean) / self.std

        return img_array


class RuleBasedSelector:
    """
    Rule-based robot selector for fallback when model unavailable.
    Uses keyword matching and heuristics.
    """

    KEYWORDS = {
        "Drone": ["aerial", "fly", "survey", "high", "inspect from above", "bird", "overview", "fast"],
        "Humanoid": ["human", "interact", "door", "button", "tool", "delicate", "manipulation"],
        "Robot with Legs": ["rough", "terrain", "stairs", "rubble", "uneven", "mountain", "climb"],
        "Robot with Wheels": ["flat", "warehouse", "indoor", "road", "fast ground", "patrol"],
        "Underwater Robot": ["water", "underwater", "swim", "dive", "marine", "pool", "pipe"]
    }

    def select(self, task_description: str) -> Dict[str, Any]:
        """Select robot based on task description keywords."""
        task_lower = task_description.lower()

        scores = {}
        for robot, keywords in self.KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in task_lower)
            scores[robot] = score

        # Get top scorer(s)
        max_score = max(scores.values())
        if max_score == 0:
            # Default to Drone for general tasks
            selected = ["Drone"]
        else:
            selected = [r for r, s in scores.items() if s == max_score]

        return {
            "selected_robots": selected,
            "confidence": min(0.9, max_score * 0.2 + 0.3),
            "scores": scores
        }


class EmberVLMPiRuntime:
    """
    EmberVLM inference runtime for Raspberry Pi.

    Supports:
    - GGUF model loading (via llama.cpp bindings if available)
    - Fallback to rule-based selection
    - Minimal memory footprint
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        config_path: Optional[str] = None,
        use_rules_fallback: bool = True
    ):
        self.model_path = model_path
        self.model = None
        self.config = self._load_config(config_path)

        # Initialize components
        self.tokenizer = SimpleTokenizer()
        self.image_processor = ImageProcessor()
        self.rule_selector = RuleBasedSelector() if use_rules_fallback else None

        # Load model if path provided
        if model_path:
            self._load_model(model_path)

        # Performance tracking
        self.inference_times = []

    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration."""
        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                return json.load(f)
        return {
            "max_tokens": 100,
            "temperature": 0.7,
            "context_length": 512
        }

    def _load_model(self, model_path: str):
        """Load model from GGUF file."""
        if not Path(model_path).exists():
            logger.warning(f"Model not found: {model_path}")
            return

        # Try llama-cpp-python
        try:
            from llama_cpp import Llama
            self.model = Llama(
                model_path=model_path,
                n_ctx=self.config.get("context_length", 512),
                n_threads=4,
                verbose=False
            )
            logger.info(f"Loaded model: {model_path}")
        except ImportError:
            logger.warning("llama-cpp-python not available. Using rule-based fallback.")
        except Exception as e:
            logger.warning(f"Could not load model: {e}")

    def select_robot(
        self,
        task_description: str,
        image_path: Optional[str] = None,
        available_robots: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Select optimal robot(s) for a task.

        Args:
            task_description: Description of the task
            image_path: Optional path to scene image
            available_robots: List of available robots (default: all)

        Returns:
            Dictionary with selection results
        """
        start_time = time.time()

        if available_robots is None:
            available_robots = [r.name for r in ROBOT_FLEET]

        # Try model-based inference
        if self.model is not None:
            result = self._model_inference(task_description, image_path, available_robots)
        elif self.rule_selector:
            # Fallback to rule-based
            result = self.rule_selector.select(task_description)
        else:
            # Default selection
            result = {
                "selected_robots": ["Drone"],
                "confidence": 0.5,
                "reasoning": "Default selection (no model or rules)"
            }

        # Add timing
        inference_time = (time.time() - start_time) * 1000
        result["inference_time_ms"] = inference_time
        self.inference_times.append(inference_time)

        # Generate action plan
        result["action_plan"] = self._generate_action_plan(
            task_description,
            result["selected_robots"]
        )

        return result

    def _model_inference(
        self,
        task: str,
        image_path: Optional[str],
        robots: List[str]
    ) -> Dict[str, Any]:
        """Run model inference."""
        # Build prompt
        robot_str = ", ".join(robots)
        prompt = f"""Task: {task}
Available robots: {robot_str}

Select the best robot(s) and explain your reasoning.

<reasoning>"""

        # Generate
        try:
            output = self.model(
                prompt,
                max_tokens=self.config.get("max_tokens", 100),
                temperature=self.config.get("temperature", 0.7),
                stop=["</answer>"]
            )

            response = output["choices"][0]["text"]

            # Parse response
            return self._parse_response(response, robots)

        except Exception as e:
            logger.warning(f"Model inference failed: {e}")
            if self.rule_selector:
                return self.rule_selector.select(task)
            return {"selected_robots": robots[:1], "confidence": 0.3}

    def _parse_response(self, response: str, available_robots: List[str]) -> Dict[str, Any]:
        """Parse model response to extract robot selection."""
        selected = []

        for robot in available_robots:
            if robot.lower() in response.lower():
                selected.append(robot)

        if not selected:
            selected = [available_robots[0]]

        # Extract reasoning
        reasoning = response.split("</reasoning>")[0] if "</reasoning>" in response else response

        # Extract answer if present
        answer = ""
        if "<answer>" in response:
            answer = response.split("<answer>")[-1].split("</answer>")[0]

        return {
            "selected_robots": selected,
            "confidence": 0.8 if len(selected) > 0 else 0.5,
            "reasoning": reasoning.strip(),
            "answer": answer.strip()
        }

    def _generate_action_plan(
        self,
        task: str,
        robots: List[str]
    ) -> List[str]:
        """Generate simple action plan."""
        plan = []

        if len(robots) == 1:
            robot = robots[0]
            plan = [
                f"1. Deploy {robot} to task location",
                f"2. {robot} performs initial assessment",
                f"3. {robot} executes primary task: {task[:50]}...",
                "4. Monitor progress and adjust as needed",
                "5. Complete task and return to base"
            ]
        else:
            plan = [
                f"1. Deploy primary robot ({robots[0]}) to task location",
                f"2. {robots[0]} performs initial assessment",
                f"3. Coordinate support from: {', '.join(robots[1:])}",
                f"4. Execute task: {task[:50]}...",
                "5. Monitor multi-robot coordination",
                "6. Complete task and return all robots"
            ]

        return plan

    def plan_incident_response(
        self,
        incident_type: str,
        description: str,
        image_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Plan response to an incident.

        Args:
            incident_type: Type of incident (fire, flood, etc.)
            description: Incident description
            image_path: Optional image of incident

        Returns:
            Response plan with robot assignments
        """
        # Map incident types to recommended robots
        incident_robots = {
            "fire": ["Drone", "Robot with Wheels"],
            "flood": ["Underwater Robot", "Drone"],
            "earthquake": ["Robot with Legs", "Drone"],
            "hazmat": ["Robot with Wheels", "Drone"],
            "traffic": ["Drone", "Robot with Wheels"],
            "security": ["Drone", "Robot with Legs"]
        }

        # Get recommendations
        incident_lower = incident_type.lower()
        recommended = None
        for key, robots in incident_robots.items():
            if key in incident_lower:
                recommended = robots
                break

        if recommended is None:
            recommended = ["Drone"]  # Default

        # Generate response plan
        response_actions = [
            f"1. ALERT: {incident_type} incident detected",
            f"2. Deploy primary responder: {recommended[0]}",
            "3. Establish communication relay",
            f"4. {recommended[0]} conducts initial assessment",
            f"5. Deploy support: {recommended[1] if len(recommended) > 1 else 'standby'}",
            "6. Coordinate with emergency services",
            "7. Monitor situation continuously",
            "8. Document and report findings"
        ]

        return {
            "incident_type": incident_type,
            "severity": "high" if any(w in description.lower() for w in ["emergency", "critical", "urgent"]) else "moderate",
            "primary_robot": recommended[0],
            "support_robots": recommended[1:],
            "action_plan": response_actions,
            "reasoning": f"Selected {recommended[0]} as primary due to {incident_type} incident characteristics"
        }

    def get_performance_stats(self) -> Dict[str, float]:
        """Get inference performance statistics."""
        if not self.inference_times:
            return {"avg_ms": 0, "min_ms": 0, "max_ms": 0}

        return {
            "avg_ms": sum(self.inference_times) / len(self.inference_times),
            "min_ms": min(self.inference_times),
            "max_ms": max(self.inference_times),
            "total_inferences": len(self.inference_times)
        }

    def interactive_mode(self):
        """Run interactive command-line interface."""
        print("\n" + "=" * 50)
        print("EmberVLM Robot Fleet Manager")
        print("=" * 50)
        print("\nCommands:")
        print("  robot <task>     - Select robot for task")
        print("  incident <type>  - Plan incident response")
        print("  stats            - Show performance stats")
        print("  quit             - Exit")
        print()

        while True:
            try:
                user_input = input("EmberVLM> ").strip()

                if not user_input:
                    continue

                if user_input.lower() == "quit":
                    print("Goodbye!")
                    break

                if user_input.lower() == "stats":
                    stats = self.get_performance_stats()
                    print(f"\nPerformance Statistics:")
                    print(f"  Average inference: {stats['avg_ms']:.2f} ms")
                    print(f"  Min inference: {stats['min_ms']:.2f} ms")
                    print(f"  Max inference: {stats['max_ms']:.2f} ms")
                    print(f"  Total inferences: {stats['total_inferences']}")
                    continue

                if user_input.lower().startswith("robot "):
                    task = user_input[6:].strip()
                    result = self.select_robot(task)

                    print(f"\n📋 Task: {task}")
                    print(f"🤖 Selected: {', '.join(result['selected_robots'])}")
                    print(f"📊 Confidence: {result['confidence']*100:.1f}%")
                    print(f"⏱️  Time: {result['inference_time_ms']:.1f} ms")
                    print(f"\n📝 Action Plan:")
                    for step in result['action_plan']:
                        print(f"   {step}")
                    print()
                    continue

                if user_input.lower().startswith("incident "):
                    incident = user_input[9:].strip()
                    result = self.plan_incident_response(incident, incident)

                    print(f"\n🚨 Incident: {result['incident_type']}")
                    print(f"⚠️  Severity: {result['severity']}")
                    print(f"🤖 Primary: {result['primary_robot']}")
                    print(f"🔧 Support: {', '.join(result['support_robots']) or 'None'}")
                    print(f"\n📝 Response Plan:")
                    for step in result['action_plan']:
                        print(f"   {step}")
                    print()
                    continue

                print("Unknown command. Type 'help' for available commands.")

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='EmberVLM Raspberry Pi Inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Select robot for a task
    python pi_inference.py --task "Inspect building exterior"
    
    # Plan incident response
    python pi_inference.py --incident "Fire detected in warehouse"
    
    # Interactive mode
    python pi_inference.py --interactive
    
    # With model file
    python pi_inference.py --model embervlm.gguf --task "Underwater inspection"
        """
    )

    parser.add_argument('--model', type=str, help='Path to GGUF model file')
    parser.add_argument('--config', type=str, help='Path to config JSON')
    parser.add_argument('--task', type=str, help='Task description for robot selection')
    parser.add_argument('--incident', type=str, help='Incident type for response planning')
    parser.add_argument('--image', type=str, help='Path to scene image')
    parser.add_argument('--interactive', action='store_true', help='Run interactive mode')
    parser.add_argument('--output', type=str, help='Output JSON file')

    args = parser.parse_args()

    # Initialize runtime
    runtime = EmberVLMPiRuntime(
        model_path=args.model,
        config_path=args.config
    )

    # Handle modes
    if args.interactive:
        runtime.interactive_mode()
        return

    result = None

    if args.task:
        result = runtime.select_robot(
            task_description=args.task,
            image_path=args.image
        )

        print(f"\nTask: {args.task}")
        print(f"Selected robots: {', '.join(result['selected_robots'])}")
        print(f"Confidence: {result['confidence']*100:.1f}%")
        print(f"Inference time: {result['inference_time_ms']:.1f} ms")
        print(f"\nAction plan:")
        for step in result['action_plan']:
            print(f"  {step}")

    elif args.incident:
        result = runtime.plan_incident_response(
            incident_type=args.incident,
            description=args.incident,
            image_path=args.image
        )

        print(f"\nIncident: {result['incident_type']}")
        print(f"Severity: {result['severity']}")
        print(f"Primary robot: {result['primary_robot']}")
        print(f"Support: {', '.join(result['support_robots'])}")
        print(f"\nResponse plan:")
        for step in result['action_plan']:
            print(f"  {step}")

    else:
        parser.print_help()
        return

    # Save output if requested
    if args.output and result:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nOutput saved to: {args.output}")


if __name__ == "__main__":
    main()

