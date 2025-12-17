"""
EmberVLM Robot Selection Evaluation
Metrics and evaluation for robot fleet selection task.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Any
from dataclasses import dataclass
import json
import random

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class RobotEvalMetrics:
    """Metrics for robot selection evaluation."""
    accuracy: float
    f1_score: float
    precision: float
    recall: float
    per_robot_accuracy: Dict[str, float]
    confusion_matrix: Dict[str, Dict[str, int]]


class RobotSelectionEvaluator:
    """
    Evaluate model performance on robot selection task.
    """

    ROBOT_TYPES = [
        "Drone",
        "Humanoid",
        "Robot with Legs",
        "Robot with Wheels",
        "Underwater Robot"
    ]

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        device: str = "cuda"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def evaluate(
        self,
        eval_samples: List[Dict],
        max_samples: Optional[int] = None
    ) -> RobotEvalMetrics:
        """
        Evaluate on robot selection samples.

        Args:
            eval_samples: List of evaluation samples with 'task', 'ground_truth', 'prompt'
            max_samples: Maximum samples to evaluate

        Returns:
            RobotEvalMetrics with all metrics
        """
        if max_samples:
            eval_samples = eval_samples[:max_samples]

        self.model.eval()

        predictions = []
        ground_truths = []

        for sample in eval_samples:
            # Generate prediction
            pred_robots = self._predict_robots(sample['prompt'])
            gt_robots = sample['ground_truth']

            predictions.append(set(pred_robots))
            ground_truths.append(set(gt_robots))

        # Compute metrics
        metrics = self._compute_metrics(predictions, ground_truths)

        return metrics

    def _predict_robots(self, prompt: str) -> List[str]:
        """Generate robot selection prediction."""
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors='pt',
            max_length=512,
            truncation=True
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=100,
                temperature=0.7,
                do_sample=False  # Greedy for evaluation
            )

        # Decode
        if isinstance(outputs, dict):
            generated_ids = outputs.get('generated_ids', outputs.get('sequences'))
        else:
            generated_ids = outputs

        response = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        # Extract robot names from response
        predicted_robots = []
        for robot in self.ROBOT_TYPES:
            if robot.lower() in response.lower():
                predicted_robots.append(robot)

        return predicted_robots if predicted_robots else [self.ROBOT_TYPES[0]]

    def _compute_metrics(
        self,
        predictions: List[set],
        ground_truths: List[set]
    ) -> RobotEvalMetrics:
        """Compute evaluation metrics."""
        # Exact match accuracy
        exact_matches = sum(1 for p, g in zip(predictions, ground_truths) if p == g)
        accuracy = exact_matches / len(predictions) if predictions else 0

        # F1, precision, recall (multi-label)
        total_tp = 0
        total_fp = 0
        total_fn = 0

        for pred, gt in zip(predictions, ground_truths):
            tp = len(pred & gt)
            fp = len(pred - gt)
            fn = len(gt - pred)

            total_tp += tp
            total_fp += fp
            total_fn += fn

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # Per-robot accuracy
        per_robot_acc = {}
        for robot in self.ROBOT_TYPES:
            correct = 0
            total = 0
            for pred, gt in zip(predictions, ground_truths):
                if robot in gt:
                    total += 1
                    if robot in pred:
                        correct += 1
            per_robot_acc[robot] = correct / total if total > 0 else 0

        # Confusion matrix
        confusion = {r1: {r2: 0 for r2 in self.ROBOT_TYPES} for r1 in self.ROBOT_TYPES}
        for pred, gt in zip(predictions, ground_truths):
            for gt_robot in gt:
                for pred_robot in pred:
                    confusion[gt_robot][pred_robot] += 1

        return RobotEvalMetrics(
            accuracy=accuracy,
            f1_score=f1,
            precision=precision,
            recall=recall,
            per_robot_accuracy=per_robot_acc,
            confusion_matrix=confusion
        )

    def generate_report(self, metrics: RobotEvalMetrics) -> str:
        """Generate evaluation report."""
        report = []
        report.append("=" * 50)
        report.append("Robot Selection Evaluation Report")
        report.append("=" * 50)
        report.append(f"\nOverall Metrics:")
        report.append(f"  Accuracy:  {metrics.accuracy*100:.2f}%")
        report.append(f"  F1 Score:  {metrics.f1_score*100:.2f}%")
        report.append(f"  Precision: {metrics.precision*100:.2f}%")
        report.append(f"  Recall:    {metrics.recall*100:.2f}%")

        report.append(f"\nPer-Robot Accuracy:")
        for robot, acc in metrics.per_robot_accuracy.items():
            report.append(f"  {robot}: {acc*100:.2f}%")

        return "\n".join(report)


def evaluate_robot_selection(
    model: Any,
    tokenizer: Any,
    eval_samples: List[Dict],
    output_path: Optional[str] = None
) -> RobotEvalMetrics:
    """
    Run robot selection evaluation.

    Args:
        model: EmberVLM model
        tokenizer: Tokenizer
        eval_samples: Evaluation samples
        output_path: Path to save report

    Returns:
        Evaluation metrics
    """
    evaluator = RobotSelectionEvaluator(model, tokenizer)
    metrics = evaluator.evaluate(eval_samples)

    report = evaluator.generate_report(metrics)
    logger.info(report)

    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
            f.write("\n\n")
            f.write(json.dumps({
                'accuracy': metrics.accuracy,
                'f1_score': metrics.f1_score,
                'precision': metrics.precision,
                'recall': metrics.recall,
                'per_robot_accuracy': metrics.per_robot_accuracy
            }, indent=2))

    return metrics


if __name__ == "__main__":
    # Test robot evaluation
    print("Testing Robot Selection Evaluator...")

    # Mock predictions and ground truths
    predictions = [
        {"Drone"},
        {"Drone", "Robot with Legs"},
        {"Humanoid"},
        {"Robot with Wheels", "Drone"}
    ]

    ground_truths = [
        {"Drone"},
        {"Drone", "Robot with Legs"},
        {"Robot with Legs"},  # Wrong prediction
        {"Robot with Wheels"}  # Partial match
    ]

    # Create mock evaluator
    evaluator = RobotSelectionEvaluator.__new__(RobotSelectionEvaluator)
    evaluator.ROBOT_TYPES = RobotSelectionEvaluator.ROBOT_TYPES

    metrics = evaluator._compute_metrics(predictions, ground_truths)

    print(f"Accuracy: {metrics.accuracy*100:.2f}%")
    print(f"F1 Score: {metrics.f1_score*100:.2f}%")
    print(f"Precision: {metrics.precision*100:.2f}%")
    print(f"Recall: {metrics.recall*100:.2f}%")

    print("\nRobot evaluation tests complete!")

