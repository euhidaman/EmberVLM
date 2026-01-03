#!/usr/bin/env python
"""
EmberVLM Evaluation Script using VLMEvalKit

Evaluates EmberVLM checkpoints on standard VLM benchmarks.

Usage:
    python evaluate_vlmevalkit.py --model_path outputs/stage4 --stage 4
    python evaluate_vlmevalkit.py --model_path outputs/stage2 --benchmarks MMBench_DEV_EN_V11 MME
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_vlmevalkit():
    """Check if VLMEvalKit is available."""
    try:
        import vlmeval
        return True
    except ImportError:
        logger.error(
            "VLMEvalKit not found. Install with:\n"
            "  pip install vlmeval\n"
            "Or clone from: https://github.com/open-compass/VLMEvalKit"
        )
        return False


def run_evaluation(
    model_path: str,
    benchmarks: List[str],
    output_dir: str,
    stage: int = 4,
    batch_size: int = 1,
    num_workers: int = 4,
) -> Dict[str, Any]:
    """
    Run VLMEvalKit evaluation on EmberVLM.

    Args:
        model_path: Path to EmberVLM checkpoint
        benchmarks: List of benchmark names
        output_dir: Output directory for results
        stage: Training stage number
        batch_size: Evaluation batch size
        num_workers: Number of data workers

    Returns:
        Dictionary of benchmark results
    """
    from embervlm.evaluation.vlmevalkit_adapter import EmberVLM_VLMEval

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize model
    logger.info(f"Loading EmberVLM from {model_path}")
    model = EmberVLM_VLMEval(model_path=model_path)

    results = {}

    for benchmark in benchmarks:
        logger.info(f"Evaluating on {benchmark}...")

        try:
            # Import VLMEvalKit components
            from vlmeval.dataset import build_dataset
            from vlmeval.inference import infer_data_job
            from vlmeval.smp import dump, load

            # Build dataset
            dataset = build_dataset(benchmark)

            if dataset is None:
                logger.warning(f"Could not build dataset for {benchmark}, skipping")
                continue

            # Run inference
            pred_file = output_path / f"{benchmark}_predictions.xlsx"

            infer_data_job(
                model=model,
                dataset=dataset,
                dataset_name=benchmark,
                pred_file=str(pred_file),
                verbose=True,
            )

            # Evaluate
            if hasattr(dataset, 'evaluate'):
                score = dataset.evaluate(str(pred_file))
                results[benchmark] = score
                logger.info(f"{benchmark} Score: {score}")
            else:
                results[benchmark] = {'predictions_saved': str(pred_file)}
                logger.info(f"{benchmark} predictions saved to {pred_file}")

        except Exception as e:
            logger.error(f"Failed to evaluate {benchmark}: {e}")
            results[benchmark] = {'error': str(e)}

    # Save results
    results_file = output_path / f"evaluation_results_stage{stage}.json"
    with open(results_file, 'w') as f:
        json.dump({
            'model_path': model_path,
            'stage': stage,
            'timestamp': datetime.now().isoformat(),
            'benchmarks': benchmarks,
            'results': results,
        }, f, indent=2)

    logger.info(f"Results saved to {results_file}")

    return results


def run_quick_evaluation(
    model_path: str,
    output_dir: str,
    stage: int = 4,
) -> Dict[str, Any]:
    """
    Run quick internal evaluation (no VLMEvalKit dependency).

    Evaluates on internal validation sets and robot selection accuracy.

    Args:
        model_path: Path to EmberVLM checkpoint
        output_dir: Output directory
        stage: Training stage

    Returns:
        Evaluation results
    """
    import torch
    from embervlm.models import EmberVLM, EmberVLMConfig
    from embervlm.evaluation.metrics import (
        compute_robot_selection_metrics,
        compute_confidence_calibration,
    )
    from transformers import AutoTokenizer

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load model
    logger.info(f"Loading EmberVLM from {model_path}")

    config_path = Path(model_path) / 'config.json'
    if config_path.exists():
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        config = EmberVLMConfig.from_dict(config_dict)
    else:
        config = EmberVLMConfig()

    model = EmberVLM(config)

    # Load weights
    weights_path = Path(model_path) / 'pytorch_model.bin'
    if weights_path.exists():
        state_dict = torch.load(weights_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # Load tokenizer
    tokenizer_path = Path(model_path) / 'tokenizer'
    if tokenizer_path.exists():
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained('gpt2')

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = {
        'model_path': model_path,
        'stage': stage,
        'timestamp': datetime.now().isoformat(),
    }

    # Robot selection evaluation (Stage 3+)
    if stage >= 3:
        logger.info("Evaluating robot selection...")

        try:
            from embervlm.data.robot_loader import get_robot_selection_dataloader

            robot_dir = Path(__file__).parent.parent / 'robot-selection-dataset'
            if robot_dir.exists():
                val_loader = get_robot_selection_dataloader(
                    data_dir=str(robot_dir),
                    tokenizer=tokenizer,
                    batch_size=16,
                    split='val',
                )

                all_preds = []
                all_targets = []
                all_confidences = []

                with torch.no_grad():
                    for batch in val_loader:
                        pixel_values = batch['pixel_values'].to(device)
                        input_ids = batch['input_ids'].to(device)
                        attention_mask = batch['attention_mask'].to(device)
                        robot_targets = batch['robot_target'].to(device)

                        outputs = model(
                            input_ids=input_ids,
                            pixel_values=pixel_values,
                            attention_mask=attention_mask,
                            robot_targets=robot_targets,
                        )

                        if 'robot_logits' in outputs:
                            robot_probs = torch.softmax(outputs['robot_logits'], dim=-1)
                            robot_preds = robot_probs.argmax(dim=-1)
                            confidences = robot_probs.max(dim=-1).values

                            all_preds.extend(robot_preds.cpu().tolist())
                            all_targets.extend(robot_targets.cpu().tolist())
                            all_confidences.extend(confidences.cpu().tolist())

                if all_preds:
                    robot_metrics = compute_robot_selection_metrics(
                        all_preds, all_targets,
                        robot_names=["Drone", "Underwater", "Humanoid", "Wheeled", "Legged"]
                    )
                    results['robot_selection'] = robot_metrics

                    calibration = compute_confidence_calibration(
                        all_preds, all_targets, all_confidences
                    )
                    results['calibration'] = {
                        'ece': calibration['ece'],
                        'mce': calibration['mce'],
                    }

                    logger.info(f"Robot Selection Accuracy: {robot_metrics['accuracy']:.4f}")
                    logger.info(f"Robot Selection F1: {robot_metrics['macro_f1']:.4f}")
                    logger.info(f"ECE: {calibration['ece']:.4f}")

        except Exception as e:
            logger.warning(f"Robot selection evaluation failed: {e}")

    # Save results
    results_file = output_path / f"quick_eval_stage{stage}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Quick evaluation results saved to {results_file}")

    return results


def log_results_to_wandb(
    results: Dict[str, Any],
    stage: int,
    run_name: str = None,
):
    """Log evaluation results to W&B."""
    try:
        import wandb

        if run_name is None:
            run_name = f"embervlm_eval_stage{stage}"

        wandb.init(
            project="embervlm-evaluation",
            name=run_name,
            config={'stage': stage},
        )

        # Log benchmark scores
        for benchmark, score in results.get('results', {}).items():
            if isinstance(score, dict):
                for metric, value in score.items():
                    if isinstance(value, (int, float)):
                        wandb.log({f"{benchmark}/{metric}": value})
            elif isinstance(score, (int, float)):
                wandb.log({f"{benchmark}/score": score})

        # Log robot selection metrics
        if 'robot_selection' in results:
            for metric, value in results['robot_selection'].items():
                if isinstance(value, (int, float)):
                    wandb.log({f"robot_selection/{metric}": value})

        wandb.finish()
        logger.info("Results logged to W&B")

    except Exception as e:
        logger.warning(f"Failed to log to W&B: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate EmberVLM on VLM benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '--model_path', type=str, required=True,
        help='Path to EmberVLM checkpoint directory'
    )
    parser.add_argument(
        '--stage', type=int, default=4,
        help='Training stage (1-4)'
    )
    parser.add_argument(
        '--benchmarks', type=str, nargs='+', default=None,
        help='Benchmarks to evaluate on (default: stage-appropriate)'
    )
    parser.add_argument(
        '--output_dir', type=str, default='./eval_outputs',
        help='Output directory for results'
    )
    parser.add_argument(
        '--quick', action='store_true',
        help='Run quick internal evaluation (no VLMEvalKit)'
    )
    parser.add_argument(
        '--log_wandb', action='store_true',
        help='Log results to W&B'
    )
    parser.add_argument(
        '--batch_size', type=int, default=1,
        help='Evaluation batch size'
    )

    args = parser.parse_args()

    # Determine benchmarks
    if args.benchmarks is None:
        from embervlm.evaluation.vlmevalkit_adapter import get_benchmarks_for_stage
        args.benchmarks = get_benchmarks_for_stage(args.stage)

    logger.info(f"Evaluating Stage {args.stage} model on: {args.benchmarks}")

    if args.quick:
        results = run_quick_evaluation(
            model_path=args.model_path,
            output_dir=args.output_dir,
            stage=args.stage,
        )
    else:
        if not check_vlmevalkit():
            logger.info("Falling back to quick evaluation...")
            results = run_quick_evaluation(
                model_path=args.model_path,
                output_dir=args.output_dir,
                stage=args.stage,
            )
        else:
            results = run_evaluation(
                model_path=args.model_path,
                benchmarks=args.benchmarks,
                output_dir=args.output_dir,
                stage=args.stage,
                batch_size=args.batch_size,
            )

    if args.log_wandb:
        log_results_to_wandb(results, args.stage)

    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Model: {args.model_path}")
    print(f"Stage: {args.stage}")
    print("-"*60)

    if 'results' in results:
        for benchmark, score in results['results'].items():
            print(f"{benchmark}: {score}")

    if 'robot_selection' in results:
        print("-"*60)
        print("Robot Selection:")
        for metric, value in results['robot_selection'].items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")

    print("="*60)


if __name__ == '__main__':
    main()

