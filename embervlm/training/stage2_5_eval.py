"""
Stage 2.5: VLM Benchmark Evaluation
Evaluates the base VLM (after Stage 2) on standard benchmarks before robot-specific training.
"""

import logging
import os
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch

logger = logging.getLogger(__name__)


def check_vlmeval_installation():
    """Check if VLMEvalKit is installed and provide helpful instructions if not."""
    try:
        import vlmeval
        logger.info("✅ VLMEvalKit is installed")
        return True
    except ImportError:
        logger.error("""
╔══════════════════════════════════════════════════════════╗
║   ⚠️  VLMEvalKit Not Installed                          ║
╚══════════════════════════════════════════════════════════╝

VLMEvalKit is required for benchmarking but is not installed.

To install VLMEvalKit:

1. Run the setup script:
   python setup_vlmeval.py

2. Or install manually:
   cd ../VLMEvalKit
   pip install -e .
   cd ../EmberVLM

3. Or skip benchmarking:
   Add --skip_benchmarks flag to your training command

For more info: https://github.com/open-compass/VLMEvalKit
""")
        return False


def log_results_to_wandb(results: Dict[str, float], results_summary: Dict):
    """Log benchmark results to WandB with comprehensive tables and visualizations."""
    try:
        import wandb
        
        # Check if wandb is initialized
        if wandb.run is None:
            logger.warning("WandB not initialized, skipping logging")
            return
        
        logger.info("📊 Creating WandB tables and visualizations...")
        
        # Create main results table
        results_table = wandb.Table(columns=[
            "Benchmark", "Score (%)", "Baseline (%)", "Δ vs Baseline", "Status"
        ])
        
        for benchmark, score in results.items():
            baseline = BASELINE_SCORES.get(benchmark, 30.0)
            delta = score - baseline
            threshold = baseline * 0.8
            status = "✅ Pass" if score >= threshold else "⚠️ Below"
            
            results_table.add_data(
                benchmark,
                f"{score:.2f}",
                f"{baseline:.2f}",
                f"{delta:+.2f}",
                status
            )
        
        # Log the main results table
        wandb.log({"stage2_5/benchmark_results_table": results_table})
        
        # Create summary metrics table
        num_benchmarks = len(results)
        avg_score = sum(results.values()) / num_benchmarks if num_benchmarks > 0 else 0
        min_score = min(results.values()) if results else 0
        max_score = max(results.values()) if results else 0
        passed_count = sum(1 for b, s in results.items() if s >= BASELINE_SCORES.get(b, 30.0) * 0.8)
        
        summary_table = wandb.Table(columns=["Metric", "Value"])
        summary_table.add_data("Aggregate Score", f"{results_summary['aggregate_score']:.2f}%")
        summary_table.add_data("Average Score", f"{avg_score:.2f}%")
        summary_table.add_data("Min Score", f"{min_score:.2f}%")
        summary_table.add_data("Max Score", f"{max_score:.2f}%")
        summary_table.add_data("Benchmarks Passed", f"{passed_count}/{num_benchmarks}")
        summary_table.add_data("Quality Check", "✅ PASS" if results_summary['quality_check_passed'] else "❌ FAIL")
        
        wandb.log({"stage2_5/summary_metrics_table": summary_table})
        
        # Log individual benchmark scores as metrics
        wandb_metrics = {}
        for benchmark, score in results.items():
            wandb_metrics[f"stage2_5/benchmarks/{benchmark}"] = score
            baseline = BASELINE_SCORES.get(benchmark, 30.0)
            wandb_metrics[f"stage2_5/baseline/{benchmark}"] = baseline
            wandb_metrics[f"stage2_5/delta/{benchmark}"] = score - baseline
        
        # Log aggregate metrics
        wandb_metrics.update({
            "stage2_5/aggregate_score": results_summary['aggregate_score'],
            "stage2_5/average_score": avg_score,
            "stage2_5/min_score": min_score,
            "stage2_5/max_score": max_score,
            "stage2_5/pass_rate": passed_count / num_benchmarks * 100 if num_benchmarks > 0 else 0,
            "stage2_5/num_passed": passed_count,
            "stage2_5/num_total": num_benchmarks,
            "stage2_5/quality_check_passed": 1 if results_summary['quality_check_passed'] else 0,
        })
        
        wandb.log(wandb_metrics)
        
        # Create comparison bar chart
        benchmark_names = list(results.keys())
        data = []
        for name in benchmark_names:
            model_score = results[name]
            baseline_score = BASELINE_SCORES.get(name, 30.0)
            data.append([name, model_score, "EmberVLM"])
            data.append([name, baseline_score, "SmolVLM Baseline"])
        
        comparison_table = wandb.Table(data=data, columns=["Benchmark", "Score", "Model"])
        wandb.log({
            "stage2_5/benchmark_comparison_chart": wandb.plot.bar(
                comparison_table, "Benchmark", "Score",
                title="EmberVLM vs SmolVLM Baseline Comparison"
            )
        })
        
        # Log delta analysis table
        delta_table = wandb.Table(columns=["Benchmark", "EmberVLM", "Baseline", "Delta", "Delta %"])
        for benchmark in benchmark_names:
            score = results[benchmark]
            baseline = BASELINE_SCORES.get(benchmark, 30.0)
            delta = score - baseline
            delta_pct = (delta / baseline * 100) if baseline > 0 else 0.0
            delta_table.add_data(
                benchmark,
                f"{score:.2f}",
                f"{baseline:.2f}",
                f"{delta:+.2f}",
                f"{delta_pct:+.1f}%"
            )
        
        wandb.log({"stage2_5/delta_analysis_table": delta_table})
        
        logger.info("✅ Successfully logged benchmark results to WandB")
        logger.info(f"   - Results table: stage2_5/benchmark_results_table")
        logger.info(f"   - Summary metrics: stage2_5/summary_metrics_table")
        logger.info(f"   - Comparison chart: stage2_5/benchmark_comparison_chart")
        logger.info(f"   - Delta analysis: stage2_5/delta_analysis_table")
        
    except ImportError:
        logger.warning("WandB not installed, skipping WandB logging")
    except Exception as e:
        logger.warning(f"Failed to log results to WandB: {e}")
        import traceback
        traceback.print_exc()
        'benchmarks': [
            'MMBench_DEV_EN_V11',  # General multimodal understanding
            'TextVQA_VAL',          # OCR + VQA
            'ScienceQA_IMG',        # Reasoning
            'AI2D_TEST',            # Diagram understanding
            'ChartQA_TEST',         # Chart reasoning
            'SEED_IMG',             # Visual understanding
        ],
        'description': 'Standard VLM suite (6 benchmarks, ~1-2 hours)',
        'expected_minutes': 90,
    },
    'full': {
        'benchmarks': [
            'MMBench_DEV_EN_V11', 'SEED_IMG', 'TextVQA_VAL', 'VQAv2_VAL',
            'OCRBench', 'ChartQA_TEST', 'AI2D_TEST', 'ScienceQA_IMG',
            'MMStar', 'MMMU_DEV_VAL'
        ],
        'description': 'Full evaluation suite (10+ benchmarks, ~4-6 hours)',
        'expected_minutes': 300,
    }
}

# Baseline scores (SmolVLM-256M for reference)
BASELINE_SCORES = {
    'MMBench_DEV_EN_V11': 35.0,
    'TextVQA_VAL': 28.0,
    'ScienceQA_IMG': 42.0,
    'AI2D_TEST': 36.0,
    'ChartQA_TEST': 18.0,
    'SEED_IMG': 48.0,
}

# Quality thresholds (percentage of baseline to pass)
QUALITY_THRESHOLDS = {
    'strict': 0.85,      # Must achieve 85% of baseline
    'standard': 0.70,    # Must achieve 70% of baseline
    'permissive': 0.50,  # Must achieve 50% of baseline
    'auto': 0.65,        # Automatic: 65% of baseline
}


def run_vlmevalkit_eval(
    model_path: str,
    benchmark: str,
    output_dir: str,
    vlmeval_repo: str = "d:\\BabyLM\\VLMEvalKit"
) -> Optional[float]:
    """
    Run single benchmark using VLMEvalKit.
    
    Args:
        model_path: Path to EmberVLM checkpoint
        benchmark: Benchmark name (e.g., 'MMBench_DEV_EN_V11')
        output_dir: Output directory for results
        vlmeval_repo: Path to VLMEvalKit repository
        
    Returns:
        Score (float) or None if failed
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Running {benchmark}...")
    
    try:
        # Prepare VLMEvalKit command
        cmd = [
            "python",
            str(Path(vlmeval_repo) / "run.py"),
            "--data", benchmark,
            "--model", "embervlm_custom",  # Will need to register
            "--work-dir", str(output_dir),
            "--verbose"
        ]
        
        # Set environment variable for custom model path
        env = os.environ.copy()
        env['EMBERVLM_CHECKPOINT'] = str(model_path)
        
        # Run evaluation
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout per benchmark
        )
        
        if result.returncode != 0:
            logger.error(f"Benchmark {benchmark} failed:")
            logger.error(result.stderr)
            return None
        
        # Parse results
        result_file = output_dir / f"{benchmark}_embervlm_custom.json"
        if result_file.exists():
            with open(result_file, 'r') as f:
                results = json.load(f)
                # Extract main metric (varies by benchmark)
                score = extract_benchmark_score(results, benchmark)
                logger.info(f"✓ {benchmark}: {score:.2f}%")
                return score
        else:
            logger.warning(f"Result file not found for {benchmark}")
            return None
            
    except subprocess.TimeoutExpired:
        logger.error(f"Benchmark {benchmark} timed out")
        return None
    except Exception as e:
        logger.error(f"Error running {benchmark}: {e}")
        return None


def extract_benchmark_score(results: Dict, benchmark: str) -> float:
    """Extract the primary metric score from benchmark results."""
    # Different benchmarks use different metric names
    metric_keys = {
        'MMBench': 'Overall',
        'TextVQA': 'accuracy',
        'ScienceQA': 'accuracy',
        'AI2D': 'accuracy',
        'ChartQA': 'accuracy',
        'SEED': 'Overall',
        'OCRBench': 'Overall',
    }
    
    # Find matching metric
    for key_pattern, metric_name in metric_keys.items():
        if key_pattern in benchmark:
            if metric_name in results:
                return float(results[metric_name])
            # Try common alternatives
            for alt in ['score', 'acc', 'overall']:
                if alt in results:
                    return float(results[alt])
    
    # Fallback: return first numeric value
    for v in results.values():
        if isinstance(v, (int, float)):
            return float(v)
    
    logger.warning(f"Could not extract score from results: {results}")
    return 0.0


def compute_aggregate_score(results: Dict[str, float]) -> float:
    """
    Compute weighted aggregate score across benchmarks.
    
    Args:
        results: Dict mapping benchmark names to scores
        
    Returns:
        Weighted average score
    """
    # Category weights
    weights = {
        'mmbench': 0.25,
        'textvqa': 0.20,
        'scienceqa': 0.15,
        'ai2d': 0.15,
        'chartqa': 0.10,
        'seed': 0.15,
    }
    
    total_score = 0.0
    total_weight = 0.0
    
    for bench, score in results.items():
        bench_lower = bench.lower()
        weight = 0.1  # Default weight
        
        # Find matching weight
        for key, w in weights.items():
            if key in bench_lower:
                weight = w
                break
        
        total_score += score * weight
        total_weight += weight
        logger.info(f"  {bench:30s}: {score:6.2f}% (weight: {weight:.2f})")
    
    if total_weight > 0:
        aggregate = total_score / total_weight
    else:
        aggregate = 0.0
    
    logger.info(f"\n  {'Weighted Aggregate':30s}: {aggregate:6.2f}%")
    return aggregate


def check_quality_threshold(
    aggregate_score: float,
    results: Dict[str, float],
    threshold_mode: str = 'auto'
) -> Tuple[bool, str]:
    """
    Check if model meets quality threshold.
    
    Args:
        aggregate_score: Weighted aggregate score
        results: Individual benchmark results
        threshold_mode: 'strict', 'standard', 'permissive', or 'auto'
        
    Returns:
        (passed, explanation)
    """
    threshold_pct = QUALITY_THRESHOLDS.get(threshold_mode, QUALITY_THRESHOLDS['auto'])
    
    # Compute expected aggregate baseline
    baseline_aggregate = 0.0
    total_weight = 0.0
    weights = {'mmbench': 0.25, 'textvqa': 0.20, 'scienceqa': 0.15, 'ai2d': 0.15, 'chartqa': 0.10, 'seed': 0.15}
    
    for bench in results.keys():
        bench_lower = bench.lower()
        for key, weight in weights.items():
            if key in bench_lower:
                if bench in BASELINE_SCORES:
                    baseline_aggregate += BASELINE_SCORES[bench] * weight
                    total_weight += weight
    
    if total_weight > 0:
        baseline_aggregate /= total_weight
    else:
        baseline_aggregate = 35.0  # Fallback
    
    required_score = baseline_aggregate * threshold_pct
    passed = aggregate_score >= required_score
    
    explanation = (
        f"Aggregate score: {aggregate_score:.2f}% | "
        f"Required: {required_score:.2f}% ({threshold_pct*100:.0f}% of {baseline_aggregate:.2f}% baseline) | "
        f"Status: {'✓ PASS' if passed else '✗ FAIL'}"
    )
    
    return passed, explanation


def run_stage2_5_evaluation(
    model_path: str,
    output_dir: str,
    preset: str = 'standard',
    threshold_mode: str = 'auto',
    vlmeval_repo: str = "d:\\BabyLM\\VLMEvalKit",
) -> Tuple[bool, Dict]:
    """
    Run Stage 2.5 VLM benchmark evaluation.
    
    Args:
        model_path: Path to Stage 2 checkpoint
        output_dir: Output directory for evaluation results
        preset: 'quick', 'standard', or 'full'
        threshold_mode: Quality threshold mode
        vlmeval_repo: Path to VLMEvalKit repository
        
    Returns:
        (passed_quality_check, results_dict)
    """
    logger.info("="*80)
    logger.info("STAGE 2.5: VLM BENCHMARK EVALUATION")
    logger.info("="*80)
    
    config = BENCHMARK_PRESETS.get(preset, BENCHMARK_PRESETS['standard'])
    logger.info(f"Preset: {preset}")
    logger.info(f"Description: {config['description']}")
    logger.info(f"Benchmarks: {', '.join(config['benchmarks'])}")
    logger.info(f"Expected time: ~{config.get('expected_minutes', '?')} minutes")
    logger.info("")
    
    # Create output directory
    eval_dir = Path(output_dir) / 'stage2_5_evaluation'
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    # Run benchmarks
    results = {}
    for benchmark in config['benchmarks']:
        score = run_vlmevalkit_eval(
            model_path=model_path,
            benchmark=benchmark,
            output_dir=str(eval_dir / benchmark),
            vlmeval_repo=vlmeval_repo
        )
        
        if score is not None:
            results[benchmark] = score
        else:
            logger.warning(f"Benchmark {benchmark} failed, using 0.0")
            results[benchmark] = 0.0
    
    # Compute aggregate score
    logger.info("\n" + "="*60)
    logger.info("BENCHMARK RESULTS SUMMARY")
    logger.info("="*60)
    aggregate_score = compute_aggregate_score(results)
    
    # Check quality threshold
    passed, explanation = check_quality_threshold(aggregate_score, results, threshold_mode)
    logger.info("\n" + "="*60)
    logger.info("QUALITY CHECK")
    logger.info("="*60)
    logger.info(explanation)
    logger.info("="*60)
    
    # Save results
    results_summary = {
        'preset': preset,
        'threshold_mode': threshold_mode,
        'benchmarks': results,
        'aggregate_score': aggregate_score,
        'quality_check_passed': passed,
        'explanation': explanation,
    }
    
    with open(eval_dir / 'evaluation_summary.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    logger.info(f"\n✓ Results saved to: {eval_dir / 'evaluation_summary.json'}")
    
    # Log to WandB with comprehensive tables and visualizations
    logger.info("\n📤 Logging results to WandB...")
    log_results_to_wandb(results, results_summary)
    
    return passed, results_summary
