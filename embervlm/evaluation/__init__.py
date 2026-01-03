"""
EmberVLM Evaluation Package
"""

from embervlm.evaluation.metrics import (
    compute_robot_selection_metrics,
    compute_action_plan_metrics,
    compute_reasoning_metrics,
)

# VLMEvalKit integration (optional import)
try:
    from embervlm.evaluation.vlmevalkit_adapter import (
        EmberVLM_VLMEval,
        get_benchmarks_for_stage,
        STAGE1_BENCHMARKS,
        STAGE2_BENCHMARKS,
        STAGE3_BENCHMARKS,
        STAGE4_BENCHMARKS,
    )
    HAS_VLMEVAL = True
except ImportError:
    HAS_VLMEVAL = False
    EmberVLM_VLMEval = None
    get_benchmarks_for_stage = None
    STAGE1_BENCHMARKS = None
    STAGE2_BENCHMARKS = None
    STAGE3_BENCHMARKS = None
    STAGE4_BENCHMARKS = None

__all__ = [
    "compute_robot_selection_metrics",
    "compute_action_plan_metrics",
    "compute_reasoning_metrics",
    "EmberVLM_VLMEval",
    "get_benchmarks_for_stage",
    "STAGE1_BENCHMARKS",
    "STAGE2_BENCHMARKS",
    "STAGE3_BENCHMARKS",
    "STAGE4_BENCHMARKS",
    "HAS_VLMEVAL",
]

