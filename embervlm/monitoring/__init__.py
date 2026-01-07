"""
EmberVLM Monitoring Package

Provides comprehensive monitoring and visualization tools for training:
- W&B logging with advanced visualizations
- Carbon footprint tracking
- FLOPs counting
- Attention visualization
- Stage-specific visualizations
- Advanced 3D plots and publication-quality figures
"""

from embervlm.monitoring.wandb_logger import WandbLogger, EnhancedWandbLogger
from embervlm.monitoring.carbon_tracker import CarbonTracker
from embervlm.monitoring.flops_counter import FLOPsCounter, count_model_flops
from embervlm.monitoring.attention_viz import AttentionVisualizer, analyze_model_attention
from embervlm.monitoring.visualization import TrainingVisualizer
from embervlm.monitoring.stage_visualizations import (
    Stage1Visualizer,
    Stage2Visualizer,
    Stage3Visualizer,
    Stage4Visualizer,
    CrossStageVisualizer,
)
from embervlm.monitoring.advanced_visualizations import AdvancedVisualizer

__all__ = [
    "WandbLogger",
    "EnhancedWandbLogger",
    "CarbonTracker",
    "FLOPsCounter",
    "count_model_flops",
    "AttentionVisualizer",
    "analyze_model_attention",
    "TrainingVisualizer",
    "Stage1Visualizer",
    "Stage2Visualizer",
    "Stage3Visualizer",
    "Stage4Visualizer",
    "CrossStageVisualizer",
    "AdvancedVisualizer",
]

