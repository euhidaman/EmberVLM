"""
EmberVLM Training Package
"""

from .trainer import (
    DistributedTrainer,
    TrainingConfig,
    setup_training
)
from .stage1_align import (
    Stage1Config,
    Stage1AlignmentTrainer,
    ContrastiveLoss,
    run_stage1_alignment
)
from .stage2_instruct import (
    Stage2Config,
    Stage2InstructionTrainer,
    DistillationLoss,
    TeacherWrapper,
    run_stage2_instruction
)
from .stage3_reasoning import (
    Stage3Config,
    Stage3ReasoningTrainer,
    ReasoningFidelityLoss,
    ActionCoherenceLoss,
    CurriculumScheduler,
    run_stage3_reasoning
)
from .distillation import (
    DistillationConfig,
    DistillationLossModule,
    FeatureProjector,
    MultiLayerDistillation,
    AttentionTransfer,
    create_distillation_module
)

__all__ = [
    # Base trainer
    'DistributedTrainer',
    'TrainingConfig',
    'setup_training',
    # Stage 1
    'Stage1Config',
    'Stage1AlignmentTrainer',
    'ContrastiveLoss',
    'run_stage1_alignment',
    # Stage 2
    'Stage2Config',
    'Stage2InstructionTrainer',
    'DistillationLoss',
    'TeacherWrapper',
    'run_stage2_instruction',
    # Stage 3
    'Stage3Config',
    'Stage3ReasoningTrainer',
    'ReasoningFidelityLoss',
    'ActionCoherenceLoss',
    'CurriculumScheduler',
    'run_stage3_reasoning',
    # Distillation
    'DistillationConfig',
    'DistillationLossModule',
    'FeatureProjector',
    'MultiLayerDistillation',
    'AttentionTransfer',
    'create_distillation_module',
]

