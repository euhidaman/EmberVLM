"""
Training Visualization Script for EmberVLM
Generates real-time plots and analysis from training logs and WandB data.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import json

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11


class TrainingVisualizer:
    """Real-time training visualization from log files or WandB."""
    
    def __init__(self, output_dir: str = "./outputs/visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def plot_stage1_metrics(self, metrics_history: List[Dict], save_path: Optional[str] = None):
        """Plot Stage 1 (alignment) training curves."""
        if not metrics_history:
            print("No metrics to plot")
            return
            
        df = pd.DataFrame(metrics_history)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Stage 1: Visual-Language Alignment Training', fontsize=16, fontweight='bold')
        
        # 1. Contrastive Loss
        ax = axes[0, 0]
        if 'val_contrastive_loss' in df.columns:
            ax.plot(df.index, df['val_contrastive_loss'], 'b-', linewidth=2, label='Validation')
        if 'contrastive_loss' in df.columns:
            ax.plot(df.index, df['contrastive_loss'], 'b--', alpha=0.3, label='Train')
        ax.set_title('Contrastive Loss', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Top-1 Accuracy
        ax = axes[0, 1]
        if 'val_acc_i2t' in df.columns:
            ax.plot(df.index, df['val_acc_i2t'] * 100, 'g-', linewidth=2, label='Image→Text')
        if 'val_acc_t2i' in df.columns:
            ax.plot(df.index, df['val_acc_t2i'] * 100, 'orange', linewidth=2, label='Text→Image')
        ax.set_title('Retrieval Accuracy (Top-1)', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=5, color='r', linestyle='--', alpha=0.5, label='5% baseline')
        
        # 3. Top-5 Accuracy
        ax = axes[0, 2]
        if 'val_acc_i2t_top5' in df.columns:
            ax.plot(df.index, df['val_acc_i2t_top5'] * 100, 'g-', linewidth=2, label='Image→Text (Top-5)')
        if 'val_acc_t2i_top5' in df.columns:
            ax.plot(df.index, df['val_acc_t2i_top5'] * 100, 'orange', linewidth=2, label='Text→Image (Top-5)')
        ax.set_title('Retrieval Accuracy (Top-5)', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Mean Reciprocal Rank
        ax = axes[1, 0]
        if 'val_i2t_mrr' in df.columns:
            ax.plot(df.index, df['val_i2t_mrr'], 'g-', linewidth=2, label='Image→Text MRR')
        if 'val_t2i_mrr' in df.columns:
            ax.plot(df.index, df['val_t2i_mrr'], 'orange', linewidth=2, label='Text→Image MRR')
        ax.set_title('Mean Reciprocal Rank', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MRR')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. Similarity Distribution
        ax = axes[1, 1]
        if 'val_mean_pos_similarity' in df.columns and 'val_mean_neg_similarity' in df.columns:
            ax.plot(df.index, df['val_mean_pos_similarity'], 'g-', linewidth=2, label='Positive Pairs')
            ax.plot(df.index, df['val_mean_neg_similarity'], 'r-', linewidth=2, label='Negative Pairs')
            ax.fill_between(df.index, df['val_mean_pos_similarity'], df['val_mean_neg_similarity'], 
                           alpha=0.2, color='blue', label='Separation')
        ax.set_title('Similarity Separation', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Cosine Similarity')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 6. Learning Rate
        ax = axes[1, 2]
        if 'lr' in df.columns:
            ax.plot(df.index, df['lr'], 'purple', linewidth=2)
        ax.set_title('Learning Rate', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('LR')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved Stage 1 visualization to {save_path}")
        else:
            plt.savefig(self.output_dir / 'stage1_training.png', dpi=300, bbox_inches='tight')
            print(f"Saved Stage 1 visualization to {self.output_dir / 'stage1_training.png'}")
        
        plt.close()
        
    def plot_stage2_metrics(self, metrics_history: List[Dict], save_path: Optional[str] = None):
        """Plot Stage 2 (instruction tuning) training curves."""
        if not metrics_history:
            print("No metrics to plot")
            return
            
        df = pd.DataFrame(metrics_history)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Stage 2: Instruction Tuning Training', fontsize=16, fontweight='bold')
        
        # 1. Loss
        ax = axes[0, 0]
        if 'val_loss' in df.columns:
            ax.plot(df.index, df['val_loss'], 'b-', linewidth=2, label='Validation')
        if 'loss' in df.columns:
            ax.plot(df.index, df['loss'], 'b--', alpha=0.3, label='Train')
        ax.set_title('Instruction Loss', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Perplexity
        ax = axes[0, 1]
        if 'val_perplexity' in df.columns:
            ax.plot(df.index, df['val_perplexity'], 'purple', linewidth=2)
            # Add horizontal line at perplexity=20 (target)
            ax.axhline(y=20, color='green', linestyle='--', alpha=0.5, label='Target (PPL=20)')
            ax.axhline(y=50, color='orange', linestyle='--', alpha=0.5, label='Baseline (PPL=50)')
        ax.set_title('Perplexity', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Perplexity')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Token-level Accuracy
        ax = axes[1, 0]
        if 'val_accuracy' in df.columns:
            ax.plot(df.index, df['val_accuracy'] * 100, 'g-', linewidth=2)
            ax.axhline(y=10, color='orange', linestyle='--', alpha=0.5, label='Target (10%)')
        ax.set_title('Token-Level Accuracy', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Learning Rate
        ax = axes[1, 1]
        if 'lr' in df.columns:
            ax.plot(df.index, df['lr'], 'purple', linewidth=2)
        ax.set_title('Learning Rate', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('LR')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved Stage 2 visualization to {save_path}")
        else:
            plt.savefig(self.output_dir / 'stage2_training.png', dpi=300, bbox_inches='tight')
            print(f"Saved Stage 2 visualization to {self.output_dir / 'stage2_training.png'}")
        
        plt.close()
    
    def plot_stage3_metrics(self, metrics_history: List[Dict], save_path: Optional[str] = None):
        """Plot Stage 3 (robot selection) training curves."""
        if not metrics_history:
            print("No metrics to plot")
            return
            
        df = pd.DataFrame(metrics_history)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Stage 3: Robot Fleet Selection Training', fontsize=16, fontweight='bold')
        
        # 1. Loss
        ax = axes[0, 0]
        if 'val_loss' in df.columns:
            ax.plot(df.index, df['val_loss'], 'b-', linewidth=2)
        ax.set_title('Classification Loss', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.grid(True, alpha=0.3)
        
        # 2. Overall Accuracy
        ax = axes[0, 1]
        if 'val_accuracy' in df.columns:
            ax.plot(df.index, df['val_accuracy'] * 100, 'g-', linewidth=2)
            ax.axhline(y=50, color='orange', linestyle='--', alpha=0.5, label='Target (50%)')
        ax.set_title('Classification Accuracy', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Per-Class F1 Scores
        ax = axes[1, 0]
        robot_classes = ['Drone', 'Humanoid', 'Underwater', 'Wheels', 'Legs']
        for cls in robot_classes:
            col = f'val_f1_{cls.lower()}'
            if col in df.columns:
                ax.plot(df.index, df[col], linewidth=2, label=cls, marker='o', markersize=4)
        ax.set_title('Per-Class F1 Scores', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('F1 Score')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.0])
        
        # 4. Macro F1
        ax = axes[1, 1]
        if 'val_macro_f1' in df.columns:
            ax.plot(df.index, df['val_macro_f1'], 'purple', linewidth=2, marker='o', markersize=4)
            ax.axhline(y=0.4, color='green', linestyle='--', alpha=0.5, label='Target (0.4)')
        ax.set_title('Macro F1 Score', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Macro F1')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.0])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved Stage 3 visualization to {save_path}")
        else:
            plt.savefig(self.output_dir / 'stage3_training.png', dpi=300, bbox_inches='tight')
            print(f"Saved Stage 3 visualization to {self.output_dir / 'stage3_training.png'}")
        
        plt.close()


def parse_train_log(log_path: str) -> Dict[str, List[Dict]]:
    """Parse training log file and extract metrics."""
    stage_metrics = {
        'stage1': [],
        'stage2': [],
        'stage3': [],
        'stage4': []
    }
    
    current_stage = None
    current_epoch_metrics = {}
    
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            # Detect stage transitions
            if 'Stage 1 complete' in line or 'Starting Stage 1' in line:
                current_stage = 'stage1'
            elif 'Stage 2 complete' in line or 'Starting Stage 2' in line:
                current_stage = 'stage2'
            elif 'Stage 3 complete' in line or 'Starting Stage 3' in line:
                current_stage = 'stage3'
            elif 'Stage 4 complete' in line or 'Starting Stage 4' in line:
                current_stage = 'stage4'
            
            # Extract validation metrics
            if 'Validation metrics:' in line and current_stage:
                # Parse metrics dict from log line
                try:
                    metrics_str = line.split('Validation metrics:')[1].strip()
                    # Simple parsing (can be improved with json.loads if formatted correctly)
                    stage_metrics[current_stage].append(current_epoch_metrics.copy())
                    current_epoch_metrics = {}
                except Exception as e:
                    continue
    
    return stage_metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize EmberVLM training progress")
    parser.add_argument('--log_path', type=str, default='train.log', help='Path to training log file')
    parser.add_argument('--output_dir', type=str, default='./outputs/visualizations', help='Output directory for plots')
    parser.add_argument('--stage', type=str, default='all', choices=['all', 'stage1', 'stage2', 'stage3', 'stage4'], help='Which stage to visualize')
    
    args = parser.parse_args()
    
    visualizer = TrainingVisualizer(output_dir=args.output_dir)
    
    print(f"📊 Generating training visualizations from {args.log_path}...")
    print(f"Output directory: {args.output_dir}")
    print()
    print("Note: For real-time monitoring, check your Weights & Biases dashboard!")
    print()


