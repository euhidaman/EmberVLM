"""
Reasoning Module for EmberVLM

Implements Chain-of-Thought reasoning capabilities inspired by DeepSeek-R1.
Includes reasoning heads for robot selection and action planning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Tuple


class ReasoningAttention(nn.Module):
    """
    Self-attention layer for reasoning heads.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()

        # Compute QKV
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if attention_mask is not None:
            attn = attn + attention_mask

        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(x.dtype)
        attn = self.dropout(attn)

        # Apply attention
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, -1)
        out = self.proj(out)

        return out


class ReasoningBlock(nn.Module):
    """
    Transformer block for reasoning module.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = ReasoningAttention(hidden_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)

        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention with residual
        x = x + self.attn(self.norm1(x), attention_mask)
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        return x


class ReasoningHead(nn.Module):
    """
    Reasoning head for generating chain-of-thought reasoning.
    """

    def __init__(
        self,
        input_dim: int = 384,  # Match tinyllm/30M-0.4 hidden size
        hidden_dim: int = 192,  # Reduced to match smaller model
        num_layers: int = 2,
        num_heads: int = 4,
        num_reasoning_steps: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_reasoning_steps = num_reasoning_steps

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Reasoning blocks
        self.blocks = nn.ModuleList([
            ReasoningBlock(hidden_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Step embeddings
        self.step_embeddings = nn.Embedding(num_reasoning_steps, hidden_dim)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, input_dim)
        self.norm = nn.LayerNorm(input_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        step_indices: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate reasoning for each step.

        Args:
            hidden_states: Input hidden states [B, seq_len, input_dim]
            step_indices: Which reasoning step [B] or None for all steps
            attention_mask: Optional attention mask

        Returns:
            Dictionary containing reasoning outputs
        """
        batch_size, seq_len, _ = hidden_states.size()

        # Project to reasoning space
        x = self.input_proj(hidden_states)

        # Add step embeddings if provided
        if step_indices is not None:
            step_emb = self.step_embeddings(step_indices)  # [B, hidden_dim]
            x = x + step_emb.unsqueeze(1)

        # Forward through reasoning blocks
        for block in self.blocks:
            x = block(x, attention_mask)

        # Project back
        output = self.output_proj(x)
        output = self.norm(output)

        return {
            'reasoning_output': output,
            'reasoning_hidden': x,
        }


class RobotSelectionHead(nn.Module):
    """
    Head for selecting optimal robot from fleet.
    """

    def __init__(
        self,
        input_dim: int = 384,  # Match tinyllm/30M-0.4 hidden size
        hidden_dim: int = 192,  # Reduced to match smaller model
        num_robots: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_robots = num_robots

        # Feature extraction
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Robot classification
        self.classifier = nn.Linear(hidden_dim, num_robots)

        # Confidence estimation
        self.confidence = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Select robot from fleet.

        Args:
            hidden_states: Input hidden states [B, seq_len, input_dim]
            attention_mask: Optional attention mask

        Returns:
            Dictionary containing robot selection outputs
        """
        # Pool across sequence
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            pooled = hidden_states.mean(dim=1)

        # Extract features
        features = self.feature_net(pooled)

        # Robot logits
        robot_logits = self.classifier(features)

        # Confidence score
        confidence = self.confidence(features)

        return {
            'robot_logits': robot_logits,
            'robot_probs': F.softmax(robot_logits, dim=-1),
            'confidence': confidence,
            'features': features,
        }


class ActionPlanningHead(nn.Module):
    """
    Head for generating action plans.
    """

    def __init__(
        self,
        input_dim: int = 384,  # Match tinyllm/30M-0.4 hidden size
        hidden_dim: int = 192,  # Reduced to match smaller model
        max_plan_steps: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.max_plan_steps = max_plan_steps

        # Plan encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Step-wise plan generation
        self.step_generator = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )

        # Plan step projection
        self.step_proj = nn.Linear(hidden_dim, input_dim)

        # Plan coherence scoring
        self.coherence_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        robot_features: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate action plan.

        Args:
            hidden_states: Input hidden states [B, seq_len, input_dim]
            robot_features: Optional robot selection features [B, hidden_dim]
            attention_mask: Optional attention mask

        Returns:
            Dictionary containing action plan outputs
        """
        batch_size = hidden_states.size(0)

        # Pool and encode
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            pooled = hidden_states.mean(dim=1)

        encoded = self.encoder(pooled)

        # Combine with robot features if provided
        if robot_features is not None:
            encoded = encoded + robot_features

        # Generate plan steps
        # Initialize GRU hidden state
        h0 = encoded.unsqueeze(0).repeat(2, 1, 1)

        # Create input sequence for autoregressive generation
        plan_input = encoded.unsqueeze(1).repeat(1, self.max_plan_steps, 1)

        plan_steps, _ = self.step_generator(plan_input, h0)
        plan_steps = self.step_proj(plan_steps)

        # Compute coherence score
        first_last = torch.cat([
            plan_steps[:, 0, :encoded.size(-1)],
            plan_steps[:, -1, :encoded.size(-1)]
        ], dim=-1)
        coherence = self.coherence_scorer(first_last)

        return {
            'plan_steps': plan_steps,
            'plan_hidden': encoded,
            'coherence_score': coherence,
        }


class ReasoningModule(nn.Module):
    """
    Complete reasoning module for EmberVLM.

    Combines chain-of-thought reasoning with robot selection
    and action planning capabilities.
    """

    def __init__(
        self,
        input_dim: int = 384,  # Match tinyllm/30M-0.4 hidden size
        hidden_dim: int = 192,  # Reduced to match smaller model
        num_reasoning_layers: int = 2,
        num_reasoning_heads: int = 4,
        num_reasoning_steps: int = 4,
        num_robots: int = 5,
        max_plan_steps: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_reasoning_steps = num_reasoning_steps

        # Reasoning head
        self.reasoning_head = ReasoningHead(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_reasoning_layers,
            num_heads=num_reasoning_heads,
            num_reasoning_steps=num_reasoning_steps,
            dropout=dropout,
        )

        # Robot selection head
        self.robot_head = RobotSelectionHead(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_robots=num_robots,
            dropout=dropout,
        )

        # Action planning head
        self.action_head = ActionPlanningHead(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            max_plan_steps=max_plan_steps,
            dropout=dropout,
        )

        # Special token embeddings
        self.special_tokens = nn.ParameterDict({
            'reasoning_start': nn.Parameter(torch.randn(1, input_dim) * 0.02),
            'reasoning_end': nn.Parameter(torch.randn(1, input_dim) * 0.02),
            'robot_selection': nn.Parameter(torch.randn(1, input_dim) * 0.02),
            'action_plan': nn.Parameter(torch.randn(1, input_dim) * 0.02),
        })

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        generate_reasoning: bool = True,
        select_robot: bool = True,
        plan_actions: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Complete reasoning forward pass.

        Args:
            hidden_states: Input hidden states [B, seq_len, input_dim]
            attention_mask: Optional attention mask
            generate_reasoning: Whether to generate reasoning chain
            select_robot: Whether to perform robot selection
            plan_actions: Whether to generate action plan

        Returns:
            Dictionary containing all reasoning outputs
        """
        outputs = {}

        # Generate reasoning chain
        if generate_reasoning:
            reasoning_outputs = []
            current_hidden = hidden_states

            for step in range(self.num_reasoning_steps):
                step_idx = torch.full(
                    (hidden_states.size(0),),
                    step,
                    dtype=torch.long,
                    device=hidden_states.device
                )

                step_output = self.reasoning_head(
                    current_hidden,
                    step_indices=step_idx,
                    attention_mask=attention_mask,
                )
                reasoning_outputs.append(step_output['reasoning_output'])

                # Update hidden states with reasoning output
                current_hidden = current_hidden + 0.1 * step_output['reasoning_output']

            outputs['reasoning_chain'] = torch.stack(reasoning_outputs, dim=1)  # [B, steps, seq, dim]
            outputs['reasoning_hidden'] = current_hidden
        else:
            current_hidden = hidden_states

        # Robot selection
        if select_robot:
            robot_outputs = self.robot_head(current_hidden, attention_mask)
            outputs.update({
                'robot_logits': robot_outputs['robot_logits'],
                'robot_probs': robot_outputs['robot_probs'],
                'robot_confidence': robot_outputs['confidence'],
            })
            robot_features = robot_outputs['features']
        else:
            robot_features = None

        # Action planning
        if plan_actions:
            action_outputs = self.action_head(
                current_hidden,
                robot_features=robot_features,
                attention_mask=attention_mask,
            )
            outputs.update({
                'plan_steps': action_outputs['plan_steps'],
                'plan_coherence': action_outputs['coherence_score'],
            })

        return outputs

    def get_special_token_embedding(self, token_name: str) -> torch.Tensor:
        """Get embedding for a special token."""
        return self.special_tokens[token_name]

    def count_parameters(self) -> Dict[str, int]:
        """Count parameters in reasoning module."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable}


class ReasoningLoss(nn.Module):
    """
    Loss functions for reasoning module.
    """

    def __init__(
        self,
        num_robots: int = 5,
        reasoning_weight: float = 1.0,
        robot_weight: float = 1.0,
        action_weight: float = 1.0,
        consistency_weight: float = 0.5,
    ):
        super().__init__()

        self.reasoning_weight = reasoning_weight
        self.robot_weight = robot_weight
        self.action_weight = action_weight
        self.consistency_weight = consistency_weight

        self.robot_criterion = nn.CrossEntropyLoss()
        self.action_criterion = nn.MSELoss()

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute reasoning losses.

        Args:
            outputs: Model outputs
            targets: Target values

        Returns:
            Dictionary of losses
        """
        losses = {}
        total_loss = 0.0

        # Robot selection loss
        if 'robot_logits' in outputs and 'robot_target' in targets:
            robot_loss = self.robot_criterion(
                outputs['robot_logits'],
                targets['robot_target']
            )
            losses['robot_loss'] = robot_loss
            total_loss = total_loss + self.robot_weight * robot_loss

        # Action planning loss
        if 'plan_steps' in outputs and 'action_target' in targets:
            action_loss = self.action_criterion(
                outputs['plan_steps'],
                targets['action_target']
            )
            losses['action_loss'] = action_loss
            total_loss = total_loss + self.action_weight * action_loss

        # Reasoning consistency loss
        if 'reasoning_chain' in outputs:
            # Encourage smooth transitions between reasoning steps
            reasoning_chain = outputs['reasoning_chain']
            if reasoning_chain.size(1) > 1:
                step_diffs = reasoning_chain[:, 1:] - reasoning_chain[:, :-1]
                consistency_loss = torch.mean(step_diffs ** 2)
                losses['consistency_loss'] = consistency_loss
                total_loss = total_loss + self.consistency_weight * consistency_loss

        losses['total_loss'] = total_loss
        return losses

