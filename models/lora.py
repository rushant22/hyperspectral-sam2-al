"""
models/lora.py — Low-Rank Adaptation (LoRA) for SAM2's Hiera backbone.

LoRA decomposes weight updates into two low-rank matrices:
  W' = W₀ + (α/r) · B · A

where:
  - W₀ is the frozen pretrained weight matrix (d_out × d_in)
  - A ∈ R^(r × d_in): down-projection, initialized from N(0, σ²)
  - B ∈ R^(d_out × r): up-projection, initialized to zeros
  - r: rank (hyperparameter, we use r=8)
  - α: scaling factor (we use α=16, so effective scale = α/r = 2.0)

This means:
  - At initialization, B·A = 0, so W' = W₀ (no change to pretrained behavior)
  - During training, only A and B are updated (2 × d × r params vs. d² for full)
  - At inference, B·A can be merged into W₀ for zero additional latency

Why LoRA instead of full fine-tuning:
  - SAM2 Base+ has ~80M encoder params. Fine-tuning all of them on small HSI
    datasets (Indian Pines has ~10K labeled pixels) would overfit catastrophically.
  - LoRA with r=8 adds only ~0.3M trainable params — a 250× reduction.
    For SAM2 Hiera, ["qkv"] targets the fused QKV projection.
    FusedQKVLoRA applies LoRA only to Q and V while keeping K frozen.
  - Memory: only LoRA params need optimizer states, saving ~4× VRAM vs. full.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Set


class LoRALinear(nn.Module):
    """
    A linear layer wrapped with LoRA adapters.

    Replaces a standard nn.Linear with W' = W₀ + (α/r) · B · A, where
    W₀ is frozen and only A, B are trainable.
    """

    def __init__(
        self,
        original_linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.1,
    ):
        """
        Args:
            original_linear: The pretrained nn.Linear to wrap.
            rank: LoRA rank (r). Lower = fewer params, higher = more capacity.
            alpha: Scaling factor. Effective scale is alpha/rank.
            dropout: Dropout applied to LoRA branch. Serves dual purpose:
                regularization during training AND MC-Dropout for uncertainty
                estimation during the active learning loop.
        """
        super().__init__()

        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.rank = rank
        self.scaling = alpha / rank

        # Frozen original weight and bias
        # Using register_buffer would also work, but keeping as a frozen
        # Parameter makes state_dict loading from SAM2 checkpoints easier
        self.weight = original_linear.weight
        self.weight.requires_grad = False
        if original_linear.bias is not None:
            self.bias = original_linear.bias
            self.bias.requires_grad = False
        else:
            self.bias = None

        # LoRA matrices: A (down-projection) and B (up-projection)
        # A: (rank, in_features) — initialized with Kaiming uniform
        # B: (out_features, rank) — initialized to zeros
        # Zero-init of B ensures that at the start, the LoRA contribution is 0,
        # preserving the pretrained model's behavior exactly.
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)

        # Dropout on the LoRA branch
        self.lora_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: y = W₀·x + bias + (α/r) · B · A · dropout(x)

        The dropout is applied to the input of the LoRA branch, not the output.
        This is important for MC-Dropout: during inference with dropout enabled,
        different forward passes will sample different subsets of the input
        features, giving us the stochastic variation needed for BALD computation.
        """
        # Original frozen linear transformation
        result = F.linear(x, self.weight, self.bias)

        # LoRA branch: apply dropout → A (down-project) → B (up-project) → scale
        lora_input = self.lora_dropout(x)
        lora_output = F.linear(F.linear(lora_input, self.lora_A), self.lora_B)
        result = result + self.scaling * lora_output

        return result

    def merge_lora(self) -> nn.Linear:
        """
        Merge LoRA weights into the base weight for inference efficiency.

        Returns a standard nn.Linear with W' = W₀ + (α/r) · B · A.
        No additional latency or memory at inference time.
        """
        merged = nn.Linear(self.in_features, self.out_features, bias=self.bias is not None)
        merged.weight.data = self.weight.data + self.scaling * (self.lora_B @ self.lora_A)
        if self.bias is not None:
            merged.bias.data = self.bias.data
        return merged

class FusedQKVLoRA(nn.Module):
    """
    LoRA wrapper for SAM2 Hiera's fused QKV projection.

    SAM2 uses one Linear layer that produces:

        [Q | K | V]

    Therefore:

        Q = first 1/3 of output
        K = middle 1/3
        V = last 1/3

    This wrapper applies LoRA only to Q and V while keeping K frozen.

    W' = W0 + (alpha/rank) * DeltaW

    where DeltaW is applied to the Q and V portions only.
    """

    def __init__(
        self,
        original_linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features

        if self.out_features != 3 * self.in_features:
            raise ValueError(
                f"FusedQKVLoRA expects out_features == "
                f"3 * in_features, but got "
                f"{self.in_features} -> {self.out_features}"
            )

        self.rank = rank
        self.scaling = alpha / rank

        # Frozen original fused QKV weight
        self.weight = original_linear.weight
        self.weight.requires_grad = False

        if original_linear.bias is not None:
            self.bias = original_linear.bias
            self.bias.requires_grad = False
        else:
            self.bias = None

        # ---------------------------------------------------------
        # Q LoRA
        # ---------------------------------------------------------
        self.q_lora_A = nn.Parameter(
            torch.empty(rank, self.in_features)
        )

        self.q_lora_B = nn.Parameter(
            torch.zeros(self.in_features, rank)
        )

        # ---------------------------------------------------------
        # V LoRA
        # ---------------------------------------------------------
        self.v_lora_A = nn.Parameter(
            torch.empty(rank, self.in_features)
        )

        self.v_lora_B = nn.Parameter(
            torch.zeros(self.in_features, rank)
        )

        nn.init.kaiming_uniform_(
            self.q_lora_A,
            a=5**0.5
        )

        nn.init.kaiming_uniform_(
            self.v_lora_A,
            a=5**0.5
        )

        self.lora_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward through frozen fused QKV projection plus
        trainable Q/V LoRA branches.
        """

        # Original frozen QKV projection
        result = F.linear(
            x,
            self.weight,
            self.bias
        )

        # Split into Q, K, V
        q, k, v = result.chunk(3, dim=-1)

        # Apply dropout to LoRA input
        lora_input = self.lora_dropout(x)

        # Q LoRA
        q_update = F.linear(
            F.linear(
                lora_input,
                self.q_lora_A
            ),
            self.q_lora_B
        )

        # V LoRA
        v_update = F.linear(
            F.linear(
                lora_input,
                self.v_lora_A
            ),
            self.v_lora_B
        )

        # Apply scaling
        q = q + self.scaling * q_update
        v = v + self.scaling * v_update

        # Reconstruct fused QKV
        return torch.cat(
            [q, k, v],
            dim=-1
        )

    def merge_lora(self) -> nn.Linear:
        """
        Merge Q/V LoRA updates into the original fused QKV weight.
        """

        merged = nn.Linear(
            self.in_features,
            self.out_features,
            bias=self.bias is not None,
        )

        with torch.no_grad():

            merged.weight.copy_(self.weight)

            q_delta = (
                self.scaling
                * (self.q_lora_B @ self.q_lora_A)
            )

            v_delta = (
                self.scaling
                * (self.v_lora_B @ self.v_lora_A)
            )

            d = self.in_features

            merged.weight[:d] += q_delta
            merged.weight[2 * d:3 * d] += v_delta

            if self.bias is not None:
                merged.bias.copy_(self.bias)

        return merged

def inject_lora(
    model: nn.Module,
    target_module_names: List[str],
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.1,
) -> Set[str]:
    """
    Inject LoRA adapters into specified linear layers.

    Supports:

        q_proj / v_proj
            -> standard LoRALinear

        qkv
            -> FusedQKVLoRA

    For SAM2 Hiera, targeting "qkv" applies LoRA only to
    the Q and V portions of the fused QKV projection.
    """

    injected_names = set()
    replacements = []

    for name, module in model.named_modules():

        if not isinstance(module, nn.Linear):
            continue

        if not any(
            target in name
            for target in target_module_names
        ):
            continue

        # ---------------------------------------------------------
        # SAM2 Hiera fused QKV
        # ---------------------------------------------------------
        if name.endswith(".qkv"):

            if module.out_features != 3 * module.in_features:
                raise ValueError(
                    f"Expected fused QKV layer at {name}, "
                    f"but found "
                    f"{module.in_features} -> "
                    f"{module.out_features}"
                )

            replacement = FusedQKVLoRA(
                module,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            )

        # ---------------------------------------------------------
        # Standard Linear layer
        # ---------------------------------------------------------
        else:

            replacement = LoRALinear(
                module,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            )

        replacements.append(
            (name, replacement)
        )

    # Apply replacements after traversal
    for name, replacement in replacements:

        parts = name.split(".")
        parent = model

        for part in parts[:-1]:
            parent = getattr(parent, part)

        setattr(
            parent,
            parts[-1],
            replacement
        )

        injected_names.add(name)

    return injected_names


def get_lora_params(model: nn.Module) -> List[nn.Parameter]:
    """
    Collect all LoRA parameters.

    Supports both:

    - LoRALinear
    - FusedQKVLoRA
    """

    lora_params = []

    for module in model.modules():

        if isinstance(module, LoRALinear):

            lora_params.append(module.lora_A)
            lora_params.append(module.lora_B)

        elif isinstance(module, FusedQKVLoRA):

            lora_params.append(module.q_lora_A)
            lora_params.append(module.q_lora_B)

            lora_params.append(module.v_lora_A)
            lora_params.append(module.v_lora_B)

    return lora_params


def count_trainable_params(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )
    frozen = total - trainable

    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "trainable_pct": 100.0 * trainable / max(total, 1),
    }
