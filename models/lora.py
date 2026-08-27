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


def inject_lora(
    model: nn.Module,
    target_module_names: List[str],
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.1,
) -> Set[str]:
    """
    Inject LoRA adapters into specified linear layers of a model.

    Walks the model's module tree, finds nn.Linear layers whose names
    contain any of the target strings, and replaces them with LoRALinear.

    Args:
        model: The model to inject LoRA into (e.g., SAM2's image encoder).
        target_module_names: List of substrings to match against module names.
            E.g., ["q_proj", "v_proj"] to target attention Q and V projections.
        rank: LoRA rank.
        alpha: LoRA scaling factor.
        dropout: Dropout rate for LoRA branches.

    Returns:
        Set of module names that were replaced with LoRA.

    Example:
        >>> injected = inject_lora(sam2.image_encoder, ["q_proj", "v_proj"], rank=8)
        >>> print(f"Injected LoRA into {len(injected)} layers")
    """
    injected_names = set()

    # We need to replace modules in-place. To do this safely, we collect
    # all replacements first, then apply them (can't modify dict during iteration).
    replacements = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Check if this module's name matches any target pattern
            if any(target in name for target in target_module_names):
                replacements.append((name, module))

    # Apply replacements
    for name, original_module in replacements:
        # Navigate to the parent module
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)

        # Replace the linear layer with a LoRA-wrapped version
        lora_module = LoRALinear(
            original_module, rank=rank, alpha=alpha, dropout=dropout
        )
        setattr(parent, parts[-1], lora_module)
        injected_names.add(name)

    return injected_names


def get_lora_params(model: nn.Module) -> List[nn.Parameter]:
    """
    Collect all LoRA parameters (A and B matrices) from a model.

    Used to create a separate parameter group for the optimizer, so LoRA
    params can have a different learning rate than the mask decoder params.
    """
    lora_params = []
    for module in model.modules():
        if isinstance(module, LoRALinear):
            lora_params.append(module.lora_A)
            lora_params.append(module.lora_B)
    return lora_params


def count_trainable_params(model: nn.Module) -> dict:
    """
    Count trainable vs. frozen parameters for logging/verification.

    Returns a dict with total, trainable, and frozen parameter counts.
    Call this after freezing and LoRA injection to verify the expected
    parameter budget.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "trainable_pct": 100.0 * trainable / max(total, 1),
    }
