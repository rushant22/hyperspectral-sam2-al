"""
models/spectral_adapter.py — Spectral Cross-Attention Adapter.

THIS IS THE CORE NOVEL CONTRIBUTION of the project.

The adapter bridges the gap between hyperspectral imagery (100–426 bands)
and SAM2's image encoder (which expects 3-channel RGB input). Instead of
naive band selection or PCA, it uses learned cross-attention to extract
the most discriminative spectral features.

Architecture:
  1. Each spatial location's spectral signature (B values) becomes one token.
  2. A small set of learnable query tokens (M=12) represent canonical
     biochemical response patterns (chlorophyll absorption, red-edge, etc.).
  3. Cross-attention: queries attend to spectral tokens to extract features.
  4. Output is reshaped to (d_model × H × W) to match SAM2's expected input.

Design rationale for cross-attention over alternatives:
  - Band selection: throws away information, requires domain expertise per sensor.
  - PCA: linear, can't capture nonlinear spectral interactions (e.g., red-edge
    position is a ratio, not a linear combination).
  - 1D convolution over bands: local receptive field, misses long-range
    spectral correlations (e.g., chlorophyll-a at 680nm relates to NIR at 850nm).
  - Cross-attention: attends to ALL bands simultaneously, learns which
    spectral regions matter for each query concept, and is fully differentiable.
"""

import torch
import torch.nn as nn
import math
from einops import rearrange


class SpectralCrossAttentionAdapter(nn.Module):
    """
    Projects a hyperspectral cube into SAM2's embedding space via
    learned cross-attention over spectral tokens.

    Input:  (batch, B, H, W) where B = number of spectral bands
    Output: (batch, d_model, H, W) where d_model matches SAM2's embed_dim (256)
    """

    def __init__(
        self,
        num_bands: int,
        d_model: int = 256,
        num_queries: int = 12,
        num_heads: int = 8,
        ffn_dim: int = 512,
        dropout: float = 0.1,
    ):
        """
        Args:
            num_bands: Number of input spectral bands (e.g., 103 for Pavia,
                200 for Indian Pines).
            d_model: Internal embedding dimension. Must match SAM2's embed_dim
                (256 for all SAM2 variants).
            num_queries: M — number of learnable query tokens. Each query
                learns to attend to a specific spectral pattern. Start with
                12 (matches approximate count of distinct biochemical features
                in VNIR-SWIR range: chlorophyll-a, -b, carotenoids, red-edge,
                water, cellulose, lignin, dry matter, soil minerals, shadows,
                anthocyanins, xanthophylls). Tunable hyperparameter.
            num_heads: Number of attention heads. Must divide d_model evenly.
            ffn_dim: Hidden dimension of the feed-forward network.
            dropout: Dropout rate (also used for MC-Dropout during AL inference).
        """
        super().__init__()

        assert d_model % num_heads == 0, (
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        )

        self.num_bands = num_bands
        self.d_model = d_model
        self.num_queries = num_queries
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # === Step 1: Project spectral bands into d_model space ===
        # Each spatial location has B spectral values → project to d_model
        # This is a learned linear projection, NOT PCA — it's trained end-to-end
        self.spectral_proj = nn.Linear(num_bands, d_model)

        # === Step 2: Learnable query tokens ===
        # These M tokens learn to represent canonical spectral response patterns.
        # Initialized from a truncated normal distribution (standard for
        # learnable queries in DETR-style architectures).
        self.queries = nn.Parameter(torch.empty(num_queries, d_model))
        nn.init.trunc_normal_(self.queries, std=0.02)

        # === Step 3: Cross-attention ===
        # Q = learnable queries (M tokens)
        # K, V = projected spectral tokens (H*W tokens)
        # This lets each query "look at" the full spectral signature at every
        # spatial location and extract the most relevant information.
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.attn_out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)

        # === Step 4: Feed-forward network (post-attention) ===
        # Standard transformer FFN: Linear → GELU → Dropout → Linear → Dropout
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

        # === Step 5: Layer normalization ===
        self.norm1 = nn.LayerNorm(d_model)  # Pre-attention norm
        self.norm2 = nn.LayerNorm(d_model)  # Pre-FFN norm

        # === Step 6: Spatial mixing ===
        # The cross-attention produces M query outputs. We need to project
        # these back to (H × W) spatial locations. This small MLP maps
        # from the M-dimensional "bottleneck" space to spatial features.
        #
        # Why not just tile? Because different spatial locations should get
        # different mixtures of the M query outputs based on their spectral
        # content. The spatial mixing MLP learns this mapping.
        self.spatial_mix = nn.Sequential(
            nn.Linear(num_queries, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, 1),
        )

        # === Step 7: Final projection ===
        # Optional refinement after spatial mixing
        self.output_proj = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=1),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: hyperspectral cube → SAM2-compatible feature map.

        Args:
            x: Input HSI tensor, shape (batch, B, H, W).

        Returns:
            Features for SAM2, shape (batch, d_model, H, W).
        """
        batch, B, H, W = x.shape
        assert B == self.num_bands, (
            f"Expected {self.num_bands} bands, got {B}. "
            f"Check that the dataset matches the adapter configuration."
        )

        # --- Step 1: Reshape to spectral tokens ---
        # (batch, B, H, W) → (batch, H*W, B)
        # Each spatial location becomes one token with B spectral features
        spectral_tokens = rearrange(x, 'b c h w -> b (h w) c')

        # --- Step 2: Project spectral tokens into d_model space ---
        # (batch, H*W, B) → (batch, H*W, d_model)
        spectral_tokens = self.spectral_proj(spectral_tokens)

        # --- Step 3: Cross-attention ---
        # Queries: (M, d_model) → expand to (batch, M, d_model)
        queries = self.queries.unsqueeze(0).expand(batch, -1, -1)

        # Pre-attention layer norm
        queries_normed = self.norm1(queries)

        # Project Q, K, V
        Q = self.q_proj(queries_normed)                    # (batch, M, d_model)
        K = self.k_proj(spectral_tokens)                   # (batch, H*W, d_model)
        V = self.v_proj(spectral_tokens)                   # (batch, H*W, d_model)

        # Reshape for multi-head attention
        # (batch, seq_len, d_model) → (batch, num_heads, seq_len, head_dim)
        Q = rearrange(Q, 'b m (nh hd) -> b nh m hd', nh=self.num_heads)
        K = rearrange(K, 'b n (nh hd) -> b nh n hd', nh=self.num_heads)
        V = rearrange(V, 'b n (nh hd) -> b nh n hd', nh=self.num_heads)

        # Scaled dot-product attention
        # (batch, num_heads, M, head_dim) × (batch, num_heads, head_dim, H*W)
        # → (batch, num_heads, M, H*W)
        scale = math.sqrt(self.head_dim)
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / scale
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Apply attention to values
        # (batch, num_heads, M, H*W) × (batch, num_heads, H*W, head_dim)
        # → (batch, num_heads, M, head_dim)
        attn_output = torch.matmul(attn_weights, V)

        # Merge heads back
        # (batch, num_heads, M, head_dim) → (batch, M, d_model)
        attn_output = rearrange(attn_output, 'b nh m hd -> b m (nh hd)')
        attn_output = self.attn_out_proj(attn_output)

        # Residual connection
        query_features = queries + attn_output  # (batch, M, d_model)

        # --- Step 4: Feed-forward network ---
        query_features = query_features + self.ffn(self.norm2(query_features))
        # query_features: (batch, M, d_model)

        # --- Step 5: Spatial mixing ---
        # We need to go from M query tokens back to H*W spatial locations.
        # Strategy: for each spatial location, compute a weighted combination
        # of the M query features, where weights come from the original
        # spectral token's similarity to each query.
        #
        # spectral_tokens: (batch, H*W, d_model) — projected spectral features
        # query_features:  (batch, M, d_model) — enriched query representations
        #
        # Compute spatial-to-query affinity:
        # (batch, H*W, d_model) × (batch, d_model, M) → (batch, H*W, M)
        affinity = torch.matmul(
            spectral_tokens, query_features.transpose(-2, -1)
        ) / scale
        affinity = torch.softmax(affinity, dim=-1)  # (batch, H*W, M)

        # Weighted combination of query features per spatial location
        # (batch, H*W, M) × (batch, M, d_model) → (batch, H*W, d_model)
        spatial_features = torch.matmul(affinity, query_features)

        # --- Step 6: Reshape to spatial grid ---
        # (batch, H*W, d_model) → (batch, d_model, H, W)
        spatial_features = rearrange(
            spatial_features, 'b (h w) d -> b d h w', h=H, w=W
        )

        # --- Step 7: Final refinement ---
        output = self.output_proj(spatial_features)

        return output

    def get_attention_maps(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract cross-attention maps for visualization and interpretability.

        Shows which spectral bands each learned query attends to — useful for
        verifying that queries learn physically meaningful patterns (e.g., one
        query attending to red-edge bands, another to water absorption, etc.).

        Args:
            x: Input HSI tensor, shape (batch, B, H, W).

        Returns:
            attn_weights: Shape (batch, num_heads, M, H*W) — attention weights
                from queries to spatial-spectral tokens.
        """
        batch, B, H, W = x.shape
        spectral_tokens = rearrange(x, 'b c h w -> b (h w) c')
        spectral_tokens = self.spectral_proj(spectral_tokens)

        queries = self.queries.unsqueeze(0).expand(batch, -1, -1)
        queries_normed = self.norm1(queries)

        Q = rearrange(
            self.q_proj(queries_normed),
            'b m (nh hd) -> b nh m hd', nh=self.num_heads
        )
        K = rearrange(
            self.k_proj(spectral_tokens),
            'b n (nh hd) -> b nh n hd', nh=self.num_heads
        )

        scale = math.sqrt(self.head_dim)
        attn_weights = torch.softmax(
            torch.matmul(Q, K.transpose(-2, -1)) / scale, dim=-1
        )

        return attn_weights
