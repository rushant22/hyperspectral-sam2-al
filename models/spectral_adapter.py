"""
models/spectral_adapter.py — Spectral Cross-Attention Adapter.

Core contribution:
    Adapt hyperspectral imagery to the SAM2 embedding space by performing
    cross-attention over the spectral bands at each spatial location.

Input:
    (batch, num_bands, H, W)

For every spatial pixel:
    spectral signature:
        [band_1, band_2, ..., band_B]

    becomes B spectral tokens.

Learnable query tokens attend over those B spectral tokens and extract
spectral-response features.

Output:
    (batch, d_model, H, W)
"""

import math

import torch
import torch.nn as nn
from einops import rearrange


class SpectralCrossAttentionAdapter(nn.Module):
    """
    Projects a hyperspectral cube into SAM2 embedding space using
    cross-attention over spectral bands independently at each spatial
    location.

    Input:
        (batch, B, H, W)

    Output:
        (batch, d_model, H, W)
    """

    def __init__(
        self,
        num_bands: int,
        d_model: int = 256,
        num_queries: int = 12,
        num_heads: int = 8,
        ffn_dim: int = 512,
        dropout: float = 0.1,
        pixel_chunk_size: int = 1024,
    ):
        super().__init__()

        assert d_model % num_heads == 0, (
            f"d_model ({d_model}) must be divisible by "
            f"num_heads ({num_heads})"
        )

        self.num_bands = num_bands
        self.d_model = d_model
        self.num_queries = num_queries
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.pixel_chunk_size = pixel_chunk_size

        # ------------------------------------------------------------
        # 1. Spectral token embedding
        # ------------------------------------------------------------
        #
        # Each spectral band is initially a scalar value for a pixel.
        # Map that scalar into the d_model-dimensional token space.
        #
        # Input:
        #     (N_pixels, B, 1)
        #
        # Output:
        #     (N_pixels, B, d_model)
        #
        self.spectral_proj = nn.Linear(1, d_model)

        # Learned spectral-position embedding.
        #
        # This allows the model to distinguish:
        #     band 50 from band 200
        #
        # even if their reflectance values happen to be similar.
        self.spectral_pos = nn.Parameter(
            torch.empty(1, num_bands, d_model)
        )
        nn.init.trunc_normal_(self.spectral_pos, std=0.02)

        # ------------------------------------------------------------
        # 2. Learnable spectral-response queries
        # ------------------------------------------------------------
        #
        # M=12 learned queries represent latent spectral response
        # patterns.
        #
        self.queries = nn.Parameter(
            torch.empty(num_queries, d_model)
        )
        nn.init.trunc_normal_(self.queries, std=0.02)

        # ------------------------------------------------------------
        # 3. Cross-attention projections
        # ------------------------------------------------------------

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.attn_out_proj = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(dropout)

        # ------------------------------------------------------------
        # 4. Transformer FFN
        # ------------------------------------------------------------

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

        # ------------------------------------------------------------
        # 5. Query aggregation
        # ------------------------------------------------------------
        #
        # After attention we have M query representations per pixel.
        #
        # Learn a mixture of the M query features for each pixel.
        #
        self.query_gate = nn.Sequential(
            nn.Linear(d_model, num_queries),
        )

        # ------------------------------------------------------------
        # 6. Final spatial refinement
        # ------------------------------------------------------------

        self.output_proj = nn.Sequential(
            nn.Conv2d(
                d_model,
                d_model,
                kernel_size=1,
            ),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
        )

    def _cross_attention(
        self,
        spectral_tokens: torch.Tensor,
    ):
        """
        Perform query-to-spectral cross-attention.

        Args:
            spectral_tokens:
                (N_pixels, B, d_model)

        Returns:
            query_features:
                (N_pixels, M, d_model)

            attn_weights:
                (N_pixels, num_heads, M, B)
        """

        n_pixels = spectral_tokens.shape[0]

        # Expand learned queries for every pixel.
        queries = self.queries.unsqueeze(0).expand(
            n_pixels,
            -1,
            -1,
        )

        # Pre-normalization.
        queries_normed = self.norm1(queries)

        # Q from learned queries.
        Q = self.q_proj(queries_normed)

        # K,V from spectral-band tokens.
        K = self.k_proj(spectral_tokens)
        V = self.v_proj(spectral_tokens)

        # ------------------------------------------------------------
        # Multi-head reshape
        # ------------------------------------------------------------

        Q = rearrange(
            Q,
            "n m (h d) -> n h m d",
            h=self.num_heads,
        )

        K = rearrange(
            K,
            "n b (h d) -> n h b d",
            h=self.num_heads,
        )

        V = rearrange(
            V,
            "n b (h d) -> n h b d",
            h=self.num_heads,
        )

        # ------------------------------------------------------------
        # Spectral cross-attention
        # ------------------------------------------------------------

        scale = math.sqrt(self.head_dim)

        attn_weights = torch.matmul(
            Q,
            K.transpose(-2, -1),
        ) / scale

        # Shape:
        #     (N_pixels, heads, M, B)

        attn_weights = torch.softmax(
            attn_weights,
            dim=-1,
        )

        attn_weights = self.attn_dropout(
            attn_weights
        )

        # ------------------------------------------------------------
        # Weighted spectral features
        # ------------------------------------------------------------

        attn_output = torch.matmul(
            attn_weights,
            V,
        )

        # (N_pixels, heads, M, head_dim)
        # →
        # (N_pixels, M, d_model)

        attn_output = rearrange(
            attn_output,
            "n h m d -> n m (h d)",
        )

        attn_output = self.attn_out_proj(
            attn_output
        )

        # Residual connection.
        query_features = queries + attn_output

        # Transformer FFN.
        query_features = (
            query_features
            + self.ffn(
                self.norm2(query_features)
            )
        )

        return query_features, attn_weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert HSI cube into SAM2-compatible feature map.

        Args:
            x:
                (batch, B, H, W)

        Returns:
            (batch, d_model, H, W)
        """

        batch, B, H, W = x.shape

        assert B == self.num_bands, (
            f"Expected {self.num_bands} bands, got {B}. "
            "Check the dataset and adapter configuration."
        )

        # ------------------------------------------------------------
        # Rearrange pixels
        # ------------------------------------------------------------
        #
        # Each spatial pixel becomes an independent spectral sequence.
        #
        # (batch, B, H, W)
        #       ↓
        # (batch*H*W, B, 1)
        #

        pixels = rearrange(
            x,
            "b c h w -> (b h w) c 1",
        )

        total_pixels = pixels.shape[0]

        # Output buffer.
        output_features = torch.empty(
            total_pixels,
            self.d_model,
            device=x.device,
            dtype=x.dtype,
        )

        # ------------------------------------------------------------
        # Process pixels in chunks
        # ------------------------------------------------------------

        for start in range(
            0,
            total_pixels,
            self.pixel_chunk_size,
        ):

            end = min(
                start + self.pixel_chunk_size,
                total_pixels,
            )

            pixel_chunk = pixels[start:end]

            # --------------------------------------------------------
            # Spectral token embedding
            # --------------------------------------------------------

            spectral_tokens = self.spectral_proj(
                pixel_chunk
            )

            # Add learned spectral-position information.
            spectral_tokens = (
                spectral_tokens
                + self.spectral_pos
            )

            # --------------------------------------------------------
            # Spectral cross-attention
            # --------------------------------------------------------

            query_features, _ = self._cross_attention(
                spectral_tokens
            )

            # --------------------------------------------------------
            # Query aggregation
            # --------------------------------------------------------
            #
            # query_features:
            #     (N, M, d_model)
            #
            # Generate a query mixture for each pixel.
            #

            pooled = query_features.mean(
                dim=1
            )

            gate_logits = self.query_gate(
                pooled
            )

            gate = torch.softmax(
                gate_logits,
                dim=-1,
            )

            # Weighted combination of M query features.
            spatial_features = torch.sum(
                query_features
                * gate.unsqueeze(-1),
                dim=1,
            )

            output_features[start:end] = (
                spatial_features
            )

        # ------------------------------------------------------------
        # Restore spatial layout
        # ------------------------------------------------------------

        output = rearrange(
            output_features,
            "(b h w) d -> b d h w",
            b=batch,
            h=H,
            w=W,
        )

        # Final spatial refinement.
        output = self.output_proj(output)

        return output

    @torch.no_grad()
    def get_attention_maps(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return spectral attention maps.

        Args:
            x:
                (batch, B, H, W)

        Returns:
            Attention maps:
                (batch, num_heads, num_queries, B, H, W)

        The final dimension B corresponds to spectral bands.

        Therefore these maps can be analyzed to determine which
        wavelengths each learned query attends to.
        """

        self.eval()

        batch, B, H, W = x.shape

        assert B == self.num_bands

        pixels = rearrange(
            x,
            "b c h w -> (b h w) c 1",
        )

        all_attention = []

        for start in range(
            0,
            pixels.shape[0],
            self.pixel_chunk_size,
        ):

            end = min(
                start + self.pixel_chunk_size,
                pixels.shape[0],
            )

            pixel_chunk = pixels[start:end]

            spectral_tokens = self.spectral_proj(
                pixel_chunk
            )

            spectral_tokens = (
                spectral_tokens
                + self.spectral_pos
            )

            _, attn_weights = self._cross_attention(
                spectral_tokens
            )

            all_attention.append(
                attn_weights
            )

        attention = torch.cat(
            all_attention,
            dim=0,
        )

        #:
        # (B*H*W, heads, M, spectral_bands)
        #
        # →
        # (B, heads, M, spectral_bands, H, W)

        attention = rearrange(
            attention,
            "(b h w) nh m s -> b nh m s h w",
            b=batch,
            h=H,
            w=W,
        )

        return attention