"""Causal ViT Predictor for DINO-WM, adapted from dino_wm/models/vit.py.

Key changes vs upstream:
  - dim 384 -> 768 to match DINOv3 ViT-B/16
  - Attention uses F.scaled_dot_product_attention (Flash-Attention v2 on A100 BF16),
    avoiding materialization of the (B, heads, T, T) score matrix. Critical here
    because T = num_frames * num_patches = 3 * 770 = 2310 makes a naive
    matmul ~80 GiB at batch=256.
"""

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange


def generate_mask_matrix(npatch, nwindow):
    """Generate a block-causal attention mask.

    Each frame of npatch tokens can attend to itself and all previous frames,
    but not to future frames.
    """
    zeros = torch.zeros(npatch, npatch)
    ones = torch.ones(npatch, npatch)
    rows = []
    for i in range(nwindow):
        row = torch.cat([ones] * (i + 1) + [zeros] * (nwindow - i - 1), dim=1)
        rows.append(row)
    mask = torch.cat(rows, dim=0).unsqueeze(0).unsqueeze(0)
    return mask


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, num_patches=1, num_frames=1):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attn_dropout_p = float(dropout)   # SDPA takes a scalar p

        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

        # Pre-compute block-causal mask (1, 1, T_max, T_max). Bool: True = attend.
        # F.scaled_dot_product_attention's `attn_mask` semantics for bool tensors:
        #   True  -> participate in attention
        #   False -> masked out
        # generate_mask_matrix returns 1.0 where attention is allowed; cast to bool.
        self.register_buffer(
            "bias",
            generate_mask_matrix(num_patches, num_frames).bool(),
            persistent=False,
        )

    def forward(self, x):
        B, T, _ = x.size()
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        # Flash Attention v2 path on A100 BF16; falls back to memory-efficient
        # attention otherwise. Either way, the (B, h, T, T) score matrix is NOT
        # materialized — memory becomes O(B*h*T*d) instead of O(B*h*T^2).
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=self.bias[:, :, :T, :T],
            dropout_p=self.attn_dropout_p if self.training else 0.0,
            scale=self.scale,
        )
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0,
                 num_patches=1, num_frames=1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout,
                                  num_patches=num_patches, num_frames=num_frames),
                        FeedForward(dim, mlp_dim, dropout=dropout),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class ViTPredictor(nn.Module):
    """Causal Vision Transformer predictor for world model dynamics.

    Predicts next-frame patch tokens given a sequence of frames, each
    represented as (num_patches_per_frame) tokens. Uses block-causal
    attention so each frame can only attend to current + past frames.

    Args:
        num_patches: Number of tokens per frame (visual patches + proprio + action).
        num_frames: Number of frames in the input sequence.
        dim: Token embedding dimension (768 for DINOv3).
        depth: Number of transformer layers.
        heads: Number of attention heads.
        mlp_dim: Feed-forward hidden dimension.
        dim_head: Dimension per attention head.
        dropout: Dropout rate.
        emb_dropout: Embedding dropout rate.
    """

    def __init__(
        self,
        *,
        num_patches,
        num_frames,
        dim,
        depth,
        heads,
        mlp_dim,
        pool="cls",
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        assert pool in {"cls", "mean"}

        self.num_patches = num_patches
        self.num_frames = num_frames

        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_frames * num_patches, dim)
        )
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            dim, depth, heads, dim_head, mlp_dim, dropout,
            num_patches=num_patches, num_frames=num_frames,
        )
        self.pool = pool

    def forward(self, x):
        """Forward pass.

        Args:
            x: (B, num_frames * num_patches, dim)

        Returns:
            (B, num_frames * num_patches, dim) — predicted tokens.
        """
        b, n, _ = x.shape
        x = x + self.pos_embedding[:, :n]
        x = self.dropout(x)
        x = self.transformer(x)
        return x
