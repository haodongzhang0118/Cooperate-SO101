"""Proprioceptive/Action embedding via Conv1d, adapted from dino_wm/models/proprio.py.

Key change: emb_dim 384 -> 768 to match DINOv3.
"""

import torch.nn as nn


class ProprioceptiveEmbedding(nn.Module):
    """Embeds proprioceptive state or action vectors into the token space.

    Uses a 1D convolution to project from input dimension to embedding dimension.

    Args:
        num_frames: Number of temporal frames.
        tubelet_size: Kernel size / stride for temporal convolution (1 = per-frame).
        in_chans: Input channel dimension (e.g., 6 for single-arm joints).
        emb_dim: Output embedding dimension (768 for DINOv3).
    """

    def __init__(self, num_frames=16, tubelet_size=1, in_chans=8, emb_dim=768):
        super().__init__()
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.in_chans = in_chans
        self.emb_dim = emb_dim

        self.patch_embed = nn.Conv1d(
            in_chans, emb_dim, kernel_size=tubelet_size, stride=tubelet_size
        )

    def forward(self, x):
        """Embed proprioceptive/action vectors.

        Args:
            x: (B, T, D) where D = in_chans.

        Returns:
            (B, T, emb_dim)
        """
        # (B, T, D) -> (B, D, T) -> Conv1d -> (B, emb_dim, T) -> (B, T, emb_dim)
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        return x
