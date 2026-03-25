"""DINOv3 (ViT-B/16) frozen visual encoder via TIMM."""

import torch
import torch.nn as nn
import timm


class DINOv3Encoder(nn.Module):
    """Frozen DINOv3 ViT-B/16 encoder that extracts patch tokens from images.

    Output: (B, 256, 768) — 256 patch tokens of dimension 768.
    The first 5 prefix tokens (1 CLS + 4 register) are stripped.
    """

    def __init__(self, img_size: int = 256):
        super().__init__()
        self.model = timm.create_model(
            "vit_base_patch16_dinov3.lvd1689m",
            pretrained=True,
            img_size=img_size,
        )
        self.model.eval()
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        self.img_size = img_size
        self.embed_dim = 768
        self.num_patches = 256  # (256/16)^2 = 16*16
        self.num_prefix_tokens = 5  # 1 CLS + 4 register

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract patch tokens from images.

        Args:
            x: (B, 3, H, W) — raw images (will be resized if needed).

        Returns:
            (B, 256, 768) — patch tokens only (prefix tokens stripped).
        """
        with torch.no_grad():
            # forward_features returns (B, num_prefix + num_patches, embed_dim)
            features = self.model.forward_features(x)
            # Strip prefix tokens (CLS + registers)
            patch_tokens = features[:, self.num_prefix_tokens:, :]
        return patch_tokens

    def train(self, mode=True):
        # Always stay in eval mode
        super().train(False)
        return self
