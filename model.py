import torch
import torch.nn as nn
import numpy as np


class FourierEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels, scale=20.0):
        super().__init__()
        self.register_buffer('B', torch.randn(in_channels, out_channels // 2) * scale)

    def forward(self, x):
        x_proj = 2 * np.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class FullImageFlowModel(nn.Module):
    def __init__(self, cond_dim=512, num_classes=10):
        super().__init__()
        # num_classes + 1 to account for the "null" token at index 10
        self.class_emb = nn.Embedding(num_classes + 1, cond_dim)
        self.time_emb = FourierEmbedding(1, 64, scale=20.0)
        self.cond_proj = nn.Linear(cond_dim + 64, 128)

        # Dilated Backbone to fix the "Blob" problem by increasing receptive field
        self.net = nn.Sequential(
            nn.Conv2d(1 + 128, 64, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=2, dilation=2),  # Sees 5x5
            nn.SiLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=4, dilation=4),  # Sees 13x13
            nn.SiLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=6, dilation=6),  # Sees 25x25 (Almost full image)
            nn.SiLU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1)
        )

    def forward(self, x_t, t, labels):
        B, C, H, W = x_t.shape
        c_e = self.class_emb(labels)
        t_e = self.time_emb(t)

        cond = self.cond_proj(torch.cat([c_e, t_e], dim=-1))
        cond_map = cond.view(B, 128, 1, 1).expand(B, 128, H, W)

        return self.net(torch.cat([x_t, cond_map], dim=1))