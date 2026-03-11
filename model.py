import torch
import torch.nn as nn

import torch
import torch.nn as nn
import numpy as np


class FourierEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels, scale=20.0):
        super().__init__()
        # Fixed random frequencies to project low-dim input to high-dim
        self.register_buffer('B', torch.randn(in_channels, out_channels // 2) * scale)

    def forward(self, x):
        # x: [Batch, in_channels]
        x_proj = 2 * np.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        return x + self.net(x)


class FullImageFlowModel(nn.Module):
    def __init__(self, cond_dim=512, num_classes=10):
        super().__init__()
        # Embeddings for class and time
        self.class_emb = nn.Embedding(num_classes, cond_dim)
        self.time_emb = FourierEmbedding(1, 64, scale=20.0)

        # Projection for conditioning
        self.cond_proj = nn.Linear(cond_dim + 64, 128)

        # A simple Convolutional Encoder-Decoder (No patches!)
        self.net = nn.Sequential(
            nn.Conv2d(1 + 1, 64, kernel_size=3, padding=1),  # +1 for time/cond channel
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1)  # Output velocity [B, 1, 28, 28]
        )

    def forward(self, x_t, t, labels):
        # x_t: [B, 1, 28, 28]
        B, C, H, W = x_t.shape

        # Merge conditioning into a spatial map
        c_e = self.class_emb(labels)
        t_e = self.time_emb(t)
        cond = self.cond_proj(torch.cat([c_e, t_e], dim=-1))  # [B, 128]

        # Broadcast conditioning to match image size [B, 1, 28, 28]
        cond_map = cond[:, :1].view(B, 1, 1, 1).expand(B, 1, H, W)

        # Concatenate image and condition channel
        input_tensor = torch.cat([x_t, cond_map], dim=1)

        return self.net(input_tensor)