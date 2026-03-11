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
        self.class_emb = nn.Embedding(num_classes, cond_dim)
        self.time_emb = FourierEmbedding(1, 64, scale=20.0)

        # Project 576 -> 64 to use as conditioning channels
        self.cond_proj = nn.Linear(cond_dim + 64, 64)

        # A Mini-UNet style architecture
        self.down1 = nn.Conv2d(1 + 64, 64, kernel_size=3, padding=1)
        self.down2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)  # 14x14

        self.mid = ResBlock(128)  # You'll need to update ResBlock to use Conv2d

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # 28x28
        self.out = nn.Conv2d(64 + 64, 1, kernel_size=3, padding=1)

    def forward(self, x_t, t, labels):
        B, _, H, W = x_t.shape

        # Fix the conditioning: Use all dimensions!
        c_e = self.class_emb(labels)
        t_e = self.time_emb(t)
        cond = self.cond_proj(torch.cat([c_e, t_e], dim=-1))  # [B, 64]
        cond_map = cond.view(B, 64, 1, 1).expand(B, 64, H, W)

        # Encoder
        x = torch.cat([x_t, cond_map], dim=1)
        feat1 = torch.relu(self.down1(x))
        feat2 = torch.relu(self.down2(feat1))

        # Bottleneck (Global context)
        # Note: If your ResBlock is Linear, swap it for a simple Conv block here
        mid = torch.relu(self.mid(feat2)) if hasattr(self, 'mid') else feat2

        # Decoder with Skip Connection
        up = torch.relu(self.up1(mid))
        out = self.out(torch.cat([up, cond_map], dim=1))
        return out