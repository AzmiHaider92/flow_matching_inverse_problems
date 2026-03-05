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


class PatchFlowModel(nn.Module):
    def __init__(self, patch_size=7, cond_dim=512, num_classes=10):
        super().__init__()
        self.patch_dim = patch_size * patch_size

        # Embeddings (Keep your Fourier scale at 20.0 as you have it)
        self.class_emb = nn.Embedding(num_classes, cond_dim)
        self.pos_emb = FourierEmbedding(2, 128, scale=20.0)
        self.time_emb = FourierEmbedding(1, 64, scale=20.0)

        self.meta_net = nn.Sequential(
            nn.Linear(cond_dim + 128 + 64, cond_dim),
            nn.SiLU()
        )

        # Deeper backbone with Skip Connections
        self.in_layer = nn.Linear(self.patch_dim + cond_dim, 512)
        self.blocks = nn.Sequential(
            ResBlock(512),
            ResBlock(512)
        )
        self.out_layer = nn.Linear(512, self.patch_dim)
        self.act = nn.SiLU()

    def forward(self, x_t, t, coords, labels):
        c_e = self.class_emb(labels)
        p_e = self.pos_emb(coords)
        t_e = self.time_emb(t)

        meta = self.meta_net(torch.cat([c_e, p_e, t_e], dim=-1))

        h = self.act(self.in_layer(torch.cat([x_t, meta], dim=-1)))
        h = self.blocks(h)
        return self.out_layer(h)