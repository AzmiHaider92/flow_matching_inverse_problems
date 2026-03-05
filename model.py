import torch
import torch.nn as nn

import torch
import torch.nn as nn
import numpy as np


class FourierEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels, scale=10.0):
        super().__init__()
        # Fixed random frequencies to project low-dim input to high-dim
        self.register_buffer('B', torch.randn(in_channels, out_channels // 2) * scale)

    def forward(self, x):
        # x: [Batch, in_channels]
        x_proj = 2 * np.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class PatchFlowModel(nn.Module):
    def __init__(self, patch_size=7, cond_dim=256, num_classes=10):
        super().__init__()
        self.patch_dim = patch_size * patch_size

        # 1. Improved Embeddings
        self.class_emb = nn.Embedding(num_classes, cond_dim)
        self.pos_emb = FourierEmbedding(2, 64)  # For (x, y)
        self.time_emb = FourierEmbedding(1, 64)  # For t

        # Projection for the metadata
        self.meta_net = nn.Sequential(
            nn.Linear(cond_dim + 64 + 64, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim)
        )

        # 2. ResNet-style Backbone for the Vector Field
        # We use SiLU (Swish) as it's standard for generative flows
        self.fc1 = nn.Linear(self.patch_dim + cond_dim, 512)
        self.fc2 = nn.Linear(512 + cond_dim, 512)  # Skip connection from meta
        self.fc3 = nn.Linear(512, self.patch_dim)
        self.act = nn.SiLU()

    def forward(self, x_t, t, coords, labels):
        # x_t: [B, 49], t: [B, 1], coords: [B, 2], labels: [B]

        # Embed metadata
        c_e = self.class_emb(labels)
        p_e = self.pos_emb(coords)
        t_e = self.time_emb(t)

        meta = self.meta_net(torch.cat([c_e, p_e, t_e], dim=-1))  # [B, cond_dim]

        # First layer
        h = self.act(self.fc1(torch.cat([x_t, meta], dim=-1)))

        # Second layer with Skip Connection (Injection of meta again)
        h = self.act(self.fc2(torch.cat([h, meta], dim=-1)))

        # Output layer
        return self.fc3(h)