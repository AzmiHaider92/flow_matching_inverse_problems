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

        # This maps the 512 + 64 features into a shape we can inject into the image
        self.cond_proj = nn.Linear(cond_dim + 64, 128)

        # Backbone
        self.net = nn.Sequential(
            # Input: 1 (image) + 128 (condition channels) = 129
            nn.Conv2d(1 + 128, 64, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1)
        )

    def forward(self, x_t, t, labels):
        # x_t: [B, 1, 28, 28]
        B, C, H, W = x_t.shape

        # 1. Get Embeddings: result is [B, 576]
        c_e = self.class_emb(labels)
        t_e = self.time_emb(t)

        # 2. Project conditioning: result is [B, 128]
        # This is where your mat1/mat2 error was likely happening.
        # Ensure we are only passing the 2D [B, 576] tensor here.
        cond = self.cond_proj(torch.cat([c_e, t_e], dim=-1))

        # 3. Broadcast to Spatial: [B, 128, 28, 28]
        # We turn the vector into a "stack of maps" so the Conv layer can read it
        cond_map = cond.view(B, 128, 1, 1).expand(B, 128, H, W)

        # 4. Concatenate and run ConvNet
        # Input to Conv2d is now [B, 129, 28, 28]
        input_tensor = torch.cat([x_t, cond_map], dim=1)

        return self.net(input_tensor)