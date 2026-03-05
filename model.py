import torch
import torch.nn as nn


# --- Model Definition ---
class PatchFlowModel(nn.Module):
    def __init__(self, patch_size=7, cond_dim=128, num_classes=10):  # Changed to 10
        super().__init__()
        self.patch_dim = patch_size * patch_size
        self.class_emb = nn.Embedding(num_classes, cond_dim)
        self.meta_projection = nn.Linear(3 + cond_dim, cond_dim)

        self.net = nn.Sequential(
            nn.Linear(self.patch_dim + cond_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.patch_dim)
        )

    def forward(self, x_t, t, coords, labels):
        c_emb = self.class_emb(labels)
        meta_input = torch.cat([t, coords, c_emb], dim=-1)
        meta = torch.relu(self.meta_projection(meta_input))
        return self.net(torch.cat([x_t, meta], dim=-1))