import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from datetime import datetime

# Assuming these are in your local directory
from data import EMNIST
from model import PatchFlowModel

# --- Setup Experiment Directory ---
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
exp_name = f"experiment_mnist__{timestamp}"
output_dir = os.path.join("outputs", exp_name)
os.makedirs(output_dir, exist_ok=True)

image_size = 28


# --- 3x3 Visualization Function ---
@torch.no_grad()
def visualize_grid(model, epoch, num_classes, steps=25, PATCH_SIZE=7, device='cpu'):
    model.eval()
    grid_res = image_size // PATCH_SIZE
    fig, axes = plt.subplots(3, 3, figsize=(8, 8))

    for idx in range(9):
        # Sample a random class from the available range
        label_val = torch.randint(0, num_classes, (1,)).to(device)
        full_img = torch.zeros((image_size, image_size)).to(device)

        for i in range(grid_res):
            for j in range(grid_res):
                y, x = i * PATCH_SIZE, j * PATCH_SIZE
                # Normalized coordinates
                coords = torch.tensor([[x / (image_size - PATCH_SIZE),
                                        y / (image_size - PATCH_SIZE)]]).to(device).float()

                # ODE Solve (Euler)
                xt = torch.randn(1, PATCH_SIZE ** 2).to(device)
                dt = 1.0 / steps
                for s in range(steps):
                    t = torch.ones(1, 1).to(device) * (s / steps)
                    vt = model(xt, t, coords, label_val)
                    xt = xt + vt * dt

                full_img[y:y + PATCH_SIZE, x:x + PATCH_SIZE] = xt.view(PATCH_SIZE, PATCH_SIZE)

        ax = axes[idx // 3, idx % 3]
        ax.imshow(full_img.cpu().numpy(), cmap='gray', vmin=0, vmax=1)
        ax.set_title(f"Class: {label_val.item()}")
        ax.axis('off')

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"epoch_{epoch:03d}.png")
    print(f"Saved at: {exp_name}")
    plt.savefig(save_path)
    plt.close()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--patch_size', type=int, default=7)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--fm_steps', type=int, default=25)
    parser.add_argument('--class_range', type=int, nargs='*', default=[0, 10])
    return parser.parse_args()


# --- Main Logic ---
if __name__ == "__main__":
    config = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Setup Data
    dataset = EMNIST(train=True, class_range=config.class_range, device=device)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    # 2. Setup Model & Optimizer
    model = PatchFlowModel(patch_size=config.patch_size, num_classes=dataset.num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    # Adding a scheduler to drop LR by half if loss doesn't improve for 5 epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    print(f"Starting Training on {device}...")
    print(f"Results will be saved to: {output_dir}")

    for epoch in range(config.n_epochs):
        model.train()
        total_loss = 0

        for images, labels in loader:
            B = images.shape[0]

            # Sample random patch locations
            max_offset = image_size - config.patch_size
            x_idx = torch.randint(0, max_offset + 1, (B,))
            y_idx = torch.randint(0, max_offset + 1, (B,))

            # --- OPTIMIZED VECTORIZED PATCH EXTRACTION ---
            # Unfold creates a view of all possible patches: [B, 1, H_patches, W_patches, P, P]
            all_possible = images.unfold(2, config.patch_size, 1).unfold(3, config.patch_size, 1)
            # Select the patches using our random indices
            patches = all_possible[torch.arange(B), 0, y_idx, x_idx].reshape(B, -1).to(device)
            # ----------------------------------------------

            coords = torch.stack([x_idx.float() / max_offset,
                                  y_idx.float() / max_offset], dim=1).to(device)

            # Flow Matching: Linear Interpolation
            x0 = torch.randn_like(patches)
            x1 = patches
            t = torch.rand(B, 1).to(device)

            # xt = (1-t)x0 + tx1
            xt = (1 - t) * x0 + t * x1
            ut = x1 - x0  # Target velocity

            vt = model(xt, t, coords, labels)
            loss = torch.mean((vt - ut) ** 2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        scheduler.step(avg_loss)

        print(f"Epoch {epoch + 1:03d} | Loss: {avg_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Visualize periodically
        if (epoch + 1) % 10 == 0 or epoch == 0:
            visualize_grid(model, epoch + 1, dataset.num_classes, config.fm_steps, config.patch_size, device)