import argparse

import torch.optim as optim
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
import os
from datetime import datetime
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
def visualize_grid(model, epoch, steps=25, PATCH_SIZE=7):
    model.eval()
    grid_res = image_size // PATCH_SIZE
    fig, axes = plt.subplots(3, 3, figsize=(8, 8))

    for idx in range(9):
        # Sample a digit 0-9
        label_val = torch.randint(0, 10, (1,)).to(device)
        full_img = torch.zeros((image_size, image_size)).to(device)

        for i in range(grid_res):
            for j in range(grid_res):
                y, x = i * PATCH_SIZE, j * PATCH_SIZE
                coords = torch.tensor([[x / (image_size-PATCH_SIZE), y / (image_size-PATCH_SIZE)]]).to(device)

                xt = torch.randn(1, PATCH_SIZE ** 2).to(device)
                dt = 1.0 / steps
                for s in range(steps):
                    t = torch.ones(1, 1).to(device) * (s / steps)
                    vt = model(xt, t, coords, label_val)
                    xt = xt + vt * dt

                full_img[y:y + PATCH_SIZE, x:x + PATCH_SIZE] = xt.view(PATCH_SIZE, PATCH_SIZE)

        ax = axes[idx // 3, idx % 3]
        ax.imshow(full_img.cpu().numpy(), cmap='gray', vmin=0, vmax=1)
        ax.set_title(f"Digit: {label_val.item()}")
        ax.axis('off')

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"epoch_{epoch:03d}.png")
    plt.savefig(save_path)
    plt.close()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--patch_size', type=int, default=7)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--fm_steps', type=int, default=25)
    parser.add_argument('--class_range', type=int, nargs='*', default=[0,10])#[10,47])

    return parser.parse_args()


# --- Main Logic ---
if __name__ == "__main__":
    config = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Setup Data
    dataset = EMNIST(train=True,class_range=config.class_range, device=device)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    # 2. Setup Model
    model = PatchFlowModel(patch_size=config.patch_size, num_classes=dataset.num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    print("Starting Training...")
    for epoch in range(config.n_epochs):
        model.train()
        total_loss = 0

        for batch_idx, (images, labels) in enumerate(loader):
            B = images.shape[0]

            # Sample random patch locations
            x_idx = torch.randint(0, image_size - config.patch_size + 1, (B,))
            y_idx = torch.randint(0, image_size - config.patch_size + 1, (B,))

            # Vectorized patch extraction
            patches = torch.stack([
                images[i, 0, y_idx[i]:y_idx[i] + config.patch_size, x_idx[i]:x_idx[i] + config.patch_size].flatten()
                for i in range(B)
            ]).to(device)

            coords = torch.stack([x_idx.float() / (image_size - config.patch_size),
                                  y_idx.float() / (image_size - config.patch_size)], dim=1).to(device)

            # Flow Matching
            x0 = torch.randn_like(patches)
            x1 = patches
            t = torch.rand(B, 1).to(device)

            xt = (1 - t) * x0 + t * x1
            ut = x1 - x0

            vt = model(xt, t, coords, labels)
            loss = torch.mean((vt - ut) ** 2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1} | Loss: {total_loss / len(loader):.4f}")

        # Visualize one sample 
        if (epoch + 1) % 10 == 0:
            visualize_grid(model, epoch, config.fm_steps, config.patch_size)
