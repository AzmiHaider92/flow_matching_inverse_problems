import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from datetime import datetime

# Local imports
from data import EMNIST
from model import PatchFlowModel

image_size = 28


@torch.no_grad()
def visualize_grid(model, epoch, num_classes, output_path, steps=25, PATCH_SIZE=7, device='cpu', overlap=True):
    model.eval()
    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    stride = 2 if overlap else PATCH_SIZE

    for idx in range(9):
        label_val = torch.randint(0, num_classes, (1,)).to(device)
        canvas = torch.zeros((image_size, image_size)).to(device)
        counts = torch.zeros((image_size, image_size)).to(device)

        # Use Global Noise Map to ensure structural consistency
        global_noise = torch.randn(1, 1, image_size, image_size).to(device)

        if overlap:
            y_m, x_m = torch.meshgrid(torch.linspace(-1, 1, PATCH_SIZE), torch.linspace(-1, 1, PATCH_SIZE),
                                      indexing='ij')
            mask = torch.exp(-(x_m ** 2 + y_m ** 2) / 0.5).to(device)
        else:
            mask = torch.ones((PATCH_SIZE, PATCH_SIZE)).to(device)

        for y_start in range(0, image_size - PATCH_SIZE + 1, stride):
            for x_start in range(0, image_size - PATCH_SIZE + 1, stride):
                coords = torch.tensor([[x_start / (image_size - PATCH_SIZE),
                                        y_start / (image_size - PATCH_SIZE)]]).to(device).float()

                x0 = global_noise[:, :, y_start:y_start + PATCH_SIZE, x_start:x_start + PATCH_SIZE].reshape(1, -1)

                xt = x0
                dt = 1.0 / steps
                for s in range(steps):
                    t = torch.ones(1, 1).to(device) * (s / steps)
                    vt = model(xt, t, coords, label_val)
                    xt = xt + vt * dt

                patch = xt.view(PATCH_SIZE, PATCH_SIZE) * mask
                canvas[y_start:y_start + PATCH_SIZE, x_start:x_start + PATCH_SIZE] += patch
                counts[y_start:y_start + PATCH_SIZE, x_start:x_start + PATCH_SIZE] += mask

        full_img = canvas / (counts + 1e-8)
        ax = axes[idx // 3, idx % 3]
        ax.imshow(full_img.cpu().numpy(), cmap='gray', vmin=0, vmax=1)
        ax.set_title(f"Class: {label_val.item()}")
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['train', 'eval'], default='train')
    parser.add_argument('--ckpt_path', type=str, default=None, help="Path to load weights for eval or resume")
    parser.add_argument('--n_epochs', type=int, default=500)
    parser.add_argument('--vis_every', type=int, default=50)
    parser.add_argument('--patch_size', type=int, default=7)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--lr_factor', type=float, default=0.1)

    parser.add_argument('--fm_steps', type=int, default=64)
    parser.add_argument('--class_range', type=int, nargs='*', default=[0, 10])
    parser.add_argument('--overlap', action='store_true', help="Use overlapping patches in visualization")
    return parser.parse_args()


if __name__ == "__main__":
    config = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Directory Setup
    if config.mode == 'train':
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = os.path.join("outputs", f"experiment_mnist_patch{config.patch_size}_overlap{config.overlap}_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)
    else:
        if config.ckpt_path is None:
            raise ValueError("You must provide --ckpt_path in eval mode.")
        # Save eval images in the same folder as the checkpoint
        run_dir = os.path.dirname(config.ckpt_path)
        print(f"Eval mode: Results will be saved in {run_dir}")

    # 2. Data & Model Setup
    dataset = EMNIST(train=(config.mode == 'train'), class_range=config.class_range, device=device)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    model = PatchFlowModel(patch_size=config.patch_size, num_classes=dataset.num_classes).to(device)

    if config.ckpt_path:
        print(f"Loading checkpoint from {config.ckpt_path}")
        # Using weights_only=True is a modern security best practice for torch.load
        model.load_state_dict(torch.load(config.ckpt_path, map_location=device))

    # --- EVALUATION MODE ---
    if config.mode == 'eval':
        print("Running Evaluation...")
        # Add a unique suffix so we don't overwrite training images
        timestamp_eval = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_name = f"eval_overlap_{config.overlap}_{timestamp_eval}.png"

        visualize_grid(model, 0, dataset.num_classes, os.path.join(run_dir, out_name),
                       config.fm_steps, config.patch_size, device, overlap=config.overlap)
        print(f"Evaluation image saved to: {os.path.join(run_dir, out_name)}")

    # --- TRAINING MODE ---
    else:
        optimizer = optim.Adam(model.parameters(), lr=config.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.n_epochs, eta_min=config.lr * config.lr_factor)

        for epoch in range(config.n_epochs):
            model.train()
            total_loss = 0
            for images, labels in loader:
                B = images.shape[0]
                max_off = image_size - config.patch_size
                x_idx, y_idx = torch.randint(0, max_off + 1, (B,)), torch.randint(0, max_off + 1, (B,))

                all_p = images.unfold(2, config.patch_size, 1).unfold(3, config.patch_size, 1)
                patches = all_p[torch.arange(B), 0, y_idx, x_idx].reshape(B, -1).to(device)
                coords = torch.stack([x_idx.float() / max_off, y_idx.float() / max_off], dim=1).to(device)

                x0, x1 = torch.randn_like(patches), patches
                t = torch.rand(B, 1).to(device)
                xt = (1 - t) * x0 + t * x1
                ut = x1 - x0

                vt = model(xt, t, coords, labels)
                loss = torch.mean((vt - ut) ** 2)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            scheduler.step()
            avg_loss = total_loss / len(loader)
            print(f"Epoch {epoch + 1:03d} | Loss: {avg_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

            if (epoch + 1) % config.vis_every == 0:
                # Save Visualization
                img_path = os.path.join(run_dir, f"epoch_{epoch + 1:03d}.png")
                visualize_grid(model, epoch + 1, dataset.num_classes, img_path,
                               config.fm_steps, config.patch_size, device, overlap=config.overlap)

                # Save Checkpoint
                ckpt_path = os.path.join(run_dir, "last_model.pt")
                torch.save(model.state_dict(), ckpt_path)

        print(f"Results saved to: {run_dir}")
