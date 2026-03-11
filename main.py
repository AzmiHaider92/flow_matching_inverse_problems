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
from model import FullImageFlowModel

image_size = 28


@torch.no_grad()
def visualize_grid(model, loader, output_path, steps=25, patch_size=7, device='cpu'):
    model.eval()
    n_rows = 4
    fig, axes = plt.subplots(n_rows, 3, figsize=(10, 12))

    # Get a batch of real data
    images, labels = next(iter(loader))
    images, labels = images[:n_rows].to(device), labels[:n_rows].to(device)
    B = images.shape[0]

    # 1. Generate Random Patch Locations for the visualization
    max_off = images.shape[-1] - patch_size
    x_idx = torch.randint(0, max_off + 1, (B,))
    y_idx = torch.randint(0, max_off + 1, (B,))

    # --- COLUMN 2: Create the "Patch Image" (Masked GT) ---
    # Start with a black canvas and paste only the ground truth patch
    patch_images = torch.zeros_like(images)
    for i in range(B):
        y, x = y_idx[i], x_idx[i]
        patch_images[i, 0, y:y + patch_size, x:x + patch_size] = \
            images[i, 0, y:y + patch_size, x:x + patch_size]

    # --- COLUMN 3: Generate the Predicted Image (ODE Solve) ---
    # Start from pure noise
    x_hat = torch.randn_like(images)
    dt = 1.0 / steps

    for s in range(steps):
        # Time goes from 1 (noise) to 0 (clean)
        t = torch.ones(B, 1).to(device) * (1.0 - s * dt)

        # Predict velocity
        vt = model(x_hat, t, labels)

        # Euler step: x_{t-dt} = x_t - vt * dt
        x_hat = x_hat - vt * dt

    # --- Plotting ---
    for i in range(n_rows):
        # Col 1: Ground Truth
        axes[i, 0].imshow(images[i, 0].cpu(), cmap='gray', vmin=0, vmax=1)
        axes[i, 0].set_title("Ground Truth")

        # Col 2: The Patch (Observation)
        axes[i, 1].imshow(patch_images[i, 0].cpu(), cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title(f"Patch (y) at {x_idx[i].item()},{y_idx[i].item()}")

        # Col 3: Model Prediction
        axes[i, 2].imshow(x_hat[i, 0].cpu(), cmap='gray', vmin=0, vmax=1)
        axes[i, 2].set_title("Generated X")

    for ax in axes.flatten():
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def f_project(X, y_idx, x_idx, patch_size):
    B = X.shape[0]
    patches = []
    for i in range(B):
        # Use the passed-in indices instead of random ones
        patch = X[i, 0, y_idx[i]:y_idx[i] + patch_size, x_idx[i]:x_idx[i] + patch_size]
        patches.append(patch.reshape(-1))
    return torch.stack(patches)


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
    # --- ADD THESE TO YOUR get_args() function ---
    parser.add_argument('--guide_steps', type=int, default=10, help="Steps for guided sampling")
    parser.add_argument('--eta', type=float, default=0.1, help="Guidance scale (GPS strength)")

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
        run_dir = os.path.join("outputs", f"experiment_mnist_patch{config.patch_size}_{timestamp}")
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
    model = FullImageFlowModel(num_classes=dataset.num_classes).to(device)

    if config.ckpt_path:
        print(f"Loading checkpoint from {config.ckpt_path}")
        # Using weights_only=True is a modern security best practice for torch.load
        model.load_state_dict(torch.load(config.ckpt_path, map_location=device))

    # --- EVALUATION MODE ---
    if config.mode == 'eval':
        print("Running Evaluation...")
        # Add a unique suffix so we don't overwrite training images
        timestamp_eval = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_name = f"eval_{timestamp_eval}.png"

        visualize_grid(model, loader, out_name, steps=config.guide_steps,
                       patch_size=config.patch_size, device=device)
        print(f"Evaluation image saved to: {os.path.join(run_dir, out_name)}")

    # --- TRAINING MODE (IGFM) ---
    else:
        optimizer = optim.Adam(model.parameters(), lr=config.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.n_epochs,
                                                         eta_min=config.lr * config.lr_factor)

        for epoch in range(config.n_epochs):
            model.train()
            total_loss = 0

            for images, labels in loader:
                B = images.shape[0]
                images, labels = images.to(device), labels.to(device)

                # --- FIX: Sample indices ONCE here ---
                max_off = image_size - config.patch_size
                x_idx = torch.randint(0, max_off + 1, (B,), device=device)
                y_idx = torch.randint(0, max_off + 1, (B,), device=device)

                # Use these fixed indices for both real and hallucinated projections
                y_obs = f_project(images, y_idx, x_idx, config.patch_size)

                # --- PHASE 1: GUIDED SAMPLING ---
                model.eval()
                with torch.no_grad():
                    x_hat = torch.randn(B, 1, image_size, image_size).to(device)
                    dt = 1.0 / config.guide_steps

                    for s in range(config.guide_steps):
                        t_curr = torch.ones(B, 1).to(device) * (1.0 - s * dt)
                        vt = model(x_hat, t_curr, labels)
                        x_hat = x_hat - vt * dt

                        with torch.enable_grad():
                            x_hat.requires_grad_(True)
                            # Pass the SAME indices here
                            prediction_y = f_project(x_hat, y_idx, x_idx, config.patch_size)
                            guidance_loss = torch.sum((prediction_y - y_obs) ** 2)
                            grad = torch.autograd.grad(guidance_loss, x_hat)[0]
                            x_hat = x_hat.detach() - config.eta * grad

                # --- PHASE 2: FLOW MATCHING (The "Study") ---
                model.train()
                # The model now trains to generate the FULL image it just hallucinated
                x1 = x_hat.detach()
                x0 = torch.randn_like(x1)
                # 1. Reshape t for broadcasting: [B] -> [B, 1, 1, 1]
                t = torch.rand(B, device=device).unsqueeze(-1)
                t_view = t.view(B, 1, 1, 1)

                # 2. Standard Flow: xt = (1 - t)*x1 + t*x0
                # At t=0, xt = x1 (Clean)
                # At t=1, xt = x0 (Noise)
                xt = (1 - t_view) * x1 + t_view * x0

                # 3. Target Velocity: The vector pointing FROM Noise TO Image
                # Since we are solving from t=1 down to t=0:
                ut = x1 - x0

                # 4. Training Loss
                vt_pred = model(xt, t, labels)
                loss = torch.mean((vt_pred - ut) ** 2)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            scheduler.step()
            avg_loss = total_loss / len(loader)
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch + 1:03d} | Loss: {avg_loss:.6f} | LR: {current_lr:.2e}")

            if (epoch + 1) % config.vis_every == 0:
                img_path = os.path.join(run_dir, f"epoch_{epoch + 1:03d}.png")
                visualize_grid(model, loader, img_path, steps=config.guide_steps,
                               patch_size=config.patch_size, device=device)
                torch.save(model.state_dict(), os.path.join(run_dir, "last_model.pt"))
                print(f"Eval saved to: {run_dir}")

        print(f"Results saved to: {run_dir}")
