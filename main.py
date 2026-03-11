import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from datetime import datetime
from data import EMNIST
from model import FullImageFlowModel
import torch.nn.functional as F

image_size = 28


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

    parser.add_argument('--fm_steps', type=int, default=100)
    parser.add_argument('--class_range', type=int, nargs='*', default=[0, 10])
    parser.add_argument('--overlap', action='store_true', help="Use overlapping patches in visualization")
    return parser.parse_args()


def get_foreground_indices(images, patch_size):
    B, _, H, W = images.shape
    device = images.device
    mask = (images.view(B, -1) > 0.1).float()
    # Pick foreground pixels
    indices = torch.multinomial(mask + 1e-8, 1).squeeze(1)
    py, px = indices // W, indices % W
    # Add random offsets
    off_y = torch.randint(-patch_size + 1, 1, (B,), device=device)
    off_x = torch.randint(-patch_size + 1, 1, (B,), device=device)
    y_idx = torch.clamp(py + off_y, 0, H - patch_size)
    x_idx = torch.clamp(px + off_x, 0, W - patch_size)
    return y_idx.long(), x_idx.long()


def f_project(X, y_idx, x_idx, patch_size):
    B, C, H, W = X.shape
    patches = F.unfold(X, kernel_size=patch_size)
    stride_w = W - patch_size + 1
    linear_indices = y_idx * stride_w + x_idx
    linear_indices = linear_indices.view(B, 1, 1).expand(-1, patch_size ** 2, -1)
    return patches.gather(2, linear_indices).squeeze(-1)


def get_cfg_velocity(model, x, t, labels, cfg_scale=3.0, null_label=10):
    v_cond = model(x, t, labels)
    if cfg_scale == 1.0: return v_cond
    v_uncond = model(x, t, torch.full_like(labels, null_label))
    return v_uncond + cfg_scale * (v_cond - v_uncond)


@torch.no_grad()
def visualize_grid(model, loader, output_path, config, device='cpu'):
    model.eval()
    n_rows = 4
    fig, axes = plt.subplots(n_rows, 6, figsize=(20, 10))
    images, labels = next(iter(loader))
    images, labels = images[:n_rows].to(device), labels[:n_rows].to(device)
    B = images.shape[0]

    y_idx, x_idx = get_foreground_indices(images, config.patch_size)
    y_obs = f_project(images, y_idx, x_idx, config.patch_size)

    # Col 1: GT | Col 2: Patch Viz
    patch_viz = torch.zeros_like(images)
    for i in range(B):
        patch_viz[i, 0, y_idx[i]:y_idx[i] + config.patch_size, x_idx[i]:x_idx[i] + config.patch_size] = \
            y_obs[i].view(config.patch_size, config.patch_size)

    # Col 3: Prior Check (Standard CFG on GT Label)
    x_prior = torch.randn_like(images)
    dt = 1.0 / config.fm_steps
    for s in range(config.fm_steps):
        t = torch.ones(B, 1, device=device) * (s * dt)
        x_prior = x_prior + get_cfg_velocity(model, x_prior, t, labels, cfg_scale=4.0) * dt

    # Col 4, 5, 6: Diversity Dreams (Null Label 10 + Random Start)
    dream_results = []
    null_labels = torch.full_like(labels, 10)
    for _ in range(3):
        x_dream = torch.randn_like(images)
        for s in range(config.fm_steps):
            t_val = s / config.fm_steps
            t = torch.ones(B, 1, device=device) * t_val

            # Use the NULL label to allow the model to choose its own digit path
            vt = model(x_dream, t, null_labels)
            x_dream = x_dream + vt * dt

            # ANNEALED GUIDANCE: Strong at start, zero at end to stop "hairs"
            current_eta = config.eta * (1.0 - t_val)
            with torch.enable_grad():
                x_dream.requires_grad_(True)
                y_hat = f_project(x_dream, y_idx, x_idx, config.patch_size)
                loss_g = F.mse_loss(y_hat, y_obs) * 0.5 * (config.patch_size ** 2)
                grad = torch.autograd.grad(loss_g, x_dream)[0]
                x_dream = x_dream.detach() - current_eta * grad
        dream_results.append(x_dream)

    imgs = [images, patch_viz] + dream_results + [x_prior]
    titles = ["GT", "Patch", "S1", "S2", "S3", "Prior"]
    for r in range(n_rows):
        for c in range(6):
            axes[r, c].imshow(imgs[c][r, 0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
            if r == 0: axes[r, c].set_title(titles[c])
            axes[r, c].axis('off')
    plt.tight_layout()
    plt.savefig(output_path);
    plt.close()


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
        img_path = os.path.join(run_dir, out_name)
        visualize_grid(model, loader, img_path, config, device=device)

        print(f"Evaluation image saved to: {img_path}")

    # --- TRAINING MODE (IGFM) ---
    else:
        optimizer = optim.Adam(model.parameters(), lr=config.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.n_epochs,
                                                         eta_min=config.lr * config.lr_factor)

        for epoch in range(config.n_epochs):
            model.train()
            total_loss = 0
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                B = images.shape[0]

                # CFG Label Dropout (10% chance)
                drop_mask = torch.rand(B, device=device) < 0.1
                train_labels = torch.where(drop_mask, torch.tensor(10, device=device), labels)

                # IGFM Indices
                y_idx, x_idx = get_foreground_indices(images, config.patch_size)
                y_obs = f_project(images, y_idx, x_idx, config.patch_size)

                # --- PHASE 1: Dream (Tuned for 7x7 Patch) ---
                model.eval()
                with torch.no_grad():
                    x_hat = torch.randn_like(images)
                    dt = 1.0 / config.guide_steps
                    for s in range(config.guide_steps):
                        t_val = s * dt
                        t_c = torch.ones(B, 1, device=device) * t_val

                        # 1. Standard Velocity
                        x_hat = x_hat + model(x_hat, t_c, labels) * dt

                        with torch.enable_grad():
                            x_hat.requires_grad_(True)
                            y_h = f_project(x_hat, y_idx, x_idx, config.patch_size)

                            # Use SUM for a high-energy signal
                            g_loss = torch.sum((y_h - y_obs) ** 2)
                            grad = torch.autograd.grad(g_loss, x_hat)[0]

                            # --- STABILIZATION ---
                            grad_norm = torch.norm(grad)
                            if grad_norm > 1e-8:
                                # Normalize, then scale back up to a "strong" unit (2.0)
                                # This makes every step meaningful without being infinite
                                grad = (grad / grad_norm) * 2.0

                                # --- ANNEALING & UPDATING ---
                            # Try increasing config.eta to 0.5 in your args for 7x7
                            current_eta = config.eta * (1.0 - t_val)
                            x_hat = x_hat.detach() - current_eta * grad

                            # Tight clamp to keep the image in the EMNIST pixel range
                            x_hat = torch.clamp(x_hat, -2.0, 2.0)

                # PHASE 2: Train with CFG Labels
                model.train()
                x1, x0 = x_hat.detach(), torch.randn_like(x_hat)
                t = torch.rand(B, 1, device=device)
                xt = (1 - t.view(B, 1, 1, 1)) * x0 + t.view(B, 1, 1, 1) * x1
                ut = x1 - x0

                vt_pred = model(xt, t, train_labels)
                loss = F.mse_loss(vt_pred, ut)

                optimizer.zero_grad();
                loss.backward();
                optimizer.step()
                total_loss += loss.item()

            scheduler.step()
            avg_loss = total_loss / len(loader)
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch + 1:03d} | Loss: {avg_loss:.6f} | LR: {current_lr:.2e}")

            if (epoch + 1) % config.vis_every == 0:
                visualize_grid(model, loader, os.path.join(run_dir, f"epoch_{epoch + 1}.png"), config, device)
                print(f"Eval saved to: {run_dir}")
                torch.save(model.state_dict(), os.path.join(run_dir, "last_model.pt"))

        print(f"Results saved to: {run_dir}")
