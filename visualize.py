import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from utils import get_cfg_velocity
from project import (
    get_foreground_indices,
    f_project,
    get_batch_unique_matrices,
    f_random_batch_projection
)

@torch.no_grad()
def visualize_grid_patch(model, loader, output_path, config, device='cpu'):
    model.eval()
    n_rows = 4
    fig, axes = plt.subplots(n_rows, 6, figsize=(20, 10))
    images, labels = next(iter(loader))
    images, labels = images[:n_rows].to(device), labels[:n_rows].to(device)
    B = images.shape[0]

    y_idx, x_idx = get_foreground_indices(images, config.patch_size, config.consistent_patch)
    y_obs = f_project(images, y_idx, x_idx, config.patch_size)

    patch_viz = torch.zeros_like(images)
    for i in range(B):
        patch_viz[i, 0, y_idx[i]:y_idx[i] + config.patch_size, x_idx[i]:x_idx[i] + config.patch_size] = \
            y_obs[i].view(config.patch_size, config.patch_size)

    x_prior = torch.randn_like(images)
    dt = 1.0 / config.fm_steps
    for s in range(config.fm_steps):
        t = torch.ones(B, 1, device=device) * (s * dt)
        x_prior = x_prior + get_cfg_velocity(model, x_prior, t, labels, cfg_scale=4.0) * dt

    dream_results = []
    null_labels = torch.full_like(labels, 10)
    for _ in range(3):
        x_dream = torch.randn_like(images)
        for s in range(config.fm_steps):
            t_val = s / config.fm_steps
            t = torch.ones(B, 1, device=device) * t_val
            vt = model(x_dream, t, null_labels)
            x_dream = x_dream + vt * dt

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
    plt.savefig(output_path)
    plt.close()

@torch.no_grad()
def visualize_grid_random_proj(model, loader, output_path, config, device='cpu'):
    model.eval()
    n_rows = 4
    fig, axes = plt.subplots(n_rows, 6, figsize=(20, 10))

    images, labels = next(iter(loader))
    images, labels = images[:n_rows].to(device), labels[:n_rows].to(device)
    B = images.shape[0]

    A_batch = get_batch_unique_matrices(images, config.n_measurements)
    y_obs = f_random_batch_projection(images, A_batch)

    dream_results = []
    null_labels = torch.full_like(labels, 10)
    dt = 1.0 / config.fm_steps

    for d_idx in range(3):
        x_dream = torch.randn_like(images)
        for s in range(config.fm_steps):
            t_val = s / config.fm_steps
            t = torch.ones(B, 1, device=device) * t_val
            vt = model(x_dream, t, null_labels)
            x_dream = x_dream + vt * dt

            current_eta = config.eta * (1.0 - t_val)
            with torch.enable_grad():
                x_dream.requires_grad_(True)
                y_hat = f_random_batch_projection(x_dream, A_batch)
                loss_g = F.mse_loss(y_hat, y_obs) * 100.0
                grad = torch.autograd.grad(loss_g, x_dream)[0]
                x_dream = x_dream.detach() - current_eta * grad
        dream_results.append(x_dream)

    x_prior = torch.randn_like(images)
    for s in range(config.fm_steps):
        t = torch.ones(B, 1, device=device) * (s * dt)
        x_prior = x_prior + get_cfg_velocity(model, x_prior, t, labels, cfg_scale=4.0) * dt

    # Error Map: Absolute difference between GT and the first guided dream
    error_map = torch.abs(images - dream_results[0])

    imgs = [images, error_map] + dream_results + [x_prior]
    titles = ["GT", "Error (GT vs S1)", "S1", "S2", "S3", "Prior"]

    for r in range(n_rows):
        for c in range(6):
            cmap = 'hot' if c == 1 else 'gray'
            axes[r, c].imshow(imgs[c][r, 0].cpu().numpy(), cmap=cmap, vmin=0, vmax=1)
            if r == 0: axes[r, c].set_title(titles[c])
            axes[r, c].axis('off')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def visualize_grid(model, loader, output_path, config, device='cpu'):
    if config.proj == 'patch':
        visualize_grid_patch(model, loader, output_path, config, device)
    else:
        visualize_grid_random_proj(model, loader, output_path, config, device)