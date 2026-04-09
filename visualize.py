import torch
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import wandb
from utils import get_cfg_velocity
from project import (
    get_foreground_indices,
    f_project,
    get_batch_unique_matrices,
    f_random_batch_projection
)
from metrics import compute_psnr, compute_ssim

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

    # Compute PSNR/SSIM on first dream result vs GT (both already in [0, 1] range for grayscale)
    samples_01 = dream_results[0].clamp(0, 1)
    gt_01 = images.clamp(0, 1)
    psnr_val = compute_psnr(samples_01, gt_01).item()
    ssim_val = compute_ssim(samples_01, gt_01).item()

    imgs = [images, patch_viz] + dream_results + [x_prior]
    titles = ["GT", "Patch", "S1", "S2", "S3", "Prior"]
    for r in range(n_rows):
        for c in range(6):
            axes[r, c].imshow(imgs[c][r, 0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
            if r == 0: axes[r, c].set_title(titles[c])
            axes[r, c].axis('off')
    fig.suptitle(f'PSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.4f}', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return {"psnr": psnr_val, "ssim": ssim_val}

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

            current_eta = config.eta * (1.0 - 0.7 * t_val)
            with torch.enable_grad():
                x_dream.requires_grad_(True)
                y_hat = f_random_batch_projection(x_dream, A_batch)
                loss_g = torch.sum((y_hat - y_obs) ** 2)
                #loss_g = F.mse_loss(y_hat, y_obs)
                grad = torch.autograd.grad(loss_g, x_dream)[0]
                x_dream = x_dream.detach() - current_eta * grad
        dream_results.append(x_dream)

    x_prior = torch.randn_like(images)
    for s in range(config.fm_steps):
        t = torch.ones(B, 1, device=device) * (s * dt)
        x_prior = x_prior + get_cfg_velocity(model, x_prior, t, labels, cfg_scale=4.0) * dt

    # Error Map: Absolute difference between GT and the first guided dream
    error_map = torch.abs(images - dream_results[0])

    # Compute PSNR/SSIM on first dream result vs GT
    samples_01 = dream_results[0].clamp(0, 1)
    gt_01 = images.clamp(0, 1)
    psnr_val = compute_psnr(samples_01, gt_01).item()
    ssim_val = compute_ssim(samples_01, gt_01).item()

    imgs = [images, error_map] + dream_results + [x_prior]
    titles = ["GT", "error (GT-S1)", "S1", "S2", "S3", "Prior"]

    for r in range(n_rows):
        for c in range(6):
            cmap = 'hot' if c == 1 else 'gray'
            axes[r, c].imshow(imgs[c][r, 0].cpu().numpy(), cmap=cmap, vmin=0, vmax=1)
            if r == 0: axes[r, c].set_title(titles[c])
            axes[r, c].axis('off')

    fig.suptitle(f'PSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.4f}', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return {"psnr": psnr_val, "ssim": ssim_val}

def visualize_grid(model, loader, output_path, config, device='cpu'):
    if config.proj == 'patch':
        return visualize_grid_patch(model, loader, output_path, config, device)
    else:
        return visualize_grid_random_proj(model, loader, output_path, config, device)


def log_guided_samples(model, loader, config, device='cpu', n_samples=100, n_vis=10):
    """Sample n_samples images with guided flow, compute PSNR/SSIM, log to wandb."""
    try:
        model.eval()
        # Collect enough GT images
        all_images, all_labels = [], []
        for images, labels in loader:
            all_images.append(images.to(device))
            all_labels.append(labels.to(device))
            if sum(x.shape[0] for x in all_images) >= n_samples:
                break
        all_images = torch.cat(all_images, dim=0)[:n_samples]
        all_labels = torch.cat(all_labels, dim=0)[:n_samples]
        B = all_images.shape[0]

        # Guided sampling (same logic as visualize dream)
        null_labels = torch.full_like(all_labels, 10)
        dt = 1.0 / config.fm_steps

        if config.proj == 'patch':
            y_idx, x_idx = get_foreground_indices(all_images, config.patch_size, config.consistent_patch)
            y_obs = f_project(all_images, y_idx, x_idx, config.patch_size)
        else:
            A_batch = get_batch_unique_matrices(all_images, config.n_measurements)
            y_obs = f_random_batch_projection(all_images, A_batch)

        x_dream = torch.randn_like(all_images)
        for s in range(config.fm_steps):
            t_val = s / config.fm_steps
            t = torch.ones(B, 1, device=device) * t_val
            with torch.no_grad():
                vt = model(x_dream, t, null_labels)
            x_dream = x_dream + vt * dt

            current_eta = config.eta * (1.0 - t_val) if config.proj == 'patch' else config.eta * (1.0 - 0.7 * t_val)
            with torch.enable_grad():
                x_dream.requires_grad_(True)
                if config.proj == 'patch':
                    y_hat = f_project(x_dream, y_idx, x_idx, config.patch_size)
                    loss_g = F.mse_loss(y_hat, y_obs) * 0.5 * (config.patch_size ** 2)
                else:
                    y_hat = f_random_batch_projection(x_dream, A_batch)
                    loss_g = torch.sum((y_hat - y_obs) ** 2)
                grad = torch.autograd.grad(loss_g, x_dream)[0]
                x_dream = x_dream.detach() - current_eta * grad

        samples_01 = x_dream.clamp(0, 1)
        gt_01 = all_images.clamp(0, 1)

        # Per-sample PSNR/SSIM
        psnr_vals = [compute_psnr(samples_01[i:i+1], gt_01[i:i+1]).item() for i in range(B)]
        ssim_vals = [compute_ssim(samples_01[i:i+1], gt_01[i:i+1]).item() for i in range(B)]
        avg_psnr = np.mean(psnr_vals)
        avg_ssim = np.mean(ssim_vals)

        wandb.log({"guided_samples/psnr": avg_psnr, "guided_samples/ssim": avg_ssim})

        # Visualize a subset
        n_vis = min(n_vis, B)
        fig, axes = plt.subplots(2, n_vis, figsize=(3 * n_vis, 6))
        if n_vis == 1:
            axes = axes.reshape(2, 1)
        for i in range(n_vis):
            gt_i = gt_01[i].cpu()
            s_i = samples_01[i].cpu()
            is_gray = gt_i.shape[0] == 1
            show_gt = gt_i.squeeze(0).numpy() if is_gray else gt_i.permute(1, 2, 0).numpy()
            show_s = s_i.squeeze(0).numpy() if is_gray else s_i.permute(1, 2, 0).numpy()
            kw = dict(cmap='gray', vmin=0, vmax=1) if is_gray else dict(vmin=0, vmax=1)
            axes[0, i].imshow(show_gt, **kw)
            axes[0, i].set_title(f'GT #{i}')
            axes[0, i].axis('off')
            axes[1, i].imshow(show_s, **kw)
            axes[1, i].set_title(f'PSNR:{psnr_vals[i]:.1f} SSIM:{ssim_vals[i]:.3f}')
            axes[1, i].axis('off')

        fig.suptitle(f'Avg PSNR: {avg_psnr:.2f} dB | Avg SSIM: {avg_ssim:.4f} (n={B})', fontsize=14)
        plt.tight_layout()
        wandb.log({"guided_samples": wandb.Image(fig)})
        plt.close(fig)

        print(f"Guided samples: PSNR={avg_psnr:.2f} dB | SSIM={avg_ssim:.4f} (n={B})")
        return {"psnr": avg_psnr, "ssim": avg_ssim}
    except Exception as e:
        print(f"Guided sample logging failed: {e}")
        return None


def log_flow_samples(model, loader, config, device='cpu', n_samples=20):
    """Unguided flow samples (CFG only, no guidance). Compute PSNR/SSIM vs GT, log to wandb."""
    try:
        model.eval()
        # Collect GT images
        all_images, all_labels = [], []
        for images, labels in loader:
            all_images.append(images.to(device))
            all_labels.append(labels.to(device))
            if sum(x.shape[0] for x in all_images) >= n_samples:
                break
        all_images = torch.cat(all_images, dim=0)[:n_samples]
        all_labels = torch.cat(all_labels, dim=0)[:n_samples]
        B = all_images.shape[0]

        # Unguided CFG sampling
        dt = 1.0 / config.fm_steps
        x = torch.randn_like(all_images)
        with torch.no_grad():
            for s in range(config.fm_steps):
                t = torch.ones(B, 1, device=device) * (s * dt)
                x = x + get_cfg_velocity(model, x, t, all_labels, cfg_scale=4.0) * dt

        samples_01 = x.clamp(0, 1)
        gt_01 = all_images.clamp(0, 1)

        psnr_val = compute_psnr(samples_01, gt_01).item()
        ssim_val = compute_ssim(samples_01, gt_01).item()
        wandb.log({"flow_samples/psnr": psnr_val, "flow_samples/ssim": ssim_val})

        grid = vutils.make_grid(samples_01, nrow=10, padding=2)
        fig, ax = plt.subplots(figsize=(12, 4))
        if grid.shape[0] == 1:
            ax.imshow(grid.squeeze(0).cpu().numpy(), cmap='gray', vmin=0, vmax=1)
        else:
            ax.imshow(grid.permute(1, 2, 0).cpu().numpy())
        ax.set_title(f'PSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.4f}')
        ax.axis('off')
        plt.tight_layout()
        wandb.log({"flow_samples": wandb.Image(fig)})
        plt.close(fig)

        print(f"Flow samples: PSNR={psnr_val:.2f} dB | SSIM={ssim_val:.4f} (n={B})")
        return {"psnr": psnr_val, "ssim": ssim_val}
    except Exception as e:
        print(f"Flow sample logging failed: {e}")
        return None