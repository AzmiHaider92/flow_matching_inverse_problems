import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from datetime import datetime
import torch.nn.functional as F
import wandb

# Local imports
from data import EMNIST
from model import FullImageFlowModel
from project import (
    get_foreground_indices,
    f_project,
    f_random_batch_projection, get_batch_unique_matrices
)
from visualize import visualize_grid, log_guided_samples, log_flow_samples


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['train', 'eval'], default='train')
    parser.add_argument('--ckpt_path', type=str, default=None, help="Path to load weights")
    parser.add_argument('--expname', type=str, default="experiment", help="Optional experiment name")
    parser.add_argument('--n_epochs', type=int, default=500)
    parser.add_argument('--vis_every', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--lr_factor', type=float, default=0.1)

    # Guidance / Sampling
    parser.add_argument('--guide_steps', type=int, default=10, help="Steps for guided sampling in training")
    parser.add_argument('--fm_steps', type=int, default=100, help="Steps for ODE integration in eval")
    parser.add_argument('--eta', type=float, default=0.1, help="Guidance scale (GPS strength)")

    # Projection Logic
    parser.add_argument('--proj', type=str, default='random', choices=['patch', 'random'],
                        help='Projection type: spatial patch or random matrix')
    parser.add_argument('--patch_size', type=int, default=7)
    parser.add_argument('--consistent_patch', action='store_true', help="Seed patch selection by image content")
    parser.add_argument('--n_measurements', type=int, default=100, help="M for random projection")

    parser.add_argument('--class_range', type=int, nargs='*', default=[0, 10])
    parser.add_argument('--wandb_project', type=str, default='IGFM', help="W&B project name")
    return parser.parse_args()


if __name__ == "__main__":
    config = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup Experiment Name
    suffix = "C" if (config.proj == "patch" and config.consistent_patch) else ""
    full_expname = f"{config.expname}-{config.proj}{suffix}"

    # 1. Directory Setup
    if config.mode == 'train':
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = os.path.join("outputs", f"{full_expname}_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)
    else:
        if config.ckpt_path is None:
            raise ValueError("You must provide --ckpt_path in eval mode.")
        run_dir = os.path.dirname(config.ckpt_path)

    # W&B Setup
    wandb.init(
        entity="viewformer",
        # Set the wandb project where this run will be logged.
        project="mnist-flow-matching",
        # Track hyperparameters and run metadata.
        config=vars(config),
        name=config.expname,
    )

    # 2. Data & Model Setup
    dataset = EMNIST(train=(config.mode == 'train'), class_range=config.class_range, device=device)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    model = FullImageFlowModel(num_classes=dataset.num_classes).to(device)

    if config.ckpt_path:
        print(f"Loading checkpoint: {config.ckpt_path}")
        model.load_state_dict(torch.load(config.ckpt_path, map_location=device))

    # --- EVALUATION MODE ---
    if config.mode == 'eval':
        print("Running Evaluation...")
        out_path = os.path.join(run_dir, f"eval_{datetime.now().strftime('%H%M%S')}.png")
        visualize_grid(model, loader, out_path, config, device=device)
        print(f"Saved to: {out_path}")

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

                # 1. Define the unique constraints for THIS batch of images
                if config.proj == 'patch':
                    y_idx, x_idx = get_foreground_indices(images, config.patch_size, config.consistent_patch)
                    y_obs = f_project(images, y_idx, x_idx, config.patch_size)
                else:
                    # UNIQUE A per image, generated once per batch
                    A_batch = get_batch_unique_matrices(images, config.n_measurements)
                    y_obs = f_random_batch_projection(images, A_batch)

                # 2. PHASE 1: Dream (Guidance)
                model.eval()
                with torch.no_grad():
                    x_hat = torch.randn_like(images)
                    dt_g = 1.0 / config.guide_steps
                    for s in range(config.guide_steps):
                        t_c = torch.ones(B, 1, device=device) * (s * dt_g)
                        x_hat = x_hat + model(x_hat, t_c, labels) * dt_g

                        with torch.enable_grad():
                            x_hat.requires_grad_(True)
                            if config.proj == 'patch':
                                y_h = f_project(x_hat, y_idx, x_idx, config.patch_size)
                                #g_loss = torch.sum((y_h - y_obs) ** 2)
                            else:
                                # Use the SAME A_batch that was generated for the GT images
                                y_h = f_random_batch_projection(x_hat, A_batch)
                                #g_loss = F.mse_loss(y_h, y_obs) * 100.0

                            g_loss = torch.sum((y_h - y_obs) ** 2)
                            grad = torch.autograd.grad(g_loss, x_hat)[0]
                            x_hat = x_hat.detach() - config.eta * grad

                # 3. PHASE 2: Train Flow Matching
                model.train()

                # RE-ADD: CFG Label Dropout (10% chance)
                drop_mask = torch.rand(B, device=device) < 0.1
                train_labels = torch.where(drop_mask, torch.tensor(10, device=device), labels)

                # Now x1 is a "reconstruction" that satisfies image-specific random projections
                x1, x0 = x_hat.detach(), torch.randn_like(x_hat)
                t = torch.rand(B, 1, device=device)

                # Linear Interpolation (Optimal Transport Path)
                xt = (1 - t.view(B, 1, 1, 1)) * x0 + t.view(B, 1, 1, 1) * x1
                ut = x1 - x0

                vt_pred = model(xt, t, train_labels) # Now train_labels exists!
                loss = F.mse_loss(vt_pred, ut)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
                model.train()
                # Now x1 is a "reconstruction" that satisfies image-specific random projections
                x1, x0 = x_hat.detach(), torch.randn_like(x_hat)
                t = torch.rand(B, 1, device=device)
                # Linear Interpolation (Optimal Transport Path)
                xt = (1 - t.view(B, 1, 1, 1)) * x0 + t.view(B, 1, 1, 1) * x1
                ut = x1 - x0

                vt_pred = model(xt, t, train_labels)
                loss = F.mse_loss(vt_pred, ut)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            scheduler.step()
            avg_loss = total_loss / len(loader)
            print(f"Epoch {epoch + 1:03d} | Loss: {avg_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
            wandb.log({"train/loss": avg_loss, "train/lr": optimizer.param_groups[0]['lr'], "epoch": epoch + 1})

            if (epoch + 1) % config.vis_every == 0:
                vis_path = os.path.join(run_dir, f"epoch_{epoch + 1}.png")
                metrics = visualize_grid(model, loader, vis_path, config, device)
                if metrics:
                    wandb.log({"eval/psnr": metrics["psnr"], "eval/ssim": metrics["ssim"], "epoch": epoch + 1})
                wandb.log({"eval/grid": wandb.Image(vis_path), "epoch": epoch + 1})
                print(f"Eval saved to: {run_dir} | PSNR: {metrics['psnr']:.2f} dB | SSIM: {metrics['ssim']:.4f}")
                log_guided_samples(model, loader, config, device, n_samples=100, n_vis=10)
                log_flow_samples(model, loader, config, device, n_samples=100)
                torch.save(model.state_dict(), os.path.join(run_dir, "last_model.pt"))

        wandb.finish()
        print(f"Training complete. Results in: {run_dir}")