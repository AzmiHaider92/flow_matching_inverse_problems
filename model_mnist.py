import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
import wandb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
#from loss import RenderingLoss
from diffusion import create_diffusion


class FourierEmbedding(nn.Module):
    def __init__(self, in_channels=1, out_channels=64, scale=20.0):
        super().__init__()
        # in_channels should be 1 because 't' is a single scalar per batch item
        self.register_buffer('B', torch.randn(in_channels, out_channels // 2) * scale)

    def forward(self, x):
        # Ensure x is (Batch, 1)
        if x.ndim == 1:
            x = x.unsqueeze(-1)
        # Handle cases where x might be (Batch,) or (Batch, 1, 1, 1)
        if x.ndim > 2:
            x = x.view(x.shape[0], 1)

        x = x.to(torch.float32)
        x_proj = 2 * np.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class FullImageFlowModel(nn.Module):
    def __init__(self, in_channels=1, obs_dim=392):
        super().__init__()
        self.time_emb = FourierEmbedding(1, 64, scale=20.0)
        self.obs_proj = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
        )
        self.cond_proj = nn.Linear(64 + 128, 128)

        self.net = nn.Sequential(
            nn.Conv2d(in_channels + 128, 64, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=2, dilation=2),
            nn.SiLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=4, dilation=4),
            nn.SiLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=6, dilation=6),
            nn.SiLU(),
            nn.Conv2d(64, in_channels, kernel_size=3, padding=1)
        )

    def forward(self, x_t, t, obs):
        B, C, H, W = x_t.shape
        t_e = self.time_emb(t)
        o_e = self.obs_proj(obs)
        cond = self.cond_proj(torch.cat([t_e, o_e], dim=-1))
        cond_map = cond.view(B, 128, 1, 1).expand(B, 128, H, W)
        return self.net(torch.cat([x_t, cond_map], dim=1))


class ImageLinearForwardModel(nn.Module):
    def __init__(self, total_pixels, n_measurements, noise_std=0):
        super().__init__()
        self.total_pixels = total_pixels
        self.n = n_measurements
        self.noise_std = noise_std
        self._cache = {}

    def make_A(self, idx, device='cpu'):
        if (idx, device) in self._cache:
            return self._cache[(idx, device)]
        gen = torch.Generator(device='cpu').manual_seed(int(idx))
        A = torch.randn(self.total_pixels, self.n, generator=gen, device='cpu')
        A = A / torch.norm(A, dim=0, keepdim=True).clamp_min(1e-12)
        A = A.to(device)
        self._cache[(idx, device)] = A
        return A

    def forward(self, x, idx):
        A = self.make_A(idx, device=x.device)
        x_flat = x.reshape(-1)
        y = A.T @ x_flat
        if self.noise_std > 0:
            y = y + self.noise_std * torch.randn(self.n, device=x.device)
        return y


class ImageFlow(pl.LightningModule):
    def __init__(self, dataset_name='cifar10', data_root='./data',
                 n_measurements=392, noise_std=1e-3,
                 flow_lr=5e-4, latent_lr=0.01,
                 flow_weight=1.0, render_weight=1.0,
                 render_loss_type='mse',
                 guided_N=10, guided_warmup_steps=3, guided_eta=0.1,
                 num_images=None, flow_model_path=None,
                 batch_size=128, fm_steps=100,
                 flow_train_epochs=10, warmup_epochs=5,
                 latent_l1_weight=0.01,
                 use_consistent_latents=True, model_type='flow'):
        super().__init__()
        self.save_hyperparameters()
        self.use_consistent_latents = use_consistent_latents
        self.model_type = model_type
        self.dataset_name = dataset_name
        self.data_root = data_root
        self.flow_lr = flow_lr
        self.latent_lr = latent_lr
        self.flow_weight = flow_weight
        self.render_weight = render_weight
        self.render_loss_type = render_loss_type
        self.guided_N = guided_N
        self.guided_warmup_steps = guided_warmup_steps
        self.guided_eta = guided_eta
        self.noise_std = noise_std
        self.batch_size = batch_size
        self.fm_steps = fm_steps
        self._flow_train_epochs = flow_train_epochs
        self._warmup_epochs = warmup_epochs
        self.latent_l1_weight = latent_l1_weight
        self._flow_model_ready = False
        self._flow_model_pretrained = False

        all_images = self._load_dataset(self.dataset_name, data_root, num_images=num_images)
        self.num_images = all_images.shape[0]
        self.channels = all_images.shape[1]
        self.img_size = all_images.shape[2]

        D = self.channels * self.img_size * self.img_size
        self.forward_model = ImageLinearForwardModel(D, n_measurements, noise_std=noise_std)
        with torch.no_grad():
            observations = []
            gen = torch.Generator().manual_seed(0)
            for i in range(self.num_images):
                A = self.forward_model.make_A(i)
                x_flat = all_images[i].reshape(-1)
                y = A.T @ x_flat
                if noise_std > 0:
                    y = y + noise_std * torch.randn(n_measurements, generator=gen)
                observations.append(y)
            observations = torch.stack(observations)
        self.register_buffer('gt_observations', observations)
        self.register_buffer('gt_images', all_images)

        # Latent Management
        if self.use_consistent_latents:
            self.latent_images = nn.Parameter(torch.zeros(self.num_images, self.channels, self.img_size, self.img_size))
        else:
            # Fixed random noise that doesn't get gradients
            random_latents = torch.randn(self.num_images, self.channels, self.img_size, self.img_size)
            self.register_buffer('latent_images', random_latents)

        # Diffusion Setup (Conditional)
        if self.model_type == 'diffusion':
            # Assuming you have a create_diffusion helper available
            self.diffusion = create_diffusion(timestep_respacing="")

        self.flow_model = FullImageFlowModel(in_channels=self.channels, obs_dim=n_measurements)
        if flow_model_path and os.path.exists(flow_model_path):
            ckpt = torch.load(flow_model_path, map_location='cpu')
            self.flow_model.load_state_dict(ckpt if not isinstance(ckpt, dict) else ckpt.get('state_dict', ckpt))
            print(f"Loaded flow model from {flow_model_path}")
            self._flow_model_ready = True
            self._flow_model_pretrained = True
            for param in self.flow_model.parameters():
                param.requires_grad = False

        #self.render_loss_fn = RenderingLoss(loss_type=self.render_loss_type)
        self.indices_to_visualize = self._pick_viz_indices()



    def _wandb_log(self, data):
        if self.logger:
            self.logger.experiment.log(data)
        elif wandb.run is not None:
            wandb.log(data)

    def _load_dataset(self, dataset_name, data_root, num_images=None):
        if dataset_name == 'mnist':
            channels = 1
            dataset_cls = datasets.MNIST
        else:
            channels = 3
            dataset_cls = datasets.CIFAR10

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5] * channels, [0.5] * channels),
        ])
        train_dataset = dataset_cls(root=data_root, train=True, download=True, transform=transform)

        images = []
        max_items = len(train_dataset) if num_images is None else min(int(num_images), len(train_dataset))
        for i in range(max_items):
            image, _ = train_dataset[i]
            images.append(image)
        return torch.stack(images)

    def _pick_viz_indices(self):
        n = self.num_images
        if n <= 4:
            return list(range(n))
        return [0, n // 4, n // 2, n - 1]

    def train_dataloader(self):
        dataset = torch.utils.data.TensorDataset(torch.arange(self.num_images))
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def configure_optimizers(self):
        params = []
        # Only optimize latents if "consistent" mode is ON
        if self.use_consistent_latents:
            params.append({'params': [self.latent_images], 'lr': self.latent_lr})

        if not self._flow_model_pretrained:
            params.append({'params': self.flow_model.parameters(), 'lr': self.flow_lr})

        return torch.optim.Adam(params)

    def configure_gradient_clipping(self, optimizer, gradient_clip_val, gradient_clip_algorithm):
        self.clip_gradients(optimizer, gradient_clip_val=1.0, gradient_clip_algorithm='norm')

    def train_train_flow_model_on_latents_batch(self, x1, obs):
        x1 = x1.to(self.device)
        obs = obs.to(self.device)
        B = x1.shape[0]

        if self.model_type == 'flow':
            # --- Flow Matching Training Logic ---
            x0 = torch.randn_like(x1)
            t = torch.rand(B, 1, device=self.device)
            xt = (1 - t.view(B, 1, 1, 1)) * x0 + t.view(B, 1, 1, 1) * x1
            ut = x1 - x0
            vt_pred = self.flow_model(xt, t, obs)
            loss = F.mse_loss(vt_pred, ut)

        else:
            # --- Diffusion Training Logic ---
            # We use the training_losses helper from your diffusion library
            t = torch.randint(0, self.diffusion.num_timesteps, (B, 1), device=self.device)
            loss_dict = self.diffusion.training_losses(
                self.flow_model, x1, t, model_kwargs={'obs': obs}
            )
            loss = loss_dict["loss"].mean()
        return loss

    def _train_flow_model_on_latents(self):
        # Prepare model for training
        for param in self.flow_model.parameters():
            param.requires_grad = True
        self.flow_model.train()

        # Create a local dataset of CURRENT latents and their corresponding measurements
        x1_all = self.latent_images.detach()
        obs_all = self.gt_observations.detach()
        dataset = torch.utils.data.TensorDataset(x1_all, obs_all)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

        optimizer = torch.optim.Adam(self.flow_model.parameters(), lr=self.flow_lr)

        for epoch in range(self._flow_train_epochs):
            total_loss = 0.0
            for x1, obs in dataloader:
                optimizer.zero_grad()
                loss = self.train_train_flow_model_on_latents_batch(x1, obs)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.flow_model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 5 == 0:
                print(
                    f"  [{self.model_type.upper()}] Epoch {epoch + 1}/{self._flow_train_epochs} Loss={total_loss / len(dataloader):.4f}")

        # Set back to eval mode for guidance/sampling
        self.flow_model.eval()
        for param in self.flow_model.parameters():
            param.requires_grad = False
        self._flow_model_ready = True

    def _guided_sampling_batch(self, gt_obs, A_batch, B):
        self.flow_model.eval()

        # Start from pure noise
        x = torch.randn(B, self.channels, self.img_size, self.img_size, device=self.device)

        if self.model_type == 'flow':
            # --- FLOW MATCHING REVERSE (ODE) ---
            dt = 1.0 / self.guided_N
            for step in range(self.guided_N):
                t_c = torch.full((B, 1), step * dt, device=self.device)
                # Standard Flow Step
                v = self.flow_model(x, t_c, gt_obs)
                x = x + v * dt

                # Measurement Guidance (DPS-style)
                with torch.enable_grad():
                    x = x.detach().requires_grad_(True)
                    x_flat = x.view(B, -1)
                    y_hat = torch.bmm(x_flat.unsqueeze(1), A_batch).squeeze(1)
                    g_loss = torch.sum((y_hat - gt_obs) ** 2)
                    grad = torch.autograd.grad(g_loss, x)[0]
                    x = x.detach() - self.guided_eta * grad

        elif self.model_type == 'diffusion':
            # --- DIFFUSION REVERSE (DDIM/Langevin) ---
            # We use the 'p_sample_loop' or 'ddim_sample_loop' from the diffusion library
            # But we need to inject the Measurement Guidance (A_batch) into the loop

            def model_fn(x, t, obs=None):
                return self.flow_model(x, t, obs)

            # Using a simplified reverse loop logic (similar to Diffusion Posterior Sampling)
            indices = list(range(self.diffusion.num_timesteps))[::-1]

            for i in indices:
                t = torch.full((B, 1), i, device=self.device, dtype=torch.long)

                # 1. Standard Diffusion Step (Denoising)
                out = self.diffusion.p_sample(
                    model_fn, x, t,
                    model_kwargs={'obs': gt_obs}
                )
                x = out["sample"]

                # 2. Measurement Guidance Step
                # This pulls the denoised estimate toward the measurements
                with torch.enable_grad():
                    x = x.detach().requires_grad_(True)
                    x_flat = x.view(B, -1)
                    y_hat = torch.bmm(x_flat.unsqueeze(1), A_batch).squeeze(1)
                    g_loss = torch.sum((y_hat - gt_obs) ** 2)
                    grad = torch.autograd.grad(g_loss, x)[0]
                    # Scale guidance by the current noise level (standard in DPS)
                    x = x.detach() - self.guided_eta * grad

        self.flow_model.train()
        return x

    def training_step_consistant_latents(self, batch, batch_idx):
        (indices,) = batch
        B = indices.shape[0]
        gt_obs = self.gt_observations[indices]
        A_batch = torch.stack([self.forward_model.make_A(idx.item(), device=self.device) for idx in indices])
        latent = self.latent_images[indices]

        # 1. MODEL REFINEMENT (Triggered FIRST as per original code)
        if not self._flow_model_pretrained and self.current_epoch >= self._warmup_epochs:
            epochs_since_warmup = self.current_epoch - self._warmup_epochs
            if epochs_since_warmup % 50 == 0 and batch_idx == 0:
                print(f"Epoch {self.current_epoch}: Training {self.model_type} model on latent images...")
                self._train_flow_model_on_latents()

        # 2. GUIDANCE LOSS
        guidance_loss = torch.tensor(0.0, device=self.device)
        use_flow = self._flow_model_ready and (
                    self._flow_model_pretrained or self.current_epoch >= self._warmup_epochs)

        if self.flow_weight > 0 and use_flow and self.use_consistent_latents:
            # Helper uses self.model_type logic internally
            guided_pred = self._guided_sampling_batch(gt_obs, A_batch, B)
            guidance_loss = F.l1_loss(guided_pred, latent)

        # 3. RENDER LOSS (Forward Model)
        x_flat = latent.view(B, -1)
        y_hat = torch.bmm(x_flat.unsqueeze(1), A_batch).squeeze(1)
        render_loss = F.mse_loss(y_hat, gt_obs)

        # 4. COMBINED LOSS
        # If use_consistent_latents is False, render_loss and guidance_loss will
        # have no effect on optimization because latent is a Buffer, not a Parameter.
        if self.flow_weight > 0 and use_flow:
            combined_loss = self.flow_weight * guidance_loss + self.render_weight * render_loss
        else:
            combined_loss = self.render_weight * render_loss

        self.log('train_render_loss', render_loss)
        self.log('train_guidance_loss', guidance_loss)
        self.log('train_combined_loss', combined_loss, prog_bar=True)

        return combined_loss

    def training_step_non_consistent(self, batch, batch_idx):
        (indices,) = batch
        B = indices.shape[0]
        gt_obs = self.gt_observations[indices]
        A_batch = torch.stack([self.forward_model.make_A(idx.item(), device=self.device) for idx in indices])
        latent = self.latent_images[indices]

        # 1. MODEL REFINEMENT (Triggered FIRST as per original code)
        if not self._flow_model_pretrained and self.current_epoch >= self._warmup_epochs:
            if batch_idx == 0:
                print(f"Epoch {self.current_epoch}: Training {self.model_type} model on latent images...")
                self._train_flow_model_on_latents()

        # 2. GUIDANCE LOSS
        #guidance_loss = torch.tensor(0.0, device=self.device)
        #use_flow = self._flow_model_ready and (
        #        self._flow_model_pretrained or self.current_epoch >= self._warmup_epochs)

        #if self.flow_weight > 0 and use_flow and self.use_consistent_latents:
        #    # Helper uses self.model_type logic internally
        #    guided_pred = self._guided_sampling_batch(gt_obs, A_batch, B)
        #    guidance_loss = F.l1_loss(guided_pred, latent)

        guided_pred = self._guided_sampling_batch(gt_obs, A_batch, B)

        # 3. RENDER LOSS (Forward Model)
        self.latent_images[indices] = guided_pred
        x_flat = guided_pred.view(B, -1)
        y_hat = torch.bmm(x_flat.unsqueeze(1), A_batch).squeeze(1)
        render_loss = F.mse_loss(y_hat, gt_obs)

        # 4. COMBINED LOSS
        # If use_consistent_latents is False, render_loss and guidance_loss will
        # have no effect on optimization because latent is a Buffer, not a Parameter.

        self.log('train_render_loss', render_loss, prog_bar=True)

        return

    def training_step(self, batch, batch_idx):
        if self.use_consistent_latents:
            loss = self.training_step_consistant_latents(batch, batch_idx)
        else:
            loss = self.training_step_non_consistent(batch, batch_idx)
        return loss

    def on_train_epoch_end(self):
        if (self.current_epoch + 1) % 50 == 0:
            self._log_flow_samples()
            self._log_guided_samples()

    @torch.no_grad()
    def generate(self, obs):
        self.flow_model.eval()
        B = obs.shape[0]

        if self.model_type == 'flow':
            z = torch.randn(B, self.channels, self.img_size, self.img_size, device=self.device)
            dt = 1.0 / self.fm_steps
            for s in range(self.fm_steps):
                t_c = torch.full((B, 1), s * dt, device=self.device)
                z = z + self.flow_model(z, t_c, obs) * dt
        else:
            # Standard diffusion sampling loop
            z = self.diffusion.p_sample_loop(
                self.flow_model,
                (B, self.channels, self.img_size, self.img_size),
                clip_denoised=True,
                model_kwargs={'obs': obs}
            )

        self.flow_model.train()
        return z

    def _to_rgb(self, x):
        if x.shape[1] == 3:
            return x
        return x.expand(-1, 3, -1, -1)

    def _psnr(self, pred, target, max_val=1.0):
        mse = F.mse_loss(pred, target)
        if mse == 0:
            return torch.tensor(float('inf'))
        return 10 * torch.log10(max_val ** 2 / mse)

    def _ssim(self, pred, target, window_size=11):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        pad = window_size // 2
        mu_p = F.avg_pool2d(pred, window_size, stride=1, padding=pad)
        mu_t = F.avg_pool2d(target, window_size, stride=1, padding=pad)
        mu_pp = mu_p * mu_p
        mu_tt = mu_t * mu_t
        mu_pt = mu_p * mu_t
        sigma_pp = F.avg_pool2d(pred * pred, window_size, stride=1, padding=pad) - mu_pp
        sigma_tt = F.avg_pool2d(target * target, window_size, stride=1, padding=pad) - mu_tt
        sigma_pt = F.avg_pool2d(pred * target, window_size, stride=1, padding=pad) - mu_pt
        ssim_map = ((2 * mu_pt + C1) * (2 * sigma_pt + C2)) / \
                   ((mu_pp + mu_tt + C1) * (sigma_pp + sigma_tt + C2))
        return ssim_map.mean()

    @torch.no_grad()
    def _guided_sample(self, indices):
        gt_obs = self.gt_observations[indices]
        A_batch = torch.stack([self.forward_model.make_A(idx.item(), device=self.device) for idx in indices])
        return self._guided_sampling_batch(gt_obs, A_batch, len(indices))

    def _log_guided_samples(self):
        try:
            indices = torch.tensor(self.indices_to_visualize, device=self.device)
            n = len(indices)
            samples = self._guided_sample(indices)
            samples_01 = samples.mul(0.5).add(0.5).clamp(0, 1)
            gt_01 = self.gt_images[indices].mul(0.5).add(0.5).clamp(0, 1)

            is_gray = self.channels == 1
            fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))
            if n == 1:
                axes = axes.reshape(2, 1)
            for i in range(n):
                gt_i = gt_01[i].cpu()
                s_i = samples_01[i].cpu()
                psnr_i = self._psnr(s_i.unsqueeze(0), gt_i.unsqueeze(0)).item()
                ssim_i = self._ssim(s_i.unsqueeze(0), gt_i.unsqueeze(0)).item()
                show_gt = gt_i.squeeze(0).numpy() if is_gray else gt_i.permute(1, 2, 0).numpy()
                show_s = s_i.squeeze(0).numpy() if is_gray else s_i.permute(1, 2, 0).numpy()
                kw = dict(cmap='gray', vmin=0, vmax=1) if is_gray else dict(vmin=0, vmax=1)
                axes[0, i].imshow(show_gt, **kw)
                axes[0, i].set_title(f'GT #{indices[i].item()}')
                axes[0, i].axis('off')
                axes[1, i].imshow(show_s, **kw)
                axes[1, i].set_title(f'PSNR:{psnr_i:.1f} SSIM:{ssim_i:.3f}')
                axes[1, i].axis('off')

            avg_psnr = np.mean([self._psnr(samples_01[i:i+1], gt_01[i:i+1]).item() for i in range(n)])
            avg_ssim = np.mean([self._ssim(samples_01[i:i+1], gt_01[i:i+1]).item() for i in range(n)])
            self._wandb_log({"guided_samples/psnr": avg_psnr, "guided_samples/ssim": avg_ssim})
            self.log('guided_psnr', avg_psnr, prog_bar=True)
            self.log('guided_ssim', avg_ssim, prog_bar=True)

            plt.tight_layout()
            self._wandb_log({"guided_samples": wandb.Image(fig)})
            plt.close(fig)
        except Exception as e:
            print(f"Guided sample logging failed: {e}")

    def _log_flow_samples(self, n_samples=20):
        try:
            import torchvision.utils as vutils

            n_samples = min(n_samples, self.num_images)
            samples = self.generate(self.gt_observations[:n_samples])
            samples_01 = samples.mul(0.5).add(0.5).clamp(0, 1)
            gt_01 = self.gt_images[:n_samples].mul(0.5).add(0.5).clamp(0, 1)

            psnr_val = self._psnr(samples_01, gt_01).item()
            ssim_val = self._ssim(samples_01, gt_01).item()
            self._wandb_log({"flow_samples/psnr": psnr_val, "flow_samples/ssim": ssim_val})
            self.log('psnr', psnr_val, prog_bar=True)
            self.log('ssim', ssim_val, prog_bar=True)

            grid = vutils.make_grid(samples_01, nrow=10, padding=2)
            fig, ax = plt.subplots(figsize=(12, 4))
            if grid.shape[0] == 1:
                ax.imshow(grid.squeeze(0).cpu().numpy(), cmap='gray', vmin=0, vmax=1)
            else:
                ax.imshow(grid.permute(1, 2, 0).cpu().numpy())
            ax.set_title(f'PSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.4f}')
            ax.axis('off')
            plt.tight_layout()
            self._wandb_log({"flow_samples": wandb.Image(fig)})
            plt.close(fig)
        except Exception as e:
            print(f"Flow sample logging failed: {e}")
