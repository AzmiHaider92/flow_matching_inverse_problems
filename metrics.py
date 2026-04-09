import torch
import torch.nn.functional as F


def compute_psnr(pred, target, max_val=1.0):
    """Compute PSNR between pred and target (both in [0, 1])."""
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return torch.tensor(float('inf'))
    return 10 * torch.log10(max_val ** 2 / mse)


def compute_ssim(pred, target, window_size=11):
    """Compute SSIM between pred and target (both in [0, 1])."""
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
