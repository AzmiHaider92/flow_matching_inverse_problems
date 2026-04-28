import argparse
import torch
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment
from model_mnist import ImageFlow, FullImageFlowModel


def psnr_matrix(a, b, max_val=2.0, chunk_size=64):
    # a: (M, C, H, W), b: (N, C, H, W). Images in [-1, 1] -> max_val=2.0
    # returns (M, N)
    M, N = a.shape[0], b.shape[0]
    a_flat = a.reshape(M, -1)
    b_flat = b.reshape(N, -1)
    out = torch.empty(M, N, device=a.device)
    for s in range(0, M, chunk_size):
        e = min(s + chunk_size, M)
        diff = a_flat[s:e, None, :] - b_flat[None, :, :]
        mse = (diff ** 2).mean(dim=-1).clamp(min=1e-12)
        out[s:e] = 10 * torch.log10(max_val ** 2 / mse)
    return out


def precision_psnr(matrix):
    return float(matrix.max(dim=0).values.mean())


def recall_psnr(matrix):
    return float(matrix.max(dim=1).values.mean())


def emd_psnr(matrix):
    cost = -matrix.detach().cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost)
    return float(-cost[row_ind, col_ind].mean())


def k_precision_recall_psnr(matrix, gt_intra, gen_intra, k=3):
    R_gt = gt_intra.topk(k + 1, dim=1, largest=True).values[:, k]
    R_gen = gen_intra.topk(k + 1, dim=1, largest=True).values[:, k]
    precision = (matrix >= R_gt[:, None]).any(dim=0).float().mean().item()
    recall = (matrix >= R_gen[None, :]).any(dim=1).float().mean().item()
    return float(precision), float(recall)


def _gaussian_window(window_size, sigma, device, dtype):
    coords = torch.arange(window_size, dtype=dtype, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g[:, None] * g[None, :]


def ssim_matrix(a, b, max_val=2.0, window_size=11, sigma=1.5, chunk_size=16):
    # a: (M, C, H, W), b: (N, C, H, W). Images in [-1, 1] -> max_val=2.0
    # returns (M, N) with mean SSIM per pair
    M, C, H, W = a.shape
    N = b.shape[0]
    device = a.device
    dtype = a.dtype

    window = _gaussian_window(window_size, sigma, device, dtype)
    window = window.expand(C, 1, window_size, window_size).contiguous()
    pad = window_size // 2
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2

    def stats(x):
        mu = F.conv2d(x, window, padding=pad, groups=C)
        mu_sq = mu * mu
        sigma_sq = F.conv2d(x * x, window, padding=pad, groups=C) - mu_sq
        return mu, mu_sq, sigma_sq

    mu_a, mu_a_sq, sigma_a_sq = stats(a)
    mu_b, mu_b_sq, sigma_b_sq = stats(b)

    out = torch.empty(M, N, device=device, dtype=dtype)
    for s in range(0, M, chunk_size):
        e = min(s + chunk_size, M)
        m = e - s
        ab = (a[s:e].unsqueeze(1) * b.unsqueeze(0)).reshape(m * N, C, H, W)
        mu_ab = F.conv2d(ab, window, padding=pad, groups=C).view(m, N, C, H, W)

        mu_a_c = mu_a[s:e].unsqueeze(1)
        mu_a_sq_c = mu_a_sq[s:e].unsqueeze(1)
        sigma_a_sq_c = sigma_a_sq[s:e].unsqueeze(1)
        mu_b_b = mu_b.unsqueeze(0)
        mu_b_sq_b = mu_b_sq.unsqueeze(0)
        sigma_b_sq_b = sigma_b_sq.unsqueeze(0)

        mu_a_mu_b = mu_a_c * mu_b_b
        sigma_ab = mu_ab - mu_a_mu_b

        num = (2 * mu_a_mu_b + C1) * (2 * sigma_ab + C2)
        den = (mu_a_sq_c + mu_b_sq_b + C1) * (sigma_a_sq_c + sigma_b_sq_b + C2)
        out[s:e] = (num / den).mean(dim=(2, 3, 4))

    return out


def precision_ssim(matrix):
    return float(matrix.max(dim=0).values.mean())


def recall_ssim(matrix):
    return float(matrix.max(dim=1).values.mean())


def emd_ssim(matrix):
    cost = -matrix.detach().cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost)
    return float(-cost[row_ind, col_ind].mean())


def k_precision_recall_ssim(matrix, gt_intra, gen_intra, k=3):
    R_gt = gt_intra.topk(k + 1, dim=1, largest=True).values[:, k]
    R_gen = gen_intra.topk(k + 1, dim=1, largest=True).values[:, k]
    precision = (matrix >= R_gt[:, None]).any(dim=0).float().mean().item()
    recall = (matrix >= R_gen[None, :]).any(dim=1).float().mean().item()
    return float(precision), float(recall)


@torch.no_grad()
def generate_samples(model, num_samples, batch_size=64):
    samples = []
    for s in range(0, num_samples, batch_size):
        b = min(batch_size, num_samples - s)
        samples.append(model.generate(n_samples=b))
    return torch.cat(samples, dim=0)


def generate_guided_samples(model, batch_size=64):
    n = model.num_images
    samples = []
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        indices = torch.arange(s, e, device=model.device)
        samples.append(model._guided_sample(indices).detach())
    return torch.cat(samples, dim=0)


def evaluate(gt, gen, k, label):
    print(f"[{label}] Computing PSNR matrix ({gt.shape[0]} GT x {gen.shape[0]} gen)...")
    matrix = psnr_matrix(gt, gen)
    precision_v = precision_psnr(matrix)
    recall_v = recall_psnr(matrix)
    emd_v = emd_psnr(matrix)

    print(f"[{label}] Computing intra PSNR matrices for k-NN precision/recall...")
    gt_intra = psnr_matrix(gt, gt)
    gen_intra = psnr_matrix(gen, gen)
    k_p, k_r = k_precision_recall_psnr(matrix, gt_intra, gen_intra, k=k)

    print(f"[{label}] Computing SSIM matrix ({gt.shape[0]} GT x {gen.shape[0]} gen)...")
    ssim_mat = ssim_matrix(gt, gen)
    precision_s = precision_ssim(ssim_mat)
    recall_s = recall_ssim(ssim_mat)
    emd_s = emd_ssim(ssim_mat)

    print(f"[{label}] Computing intra SSIM matrices for k-NN precision/recall...")
    gt_intra_s = ssim_matrix(gt, gt)
    gen_intra_s = ssim_matrix(gen, gen)
    k_p_s, k_r_s = k_precision_recall_ssim(ssim_mat, gt_intra_s, gen_intra_s, k=k)

    print(f"[{label}] precision_psnr: {precision_v}, precision_ssim: {precision_s}")
    print(f"[{label}] recall_psnr: {recall_v}, recall_ssim: {recall_s}")
    print(f"[{label}] emd_psnr: {emd_v}, emd_ssim: {emd_s}")
    print(f"[{label}] K_precision_psnr: {k_p}, K_precision_ssim: {k_p_s}")
    print(f"[{label}] K_recall_psnr: {k_r}, K_recall_ssim: {k_r_s}")


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ImageFlow.load_from_checkpoint(args.checkpoint_path)
    model = model.to(device).eval()

    if args.fm_steps > 0:
        model.fm_steps = args.fm_steps

    gt = model.gt_images.to(device)

    n_samples = args.num_samples or model.num_images
    print(f"Generating {n_samples} unguided samples...")
    gen = generate_samples(model, n_samples).to(device)
    evaluate(gt, gen, args.k, label='generated')

    if getattr(model, 'use_consistent_latents', False):
        latents = model.latent_images.detach().to(device)
        evaluate(gt, latents, args.k, label='latents')

    print(f"Generating {model.num_images} guided samples...")
    guided = generate_guided_samples(model).to(device)
    evaluate(gt, guided, args.k, label='guided')


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint_path', type=str, required=True)
    p.add_argument('--num_samples', type=int, default=0,
                   help='0 = use model.num_images (one sample per GT image)')
    p.add_argument('--k', type=int, default=5)
    p.add_argument('--fm_steps', type=int, default=0,
                   help='0 = use the value stored in the checkpoint')
    p.add_argument('--run_name', type=str, default='mnist-eval')
    args = p.parse_args()
    main(args)
