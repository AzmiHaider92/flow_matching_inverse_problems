import argparse
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from model_mnist import ImageFlow


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


@torch.no_grad()
def generate_samples(model, num_samples, device, batch_size=64):
    obs_all = model.gt_observations.to(device)
    n_avail = obs_all.shape[0]
    indices = torch.arange(num_samples, device=device) % n_avail
    samples = []
    for s in range(0, num_samples, batch_size):
        idx = indices[s:s + batch_size]
        samples.append(model.generate(obs_all[idx]))
    return torch.cat(samples, dim=0), indices


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ImageFlow.load_from_checkpoint(args.checkpoint_path)
    model = model.to(device).eval()

    n_samples = args.num_samples or model.num_images
    print(f"Generating {n_samples} samples...")
    gen, sample_obs_idx = generate_samples(model, n_samples, device)
    gt = model.gt_images.to(device)

    print(f"Computing PSNR matrix ({gt.shape[0]} GT x {gen.shape[0]} gen)...")
    matrix = psnr_matrix(gt, gen)

    paired_psnr = float(matrix[sample_obs_idx, torch.arange(n_samples, device=device)].mean())

    precision_v = precision_psnr(matrix)
    recall_v = recall_psnr(matrix)
    emd_v = emd_psnr(matrix)

    print("Computing intra PSNR matrices for k-NN precision/recall...")
    gt_intra = psnr_matrix(gt, gt)
    gen_intra = psnr_matrix(gen, gen)
    k_p, k_r = k_precision_recall_psnr(matrix, gt_intra, gen_intra, k=args.k)

    print(f"paired_psnr: {paired_psnr}")
    print(f"precision_psnr: {precision_v}, recall_psnr: {recall_v}, emd_psnr: {emd_v}")
    print(f"K_precision: {k_p}, K_recall: {k_r}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint_path', type=str, required=True)
    p.add_argument('--num_samples', type=int, default=0,
                   help='0 = use model.num_images (one sample per GT image)')
    p.add_argument('--k', type=int, default=3)
    p.add_argument('--run_name', type=str, default='mnist-eval')
    args = p.parse_args()
    main(args)
