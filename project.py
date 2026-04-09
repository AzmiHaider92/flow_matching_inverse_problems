import torch
import torch.nn.functional as F


########## the function y = f(X)
# given an image X, get a patch y
# each iteration - two possibilites:
#                  (1) get a random patch from the image - consistent_patch=False
#                  (2) get the same patch from the image - consistent_patch=True


def get_random_foreground_indices(images, patch_size):
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


def get_consistent_foreground_indices(images, patch_size):
    B, _, H, W = images.shape
    device = images.device

    y_idxs = []
    x_idxs = []

    for i in range(B):
        # Create a deterministic seed based on the image content
        # We use a large prime and the sum to create a pseudo-unique seed
        seed = int(torch.sum(images[i] * 1000).item()) % (2 ** 32)
        gen = torch.Generator(device=device).manual_seed(seed)

        mask = (images[i].view(-1) > 0.1).float()

        # Pick foreground pixels using the seeded generator
        idx = torch.multinomial(mask + 1e-8, 1, generator=gen).item()
        py, px = idx // W, idx % W

        # Add "random" offsets using the same seeded generator
        off_y = torch.randint(-patch_size + 1, 1, (1,), device=device, generator=gen).item()
        off_x = torch.randint(-patch_size + 1, 1, (1,), device=device, generator=gen).item()

        y_idxs.append(max(0, min(py + off_y, H - patch_size)))
        x_idxs.append(max(0, min(px + off_x, W - patch_size)))

    return torch.tensor(y_idxs, device=device).long(), torch.tensor(x_idxs, device=device).long()


def get_foreground_indices(images, patch_size, consistent_patch=False):
    if consistent_patch:
        y_idxs, x_idxs = get_consistent_foreground_indices(images, patch_size)
    else:
        y_idxs, x_idxs = get_random_foreground_indices(images, patch_size)
    return y_idxs, x_idxs


def f_project(X, y_idx, x_idx, patch_size):
    B, C, H, W = X.shape
    patches = F.unfold(X, kernel_size=patch_size)
    stride_w = W - patch_size + 1
    linear_indices = y_idx * stride_w + x_idx
    linear_indices = linear_indices.view(B, 1, 1).expand(-1, patch_size ** 2, -1)
    return patches.gather(2, linear_indices).squeeze(-1)


#------------------------------------------------------------------------------
# Random projection x @ A - where A is the same random matrix for image x


#def get_random_projection_matrix(total_pixels, n_measurements, device, seed=42):
#    gen = torch.Generator(device=device).manual_seed(seed)
#    # Using orthogonal initialization often performs better for reconstruction
#    # but a simple rand or randn works for basic Flow Matching guidance
#    A = torch.randn(total_pixels, n_measurements, generator=gen, device=device)
#    # Normalize columns to unit length to keep loss scales consistent
#    A = A / torch.norm(A, dim=0, keepdim=True)
#    return A


def get_batch_unique_matrices(images, n_measurements):
    """
    Returns A of shape (B, 784, M) where A[i] is unique to images[i]
    """
    B = images.shape[0]
    pixels = 784
    device = images.device
    A_batch = []

    for i in range(B):
        # Create a unique seed for this specific image content
        seed = int(torch.sum(images[i] * 1000).item()) % (2 ** 32)
        gen = torch.Generator(device=device).manual_seed(seed)

        Ai = torch.randn(pixels, n_measurements, generator=gen, device=device)
        Ai = Ai / torch.norm(Ai, dim=0, keepdim=True)
        A_batch.append(Ai)

    return torch.stack(A_batch)  # (B, 784, M)


def f_random_batch_projection(X, A_batch, noise_std=1e-3):
    """
    X: (B, 1, 28, 28)
    A_batch: (B, 784, M)
    Returns y: (B, M) where y[i] = X[i]_flat @ A_batch[i] + noise
    """
    B = X.shape[0]
    x_flat = X.view(B, 1, -1)  # (B, 1, 784)
    # Batch matrix multiplication: (B, 1, 784) @ (B, 784, M) -> (B, 1, M)
    y = torch.bmm(x_flat, A_batch)
    y = y.squeeze(1)  # (B, M)
    y = y + noise_std * torch.randn_like(y)
    return y
