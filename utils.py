import torch


def get_cfg_velocity(model, x, t, labels, cfg_scale=3.0, null_label=10):
    v_cond = model(x, t, labels)
    if cfg_scale == 1.0: return v_cond
    v_uncond = model(x, t, torch.full_like(labels, null_label))
    return v_uncond + cfg_scale * (v_cond - v_uncond)