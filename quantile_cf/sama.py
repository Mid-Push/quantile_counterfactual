"""Optimizer-aware scaling used by the SAMA hypergradient (Choe et al., 2023)."""

import torch


def precondition(vectors, params, optimizer):
    """Scale the outer gradient with respect to the outcome network before the finite difference.

    For Adam this is ``lr * (1 - beta1) * v / (sqrt(v) + eps) ** 3`` per entry, where ``v`` is
    Adam's second-moment estimate: SAMA's Adam correction with the previous gradient taken as
    zero, as in the Betty implementation used for the paper. Other optimizers are treated as SGD
    (no scaling).
    """
    if not isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW)):
        return vectors
    scaled = []
    for vector, param in zip(vectors, params):
        group = next(g for g in optimizer.param_groups if any(p is param for p in g["params"]))
        beta1, beta2 = group["betas"]
        with torch.no_grad():
            exp_avg_sq = optimizer.state[param].get("exp_avg_sq", torch.zeros_like(vector))
            scale = (1 - beta1) * beta2 * (exp_avg_sq / beta2)
            scale /= (torch.sqrt(exp_avg_sq) + group["eps"]) ** 3
        scaled.append(vector * scale * group["lr"])
    return scaled
