"""
Scatter operations for tensors.

Based on: https://github.com/rusty1s/pytorch_scatter to avoid installing the package.

Modifications: use torch.scatter_reduce_ instead of torch.scatter_.
"""

from typing import Optional

import torch


def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand(other.size())
    return src


def scatter(
    src: torch.Tensor,
    index: torch.Tensor,
    reduce: str,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
) -> torch.Tensor:
    """
    A wrapper around torch.scatter_reduce_ that creates output tensor if not provided.

    The only difference is that this wrapper creates the output tensor if not provided.

    See: https://pytorch.org/docs/stable/generated/torch.scatter_reduce.html

    Argues:
        reduce: "sum", "prod", "mean", "amax", "amin".
            Note, "amax"="max" and "amin"="min".
    """

    # index should have the same shape as src as that backward pass works,
    # per the PyTorch docs.
    index = broadcast(index, src, dim)

    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)

    # setting include_self=False so that
    return out.scatter_reduce_(dim, index, src, reduce, include_self=False)
