from __future__ import annotations

from typing import Iterable

import torch
from torch import nn


def build_mlp(dims: list[int], activation: nn.Module = nn.ReLU()) -> nn.Sequential:
    if len(dims) < 2:
        raise ValueError("'dims' must contain at least an input and output width.")

    layers: list[nn.Module] = []
    for layer_idx in range(len(dims) - 1):
        in_dim = int(dims[layer_idx])
        out_dim = int(dims[layer_idx + 1])
        if in_dim <= 0 or out_dim <= 0:
            raise ValueError("MLP layer widths must be positive integers.")

        layers.append(nn.Linear(in_dim, out_dim))
        if layer_idx < len(dims) - 2:
            layers.append(activation)

    return nn.Sequential(*layers)


class DeepSetEncoder(nn.Module):
    def __init__(
        self,
        in_features: int,
        phi_dims: Iterable[int],
        rho_dims: Iterable[int],
        pool_type: str = "max",
    ) -> None:
        super().__init__()

        if in_features <= 0:
            raise ValueError(f"'in_features' must be positive, got {in_features}.")
        if pool_type not in {"sum", "max", "mean"}:
            raise ValueError("pool_type must be one of {'sum', 'max', 'mean'}")

        phi_dims_list = [int(in_features), *[int(width) for width in phi_dims]]
        rho_dims_list = [phi_dims_list[-1], *[int(width) for width in rho_dims]]

        if len(phi_dims_list) < 2:
            raise ValueError("'phi_dims' must contain at least one output width.")
        if len(rho_dims_list) < 2:
            raise ValueError("'rho_dims' must contain at least one output width.")

        self.pool_type = pool_type
        self.phi = build_mlp(phi_dims_list)
        self.rho = build_mlp(rho_dims_list)
        self.out_dim = int(rho_dims_list[-1])

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"'x' must have shape (B, K, D), got {tuple(x.shape)}.")
        if mask.ndim != 3:
            raise ValueError(f"'mask' must have shape (B, K, 1), got {tuple(mask.shape)}.")
        if x.shape[0] != mask.shape[0] or x.shape[1] != mask.shape[1]:
            raise ValueError(
                "'x' and 'mask' must agree on batch and neighbor dimensions: "
                f"got {tuple(x.shape)} and {tuple(mask.shape)}."
            )

        batch_size, max_items, _ = x.shape
        if max_items == 0:
            return torch.zeros((batch_size, self.out_dim), device=x.device, dtype=x.dtype)

        mask_bool = mask.bool()
        if mask_bool.numel() == 0 or not bool(mask_bool.any().item()):
            return torch.zeros((batch_size, self.out_dim), device=x.device, dtype=x.dtype)

        phi_out = self.phi(x)

        if self.pool_type == "max":
            phi_out = phi_out.masked_fill(~mask_bool, float("-inf"))
            pooled_out, _ = torch.max(phi_out, dim=1)
            pooled_out = torch.nan_to_num(pooled_out, neginf=0.0)
        elif self.pool_type == "sum":
            phi_out = phi_out.masked_fill(~mask_bool, 0.0)
            pooled_out = torch.sum(phi_out, dim=1)
        else:
            phi_out = phi_out.masked_fill(~mask_bool, 0.0)
            summed = torch.sum(phi_out, dim=1)
            valid_counts = mask_bool.sum(dim=1).clamp(min=1).to(dtype=x.dtype)
            pooled_out = summed / valid_counts

        return self.rho(pooled_out)