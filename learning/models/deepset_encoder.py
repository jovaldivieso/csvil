from __future__ import annotations

from collections.abc import Callable, Iterable

import torch
from torch import nn

from learning.models.encoder import ObservationEncoder


def _build_mlp(
    dims: list[int],
    activation_factory: Callable[[], nn.Module] = nn.ReLU,
) -> nn.Sequential:
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
            layers.append(activation_factory())
    return nn.Sequential(*layers)

class DeepSetEncoder(ObservationEncoder):
    def __init__(
        self,
        in_features: int,
        phi_dims: Iterable[int] = (128, 128),
        rho_dims: Iterable[int] = (128,),
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
        self.phi = _build_mlp(phi_dims_list)
        self.rho = _build_mlp(rho_dims_list)
        self._out_dim = int(rho_dims_list[-1])

    @property
    def out_dim(self) -> int:
        return self._out_dim

    def forward(self, neighbor_obs: torch.Tensor, neighbor_mask: torch.Tensor) -> torch.Tensor:
        if neighbor_obs.ndim != 3:
            raise ValueError(f"'neighbor_obs' must have shape (B, K, D), got {tuple(neighbor_obs.shape)}.")
        if neighbor_mask.ndim != 3 or neighbor_mask.shape[2] != 1:
            raise ValueError(f"'neighbor_mask' must have shape (B, K, 1), got {tuple(neighbor_mask.shape)}.")
        if neighbor_obs.shape[0] != neighbor_mask.shape[0] or neighbor_obs.shape[1] != neighbor_mask.shape[1]:
            raise ValueError(
                "'neighbor_obs' and 'neighbor_mask' must agree on batch and neighbor dimensions."
            )

        batch_size, max_items, _ = neighbor_obs.shape
        if max_items == 0:
            return torch.zeros((batch_size, self.out_dim), device=neighbor_obs.device, dtype=neighbor_obs.dtype)

        mask_bool = neighbor_mask.bool()
        row_has_visible = mask_bool.any(dim=(1, 2))

        phi_out = self.phi(neighbor_obs)

        if self.pool_type == "max":
            phi_out = phi_out.masked_fill(~mask_bool, float("-inf"))
            pooled_out, _ = torch.max(phi_out, dim=1)
            pooled_out = torch.nan_to_num(pooled_out, neginf=0.0)
        elif self.pool_type == "sum":
            pooled_out = torch.sum(phi_out.masked_fill(~mask_bool, 0.0), dim=1)
        else:
            summed = torch.sum(phi_out.masked_fill(~mask_bool, 0.0), dim=1)
            valid_counts = mask_bool.sum(dim=1).clamp(min=1).to(dtype=neighbor_obs.dtype)
            pooled_out = summed / valid_counts

        context = self.rho(pooled_out)
        return context.masked_fill(~row_has_visible.unsqueeze(-1), 0.0)

