from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping

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
        state_dim: int,
        neighbor_feature_dim: int,
        neighbor_slots: int,
        phi_dims: Iterable[int] = (128, 128),
        rho_dims: Iterable[int] = (128,),
        pool_type: str = "max",
    ) -> None:
        super().__init__()

        if state_dim <= 0:
            raise ValueError(f"'state_dim' must be positive, got {state_dim}.")
        if neighbor_feature_dim <= 0:
            raise ValueError(f"'neighbor_feature_dim' must be positive, got {neighbor_feature_dim}.")
        if neighbor_slots < 0:
            raise ValueError(f"'neighbor_slots' must be non-negative, got {neighbor_slots}.")
        if pool_type not in {"sum", "max", "mean"}:
            raise ValueError("pool_type must be one of {'sum', 'max', 'mean'}")

        self.state_dim = int(state_dim)
        self.neighbor_feature_dim = int(neighbor_feature_dim)
        self.neighbor_slots = int(neighbor_slots)
        self.ego_dim = self.state_dim - self.neighbor_slots * (self.neighbor_feature_dim + 1)
        if self.ego_dim <= 0:
            raise ValueError("'state_dim' is too small for the packed observation layout.")

        phi_dims_list = [self.neighbor_feature_dim, *[int(width) for width in phi_dims]]
        rho_dims_list = [phi_dims_list[-1], *[int(width) for width in rho_dims]]

        if len(phi_dims_list) < 2:
            raise ValueError("'phi_dims' must contain at least one output width.")
        if len(rho_dims_list) < 2:
            raise ValueError("'rho_dims' must contain at least one output width.")

        self.pool_type = pool_type
        self.phi = _build_mlp(phi_dims_list)
        self.rho = _build_mlp(rho_dims_list)
        self._out_dim = self.ego_dim + int(rho_dims_list[-1])

    @property
    def out_dim(self) -> int:
        return self._out_dim

    def forward(self, observation_dict: Mapping[str, torch.Tensor]) -> torch.Tensor:
        environment_state = observation_dict.get("observation.environment_state")
        state = observation_dict.get("observation.state")
        raw_neighbors = observation_dict.get("observation.neighbor_state")
        raw_mask = observation_dict.get("observation.neighbor_mask")
        if environment_state is None or state is None or raw_neighbors is None or raw_mask is None:
            raise ValueError(
                "observation_dict must contain canonical environment, state, neighbor state, and mask keys."
            )
        if environment_state.ndim != 2 or state.ndim != 2:
            raise ValueError("Environment and state tensors must have shape (B, D).")
        ego_obs = torch.cat([environment_state, state], dim=-1)
        if ego_obs.ndim != 2 or ego_obs.shape[1] != self.ego_dim:
            raise ValueError(
                f"Canonical ego features must concatenate to shape (B, {self.ego_dim}), "
                f"got {tuple(ego_obs.shape)}."
            )
        if raw_neighbors.ndim != 2 or raw_mask.ndim != 2:
            raise ValueError("Canonical neighbor tensors must have shape (B, D).")
        batch_size = environment_state.shape[0]
        if state.shape[0] != batch_size or raw_neighbors.shape[0] != batch_size or raw_mask.shape[0] != batch_size:
            raise ValueError("Canonical observation tensors must agree on batch dimension.")
        try:
            neighbor_obs = raw_neighbors.view(
                batch_size, -1, self.neighbor_feature_dim
            )
            neighbor_mask = raw_mask.view(batch_size, -1, 1)
        except RuntimeError as exc:
            raise ValueError(
                "Flat neighbor tensors do not match the encoder's configured feature dimensions."
            ) from exc

        max_items = neighbor_obs.shape[1]
        if max_items == 0:
            neighbor_context = torch.zeros(
                (batch_size, self.out_dim - self.ego_dim),
                device=neighbor_obs.device,
                dtype=neighbor_obs.dtype,
            )
            return torch.cat([ego_obs, neighbor_context], dim=-1)

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
        context = context.masked_fill(~row_has_visible.unsqueeze(-1), 0.0)
        return torch.cat([ego_obs, context], dim=-1)

