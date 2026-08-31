from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from learning.models.encoder import ObservationEncoder


class GNNEncoder(ObservationEncoder):
    """
    message-passing encoder over the ego robot's visible-neighbour star;
    maps a variable-size set of visible neighbour observations to one fixed-size embedding

    Per round every visible neighbour j contributes msg(h_i, e_ij); the messages are summed
    and folded back into h_i by upd(h_i, aggregated). Only visible (robot, slot) pairs are
    gathered, which is equivalent to scatter-add over a star graph but needs no graph
    bookkeeping.

    The observation carries no neighbour node feature x_j, only the relative offset e_ij, so
    a message depends on the receiver and the offset alone. num_layers > 1 therefore adds
    depth, not communication hops.
    """

    def __init__(
        self,
        state_dim: int,
        neighbor_feature_dim: int,
        neighbor_slots: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
    ) -> None:
        super().__init__()

        if state_dim <= 0:
            raise ValueError(f"'state_dim' must be positive, got {state_dim}.")
        if neighbor_feature_dim <= 0:
            raise ValueError(
                f"'neighbor_feature_dim' must be positive, got {neighbor_feature_dim}."
            )
        if neighbor_slots < 0:
            raise ValueError(f"'neighbor_slots' must be non-negative, got {neighbor_slots}.")
        if hidden_dim <= 0:
            raise ValueError(f"'hidden_dim' must be positive, got {hidden_dim}.")
        if num_layers <= 0:
            raise ValueError(f"'num_layers' must be positive, got {num_layers}.")

        self.state_dim = int(state_dim)
        self.neighbor_feature_dim = int(neighbor_feature_dim)
        self.neighbor_slots = int(neighbor_slots)
        self.ego_dim = self.state_dim - self.neighbor_slots * (self.neighbor_feature_dim + 1)
        if self.ego_dim <= 0:
            raise ValueError("'state_dim' is too small for the packed observation layout.")

        # encoder of local ego observation to embedding h_i [H]
        self.encoder = nn.Sequential(
            nn.Linear(self.ego_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # message encoding
        # currently only edge feature e_ij (will extend maybe later for x_j if we add neighbour node features)
        self.msg_mlps = nn.ModuleList(
            nn.Sequential(
                nn.Linear(hidden_dim + self.neighbor_feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(num_layers)
        )

        # aggregate encoding
        self.upd_mlps = nn.ModuleList(
            nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(num_layers)
        )

        self._out_dim = self.ego_dim + hidden_dim

    @property
    def out_dim(self) -> int:
        return self._out_dim

    def forward(self, observation_dict: Mapping[str, torch.Tensor]) -> torch.Tensor:
        environment_state = observation_dict["observation.environment_state"]
        state = observation_dict["observation.state"]
        raw_neighbors = observation_dict["observation.neighbor_state"]
        raw_mask = observation_dict["observation.neighbor_mask"]
        ego_obs = torch.cat([environment_state, state], dim=-1)  # [batch_size, ego_dim]
        batch_size = ego_obs.shape[0]
        neighbor_obs = raw_neighbors.view(batch_size, -1, self.neighbor_feature_dim)  # [batch_size, K, neighbor_feature_dim] - one e_ij per slot
        neighbor_mask = raw_mask.view(batch_size, -1)  # [batch_size, K]s

        # The mask selects the visible neighbours. Listing every visible (robot, slot) pair
        # gives E rows -- one per observed neighbour across the whole batch -- so padded
        # slots are never fed to an MLP. Both lists are constant across rounds.
        visible_neighbors = neighbor_mask.bool()  # [B, K]
        batch_index, slot_index = visible_neighbors.nonzero(as_tuple=True)  # [E], [E] indexing into the batch and slot dimensions of neighbor_obs
        # each (batch_index, slot_index) pair is one visible neighbour
        neighbor_j = neighbor_obs[batch_index, slot_index]  # [E, neighbor_feature_dim] - vector of e_ij

        h = self.encoder(ego_obs)  # [B, hidden_dim]
        # loop is if we add communication hops (num_layers > 1), right now only num_layers = 1
        for msg_mlp, upd_mlp in zip(self.msg_mlps, self.upd_mlps):
            # compute messages from each visible neighbour to its ego robot
            msg_input = torch.cat([h[batch_index], neighbor_j], dim=-1)  # [E, hidden_dim + neighbor_feature_dim]
            messages = msg_mlp(msg_input)  # [E, hidden_dim] - one message per visible neighbour
            aggregated = torch.zeros_like(h).index_add_(0, batch_index, messages)  # [B, hidden_dim]
            # one message per observed neighbour: phi([h_i, e_ij])
            h = upd_mlp(torch.cat([h, aggregated], dim=-1))

        # raw ego features are kept alongside the message-passing embedding, matching the
        # deepset/transformer contract of out_dim = ego_dim + context
        return torch.cat([ego_obs, h], dim=-1)
