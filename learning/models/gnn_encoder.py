import torch
from torch import nn

from learning.models.encoder import ObservationEncoder


class GNNEncoder(ObservationEncoder):
    """
    message-passing encoder over the ego robot's visible-neighbour star;
    maps a variable-size set of visible neighbour observations to one fixed-size embedding

    Per round every visible neighbour j contributes msg(h_i, e_ij); the messages are summed
    and folded back into h_i by upd(h_i, aggregated). Computed dense over the padded slots,
    which is equivalent to scatter-add over a star graph but needs no graph bookkeeping.

    The observation carries no neighbour node feature x_j, only the relative offset e_ij, so
    a message depends on the receiver and the offset alone. num_layers > 1 therefore adds
    depth, not communication hops.
    """

    def __init__(
        self,
        in_features: int,
        ego_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
    ) -> None:
        super().__init__()

        if in_features <= 0:
            raise ValueError(f"'in_features' must be positive, got {in_features}.")
        if ego_dim <= 0:
            raise ValueError(f"'ego_dim' must be positive, got {ego_dim}.")
        if num_layers <= 0:
            raise ValueError(f"'num_layers' must be positive, got {num_layers}.")

        self.in_features = int(in_features)
        self.ego_dim = int(ego_dim)

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
                nn.Linear(hidden_dim + self.in_features, hidden_dim),
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

        self._out_dim = hidden_dim

    @property
    def out_dim(self) -> int:
        return self._out_dim

    def forward(self, neighbors_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        missing = [
            name
            for name in ("ego_obs", "neighbor_obs", "neighbor_mask")
            if name not in neighbors_dict
        ]
        if missing:
            raise KeyError(f"'neighbors_dict' is missing required entries: {missing}.")

        ego_obs = neighbors_dict["ego_obs"]              # [B, ego_dim]
        neighbor_obs = neighbors_dict["neighbor_obs"]    # [B, K, in_features] - in_features = edge_features
        neighbor_mask = neighbors_dict["neighbor_mask"]  # [B, K, 1]

        if neighbor_obs.ndim != 3 or neighbor_obs.shape[2] != self.in_features:
            raise ValueError(
                f"'neighbor_obs' must have shape (B, K, {self.in_features}), "
                f"got {tuple(neighbor_obs.shape)}."
            )
        if tuple(neighbor_mask.shape) != tuple(neighbor_obs.shape[:2]) + (1,):
            raise ValueError(
                f"'neighbor_mask' must have shape {tuple(neighbor_obs.shape[:2]) + (1,)}, "
                f"got {tuple(neighbor_mask.shape)}."
            )
        if tuple(ego_obs.shape) != (neighbor_obs.shape[0], self.ego_dim):
            raise ValueError(
                f"'ego_obs' must have shape {(neighbor_obs.shape[0], self.ego_dim)}, "
                f"got {tuple(ego_obs.shape)}."
            )

        # The mask selects the visible neighbours. Listing every visible (robot, slot) pair
        # gives E rows -- one per observed neighbour across the whole batch -- so padded
        # slots are never fed to an MLP. Both lists are constant across rounds.
        visible_neighbors = neighbor_mask.bool().squeeze(-1) # [batch, num_neighbors]
        batch_index, slot_index = visible_neighbors.nonzero(as_tuple=True) # [E], [E] indexing into the batch and slot dimensions of neighbor_obs
        neighbor_j = neighbor_obs[batch_index, slot_index]  # [E, in_features] - vector of e_ij

        h = self.encoder(ego_obs)  # [batch_size, hidden_dim]
        # loop is if we add communication hops (num_layers > 1), right now only num_layers = 1
        for msg_mlp, upd_mlp in zip(self.msg_mlps, self.upd_mlps):
            # compute messages from each visible neighbour to its ego robot
            msg_input = torch.cat([h[batch_index], neighbor_j], dim=-1)  # [E, hidden_dim + in_features]
            messages = msg_mlp(msg_input) # [E, hidden_dim] - one message per visible neighbour
            # sum messages per ego robot, using the batch_index
            aggregated = torch.zeros_like(h).index_add_(0, batch_index, messages)   # [batch_size, hidden_dim] - sum of messages per ego robot
            # one message per observed neighbour: phi([h_i, e_ij])
            h = upd_mlp(torch.cat([h, aggregated], dim=-1))

        # zero embedding if no visible neighbour, matching deepset/transformer
        return h.masked_fill(
            ~visible_neighbors.any(dim=1).unsqueeze(-1),
            0.0,
        )
