from collections.abc import Mapping

import torch
from torch import nn
from learning.models.encoder import ObservationEncoder

class TransformerEncoder(ObservationEncoder):
    """
    transformer encoder;
    maps a variable-size set of visible neighbour observations to one fixed-size embedding
    """

    def __init__(
        self,
        state_dim: int,
        neighbor_feature_dim: int,
        neighbor_slots: int,
        observation_horizon: int = 1,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if state_dim <= 0 or neighbor_feature_dim <= 0 or neighbor_slots < 0:
            raise ValueError("Transformer encoder dimensions must be valid.")
        if observation_horizon <= 0:
            raise ValueError("'observation_horizon' must be positive.")
        self.neighbor_feature_dim = int(neighbor_feature_dim)
        self.neighbor_slots = int(neighbor_slots)
        self.observation_horizon = int(observation_horizon)
        self.ego_dim = self._compute_ego_dim(
            int(state_dim), self.neighbor_slots, self.neighbor_feature_dim, self.observation_horizon
        )

        # no positional encoding to obtain a permutation invariant embedding

        augmented_neighbor_feature_dim = self._augmented_neighbor_feature_dim(
            self.neighbor_feature_dim, self.observation_horizon
        )
        self.input_projection = nn.Linear(augmented_neighbor_feature_dim, hidden_dim)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=2 * hidden_dim,
                dropout=dropout,
                batch_first=True,  # input shape: [batch, sequence, features]
            ),
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        self.pool_token = nn.Parameter(
            torch.zeros(1, 1, hidden_dim)
        )

        self._out_dim = self.ego_dim + hidden_dim

    @property
    def out_dim(self) -> int:
        return self._out_dim

    def forward(self, observation_dict: Mapping[str, torch.Tensor]) -> torch.Tensor:
        environment_state = observation_dict["observation.environment_state"]
        state = observation_dict["observation.state"]
        neighbor_state = observation_dict["observation.neighbor_state"]
        neighbor_mask = observation_dict["observation.neighbor_mask"]
        ego_obs = torch.cat([environment_state, state], dim=-1)
        batch_size = ego_obs.shape[0]
        neighbor_obs, neighbor_mask = self._split_neighbor_tensors(
            neighbor_state, neighbor_mask, self.neighbor_feature_dim, self.observation_horizon
        )
        neighbor_features = self._augment_with_temporal_mask(neighbor_obs, neighbor_mask)
        neighbor_mask = neighbor_mask.bool()

        x = self.input_projection(neighbor_features)

        # adds learnable token to beginning of sequence to create fixed size embedding:
        pool_token = self.pool_token.expand(x.shape[0], -1, -1)
        x = torch.cat([pool_token, x], dim=1)   # [B, N + 1, hidden_dim]

        neighbor_mask = neighbor_mask[:, :, -1]

        # converts neighbor_mask to transformer padding mask (true = ignored):
        padding_mask = ~torch.cat(
            [
                torch.ones(
                    (x.shape[0], 1),    # adds mask entry for new pool_token
                    dtype=torch.bool,
                    device=x.device,
                ),
                neighbor_mask,
            ],
            dim=1,
        )

        x = self.encoder(x, src_key_padding_mask=padding_mask)

        # returns zero embedding if no visible neighbour:
        has_visible_neighbor = neighbor_mask.any(dim=1)
        neighbor_context = x[:, 0].masked_fill(
            ~has_visible_neighbor.unsqueeze(-1),
            0.0,
        )
        return torch.cat([ego_obs, neighbor_context], dim=-1)