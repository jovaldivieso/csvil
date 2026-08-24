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
        in_features: int,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # no positional encoding to obtain a permutation invariant embedding
        
        self.input_projection = nn.Linear(in_features, hidden_dim)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=2 * hidden_dim,
                dropout=dropout,
                batch_first=True,  # input shape: [batch, sequence, features]
            ),
            num_layers=num_layers,
        )

        self.pool_token = nn.Parameter(
            torch.zeros(1, 1, hidden_dim)
        )
        
        self._out_dim = hidden_dim
        
    @property
    def out_dim(self) -> int:
        return self._out_dim

    def forward(
        self,
        neighbor_obs: torch.Tensor,
        neighbor_mask: torch.Tensor,
    ) -> torch.Tensor:
        
        x = self.input_projection(neighbor_obs)

        # adds learnable token to beginning of sequence to create fixed size embedding:
        pool_token = self.pool_token.expand(x.shape[0], -1, -1)
        x = torch.cat([pool_token, x], dim=1)   # [B, N + 1, hidden_dim]

        neighbor_mask = neighbor_mask.bool().squeeze(-1)
        
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
        return x[:, 0].masked_fill(
            ~has_visible_neighbor.unsqueeze(-1),
            0.0,
        )