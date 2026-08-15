import torch
import torch.nn as nn

class TransformerEncoder(nn.Module):
    """
    transformer encoder; takes a variable size set of relative neighbour observations
    and turns it into one fixed size embedding (can later be concatenated with robot i's own state and goal)
    """

    def __init__(self, input_dim, hidden_dim=64, num_heads=4, num_layers=1):
        super().__init__()

        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # no positional encoding to obtain a permutation invariant embedding
        
        # transformer layer for interaction between visible neighbours:
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=2 * hidden_dim,
            batch_first=True,   # input shape: [batch, sequence, features]
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # learnable token to create fixed size embedding:
        self.pool_token = nn.Parameter(
            torch.zeros(1, 1, hidden_dim)
        )

    def forward(self, neighbour_obs, neighbour_mask=None):
        
        # projects neighbour observations to hidden dimension:
        x = self.input_projection(neighbour_obs)

        # adds learnable token to beginning of sequence to create fixed size embedding:
        pool_token = self.pool_token.expand(x.shape[0], -1, -1)
        x = torch.cat([pool_token, x], dim=1)   # [B, N + 1, 64]

        if neighbour_mask is not None:
            # converts to transformer padding mask (true = ignored):            
            neighbour_mask = ~torch.cat(
                [
                    torch.ones(
                        (x.shape[0], 1),    # adds mask entry for new pool_token
                        dtype=torch.bool,
                        device=x.device,
                    ),
                    neighbour_mask,
                ],
                dim=1,
            )
    
        x = self.encoder(x, src_key_padding_mask=neighbour_mask)  
        return x[:, 0]