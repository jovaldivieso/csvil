import os
import sys
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from learning.transformer import TransformerEncoder


def main():
    model = TransformerEncoder(input_dim=2, hidden_dim=64, num_heads=4, num_layers=1)
    model.eval()

    # random test values:
    neighbour_obs = torch.tensor([
        [
            [0.8, -0.3],
            [-1.2, 0.5],
            [0.0, 0.0],
        ],
        [
            [0.4, 0.7],
            [0.0, 0.0],
            [0.0, 0.0],
        ],
    ])
    neighbour_mask = torch.tensor([
        [True, True, False],
        [True, False, False],
    ])

    with torch.no_grad():
        embedding = model(neighbour_obs, neighbour_mask)

    print("input shape:", neighbour_obs.shape)
    print("embedding shape:", embedding.shape)

    assert embedding.shape == (2, 64)

    # checks that masked neighbours do not influence embedding:
    obs_changed = neighbour_obs.clone()
    obs_changed[0, 2] = torch.tensor([100.0, -100.0])

    with torch.no_grad():
        embedding_changed = model(obs_changed, neighbour_mask)

    assert torch.allclose(embedding[0], embedding_changed[0], atol=1e-5)

    # checks permutation invariance:
    permutation = torch.tensor([1, 0, 2])

    with torch.no_grad():
        embedding_permuted = model(
            neighbour_obs[:, permutation],
            neighbour_mask[:, permutation],
        )

    assert torch.allclose(embedding, embedding_permuted, atol=1e-5)
   
    print("permutation invariance and neighbour mask test passed")


if __name__ == "__main__":
    main()