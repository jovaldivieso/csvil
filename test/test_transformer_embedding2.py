import os
import sys
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, TensorDataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from learning.transformer import TransformerEncoder


NUM_ROBOTS = 2

class TransformerPolicy(nn.Module):
    """
    transformer embedding with mlp policy head
    """

    def __init__(self, hidden_dim=64):
        super().__init__()

        self.embedding = TransformerEncoder(
            input_dim=2,
            hidden_dim=hidden_dim,
            num_heads=4,
            num_layers=1,
        )

        self.policy_head = nn.Sequential(
            nn.Linear(4 + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, ego_obs, neighbour_obs, neighbour_mask):
        # combines ego_obs and neighbour_obs embedding:
        embedding = self.embedding(neighbour_obs, neighbour_mask)
        x = torch.cat([ego_obs, embedding], dim=-1)
        return self.policy_head(x)


def frame_to_samples(frame):
    # converts a 2 robot dataset frame into one sample per robot

    env = frame["observation.environment_state"].to(torch.float32)
    state = frame["observation.state"].to(torch.float32)
    action = frame["action"].to(torch.float32)

    goal_rel = torch.stack([env[0:2], env[4:6]])
    velocity = torch.stack([state[0:2], state[2:4]])
    ego_obs = torch.cat([goal_rel, velocity], dim=-1)

    neighbour_obs = torch.stack([env[2:4], env[6:8]]).unsqueeze(1)
    neighbour_mask = torch.any(torch.abs(neighbour_obs) > 1e-9, dim=-1)

    target = action.reshape(NUM_ROBOTS, 2)

    return ego_obs, neighbour_obs, neighbour_mask, target


def build_dataset(root, val_split=0.2, seed=0):
    dataset = LeRobotDataset(repo_id="local", root=root)

    num_episodes = len(dataset.meta.episodes)

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(num_episodes, generator=generator).tolist()

    num_val = max(1, int(round(num_episodes * val_split)))
    val_episodes = set(permutation[:num_val])

    train_samples = []
    val_samples = []

    for episode in dataset.meta.episodes:
        target = (
            val_samples
            if episode["episode_index"] in val_episodes
            else train_samples
        )

        for frame_idx in range(
            episode["dataset_from_index"],
            episode["dataset_to_index"],
        ):
            ego_obs, neighbour_obs, neighbour_mask, action = (
                frame_to_samples(dataset[frame_idx])
            )

            for robot_idx in range(NUM_ROBOTS):
                target.append((
                    ego_obs[robot_idx],
                    neighbour_obs[robot_idx],
                    neighbour_mask[robot_idx],
                    action[robot_idx],
                ))

    return create_tensor_dataset(train_samples), create_tensor_dataset(val_samples)


def create_tensor_dataset(samples):
    return TensorDataset(
        torch.stack([sample[0] for sample in samples]),
        torch.stack([sample[1] for sample in samples]),
        torch.stack([sample[2] for sample in samples]),
        torch.stack([sample[3] for sample in samples]),
    )


def run_epoch(model, loader, optimizer=None):
    training = optimizer is not None

    model.train(training)

    total_loss = 0.0

    for ego_obs, neighbour_obs, neighbour_mask, action in loader:
        pred = model(ego_obs, neighbour_obs, neighbour_mask)

        loss = F.mse_loss(pred, action)

        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * ego_obs.shape[0]

    return total_loss / len(loader.dataset)


def main():
    torch.manual_seed(0)
    
    parser =argparse.ArgumentParser()
    parser.add_argument("path_to_dataset", help="path to a (2 robot) dataset root directory, e.g. 'data/lerobot_dataset_multi_robot_casadi_1786802790'")

    args = parser.parse_args()
    dataset_root = args.path_to_dataset

    train_set, valid_set = build_dataset(dataset_root)

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=32)

    model = TransformerPolicy(hidden_dim=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, 30):
        train_loss = run_epoch(model, train_loader, optimizer)

        with torch.no_grad():
            valid_loss = run_epoch(model, valid_loader)

        print(f"epoch {epoch:2d} | \n train MSE {train_loss:.4f} | \n valid MSE {valid_loss:.4f}")


if __name__ == "__main__":
    main()