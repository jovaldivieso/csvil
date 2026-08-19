"""Message-passing GNN policy for decentralized multi-robot imitation learning.

Lives next to models/mlp.py so both the trainer (learning/train_gnn.py) and the
evaluator (test/evaluate_policy.py) import the SAME model definition and the SAME
frame -> graph conversion. Sharing `frame_to_graph` is deliberate: if training
built graphs one way and rollout another, the policy would silently regress
instead of failing loudly.

Graph layout (see learning/train_gnn.py for the full rationale):
  nodes = robots, node feature = [goal_rel_x, goal_rel_y, <own proprioception>]
  edges = "robot j is currently visible to robot i", edge feature = pos_j - pos_i
  label / output = per-robot action slice
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing


# Zeroed by systems/multi_robot.py::observe() whenever the true offset exceeds
# inter_robot_visibility_radius; a real offset is astronomically unlikely to
# land exactly on 0.0, so "all-zero" is a safe stand-in for "not visible".
VISIBILITY_ZERO_EPS = 1e-9


# --------------------------------------------------------------------------- #
# One message-passing layer (uses edge features; sum-aggregates neighbours)     #
# --------------------------------------------------------------------------- #
class MPLayer(MessagePassing):
    def __init__(self, hidden: int, edge_in: int):
        # aggr='add': a node with ZERO neighbours gets a zero message vector, and
        # its own embedding still flows through update(), so it can still act
        # (e.g. head straight to goal). 'mean'/'max' would divide-by-zero / -inf here.
        super().__init__(aggr="add")
        # message MLP input:  own emb x_i [H] + neighbour emb x_j [H] | edge feat [edge_in]
        # edge_in: relative observation (relative distance between robot i and j)
        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * hidden + edge_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )

        # update MLP sees:   own emb x_i [H] | aggregated messages [H]
        self.upd_mlp = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )

    def forward(self, x, edge_index, edge_attr):
        # x:[N,H]  edge_index:[2,E]  edge_attr:[E,edge_in]  ->  out:[N,H]
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return out

    def message(self, x_i, x_j, edge_attr):
        # Called once per edge. PyG expands node tensors to edge-space:
        #   x_i:[E,H] receiver i,  x_j:[E,H] sender j,  edge_attr:[E,edge_in]
        return self.msg_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))  # [E,H]

    def update(self, aggr_out, x):
        # aggr_out:[N,H] summed messages per node ; x:[N,H] original node embedding
        return self.upd_mlp(torch.cat([x, aggr_out], dim=-1))  # [N,H]


# --------------------------------------------------------------------------- #
# Full model: node encoder -> L message-passing layers -> MLP action head       #
# --------------------------------------------------------------------------- #
class MultiRobotGNN(nn.Module):
    def __init__(self, node_in, edge_in, hidden, action_dim, num_layers=1):
        super().__init__()
        # Single Robot LOCAL Observations
        # node_in = Own state + relative goal position
        # out: observation encoding (hidden)
        self.encoder = nn.Sequential(
            nn.Linear(node_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),
        )

        # GNN Communication: Observation Encoding (hidden) -> GNN encoding
        self.layers = nn.ModuleList([MPLayer(hidden, edge_in) for _ in range(num_layers)])

        # MLP Action Head: GNN encoding -> Action
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        h = self.encoder(x)                       # [N_total, HIDDEN]
        for layer in self.layers:
            h = layer(h, edge_index, edge_attr)   # [N_total, HIDDEN]
        return self.head(h)                       # [N_total, ACTION_DIM]


# --------------------------------------------------------------------------- #
# Frame -> graph conversion, shared by training and rollout                     #
# --------------------------------------------------------------------------- #
def _flat_float32(value: Any, feature_name: str) -> np.ndarray:
    """Accept a LeRobotDataset tensor [D] or a batched rollout tensor [1, D]."""
    array = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    array = np.asarray(array, dtype=np.float32)
    if array.ndim > 1 and array.shape[0] != 1:
        raise ValueError(
            f"Feature '{feature_name}' must be unbatched or batch size 1, got shape {tuple(array.shape)}."
        )
    return array.reshape(-1)


def frame_to_graph(frame: Mapping[str, Any], num_robots: int) -> Data:
    """Convert ONE timestep into a torch_geometric graph: nodes = robots.

    `frame` is anything using the flat multi-robot LeRobot feature layout defined by
    systems/multi_robot.py::get_dataset_features() -- both a LeRobotDataset frame
    (`ds[i]`) and the rollout observation dict built by
    test/evaluate_policy.py::create_policy_input qualify:

      frame["observation.environment_state"]  [num_robots * (2 + 2 * (num_robots - 1))]
          per robot i: [goal_rel_x, goal_rel_y,
                        rel_robot_j_x, rel_robot_j_y for every j != i in index order]
          goal_rel      = goal_i - pos_i
          rel_robot_j   = pos_j - pos_i, zeroed by the simulator when j is out of range
      frame["observation.state"]  [num_robots * proprio_dim]
          per robot i: its own proprioception only (e.g. vx, vy)
      frame["action"]  [num_robots * action_dim]   -- OPTIONAL
          per robot i: the expert (centralized-MPC) action slice; becomes the label

    Every node feature is built from robot i's own local measurements alone, and an
    edge j -> i exists only while j is visible to i, so nothing global leaks in.

    Returns Data(x=[num_robots, node_in], edge_index=[2, E], edge_attr=[E, 2]) with
    y=[num_robots, action_dim] added when "action" is present.
    """
    if num_robots <= 0:
        raise ValueError(f"'num_robots' must be positive, got {num_robots}.")

    missing = [
        key for key in ("observation.environment_state", "observation.state") if key not in frame
    ]
    if missing:
        raise KeyError(f"Frame is missing required features: {missing}.")

    env = _flat_float32(frame["observation.environment_state"], "observation.environment_state")
    proprio = _flat_float32(frame["observation.state"], "observation.state")

    per_robot_env = 2 + 2 * (num_robots - 1)
    if env.shape[0] != num_robots * per_robot_env:
        raise ValueError(
            f"'observation.environment_state' must hold {num_robots * per_robot_env} values for "
            f"{num_robots} robots, got {env.shape[0]}."
        )
    if proprio.shape[0] % num_robots != 0:
        raise ValueError(
            f"'observation.state' length {proprio.shape[0]} is not divisible by num_robots={num_robots}."
        )
    proprio_dim = proprio.shape[0] // num_robots

    node_features: list[np.ndarray] = []
    src: list[int] = []
    tgt: list[int] = []
    attr: list[np.ndarray] = []

    for i in range(num_robots):
        env_block = env[i * per_robot_env:(i + 1) * per_robot_env]
        goal_rel = env_block[0:2]                                   # goal_i - pos_i
        rel_others = env_block[2:].reshape(num_robots - 1, 2)       # pos_j - pos_i per j != i
        own_proprio = proprio[i * proprio_dim:(i + 1) * proprio_dim]

        # node feature = [goal_rel_x, goal_rel_y, own proprioception]
        node_features.append(np.concatenate([goal_rel, own_proprio]))

        # edge j -> i exists only while robot j is visible to robot i, which the
        # simulator encodes by leaving the corresponding rel_robot_j term non-zero.
        other_ids = [j for j in range(num_robots) if j != i]
        for offset, j in zip(rel_others, other_ids):
            if np.all(np.abs(offset) < VISIBILITY_ZERO_EPS):
                continue  # j not visible to i this frame -> no edge
            src.append(j)
            tgt.append(i)
            attr.append(offset)

    x = torch.from_numpy(np.stack(node_features)).to(torch.float32)  # [num_robots, NODE_IN]

    if src:
        edge_index = torch.tensor([src, tgt], dtype=torch.long)            # [2, E]
        edge_attr = torch.from_numpy(np.stack(attr)).to(torch.float32)     # [E, EDGE_IN=2]
    else:
        # no robot pair visible this frame: keep shapes valid for an empty batch.
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 2), dtype=torch.float32)

    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    if "action" in frame:
        action = _flat_float32(frame["action"], "action")
        if action.shape[0] % num_robots != 0:
            raise ValueError(
                f"'action' length {action.shape[0]} is not divisible by num_robots={num_robots}."
            )
        graph.y = torch.from_numpy(action.reshape(num_robots, -1)).to(torch.float32)

    return graph


# --------------------------------------------------------------------------- #
# Rollout wrapper: gives the GNN the same policy API as MLPPolicy/ACT/Diffusion #
# --------------------------------------------------------------------------- #
class GNNPolicy(nn.Module):
    """Adapts MultiRobotGNN to the `select_action` / `reset` policy interface.

    test/evaluate_policy.py hands policies a dict of batched observation tensors and
    expects one flat action row back, so this wrapper converts
    frame dict -> graph -> per-node actions -> joint action [1, num_robots * action_dim].
    Node i is robot i, so flattening in node order matches the simulator's action layout.

    This is a CENTRALIZED convenience wrapper for simulation only: it batches all
    robots into one graph because the simulator steps them jointly. The weights stay
    decentralized -- each node consumes only its own local features plus messages
    from currently visible neighbours.
    """

    def __init__(self, gnn: MultiRobotGNN, num_robots: int):
        super().__init__()
        if num_robots <= 0:
            raise ValueError(f"'num_robots' must be positive, got {num_robots}.")
        self.gnn = gnn
        self.num_robots = int(num_robots)

    def forward(self, data: Data) -> torch.Tensor:
        return self.gnn(data)

    def select_action(self, observation: Mapping[str, torch.Tensor]) -> torch.Tensor:
        if not isinstance(observation, Mapping):
            raise TypeError(
                "GNNPolicy.select_action expects a mapping with keys "
                "'observation.environment_state' and 'observation.state'."
            )

        device = next(self.parameters()).device
        graph = frame_to_graph(observation, num_robots=self.num_robots).to(device)

        per_node_action = self.gnn(graph)      # [num_robots, ACTION_DIM]
        return per_node_action.reshape(1, -1)  # [1, num_robots * ACTION_DIM]

    def reset(self) -> None:
        """Keeps parity with other policy APIs that expose a reset hook."""
        return None


# --------------------------------------------------------------------------- #
# Checkpoint I/O                                                               #
# --------------------------------------------------------------------------- #
def gnn_checkpoint_data(
    model: MultiRobotGNN,
    node_in: int,
    edge_in: int,
    hidden: int,
    action_dim: int,
    num_layers: int,
    num_robots: int,
    **extra: Any,
) -> dict[str, Any]:
    """Bundle weights plus every hyperparameter needed to rebuild the model."""
    checkpoint: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "node_in": int(node_in),
        "edge_in": int(edge_in),
        "hidden": int(hidden),
        "action_dim": int(action_dim),
        "num_layers": int(num_layers),
        "num_robots": int(num_robots),
    }
    checkpoint.update(extra)
    return checkpoint


def load_gnn_policy(
    checkpoint_path: str | Path,
    device: torch.device | str = "cpu",
) -> GNNPolicy:
    """Rebuild a ready-to-roll-out GNNPolicy from a `.pt` written by train_gnn.py."""
    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(
            f"'{checkpoint_path}' is not a GNN checkpoint dict containing 'model_state_dict'."
        )

    required_keys = ("node_in", "edge_in", "hidden", "action_dim", "num_layers", "num_robots")
    missing = [key for key in required_keys if key not in checkpoint]
    if missing:
        raise ValueError(
            f"GNN checkpoint '{checkpoint_path}' is missing architecture keys: {missing}."
        )

    model = MultiRobotGNN(
        node_in=int(checkpoint["node_in"]),
        edge_in=int(checkpoint["edge_in"]),
        hidden=int(checkpoint["hidden"]),
        action_dim=int(checkpoint["action_dim"]),
        num_layers=int(checkpoint["num_layers"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    policy = GNNPolicy(gnn=model, num_robots=8) # num_robots is only used for frame -> graph conversion
    policy.eval()
    policy.to(device)
    return policy
