"""
Decentralized multi-robot controller via imitation learning.

Where the data comes from vs. what the network sees (read this first):
  - A CENTRALIZED CasADi MPC planner solved for BOTH robots jointly (it saw the
    full global state: both positions, both velocities, both goals) and produced
    a joint expert action per timestep. This is the GLOBAL data / the label source.
  - The dataset does NOT store that global state directly. Each robot's recorded
    observation is already LOCAL/egocentric: its own offset-to-goal, its own
    velocity, and its offset to the other robot -- but that last term is zeroed
    out whenever the other robot is farther away than `inter_robot_visibility_radius`
    (see systems/multi_robot.py::observe). So the recorded features are exactly
    what a robot with local relative-position sensing + a limited comms/sensing
    radius could measure on its own.
  - We turn each timestep into one graph: nodes = robots, using ONLY their own
    local features; edges = "robot j is currently visible to robot i", using
    ONLY the relative offset as the edge feature. The label per node is that
    robot's slice of the joint expert action.
  - Because the trained MultiRobotGNN never consumes a global/absolute state and
    never assumes a fixed number or identity of neighbours (message passing
    aggregates whatever edges exist), the SAME trained weights can run
    independently on every physical robot at test time: each one senses its own
    local observation + relative offsets to whoever is currently in range, and
    reproduces -- without any central solver at runtime -- the behaviour the
    centralized MPC expert would have computed. That's the imitation-learning
    step that turns a global/centralized expert into a decentralized policy.
  - Each timestep of each trajectory becomes ONE graph snapshot.
  - Message-passing GNN aggregates a VARIABLE number of neighbours per node
    (0..many) into a fixed-size embedding -> MLP head -> per-robot action.
  - Task = behavioural cloning: regress the expert action per robot (MSE).

Dimensions used throughout (all configurable near the bottom):
  D          = 2      spatial dimension (x, y)
  NODE_IN    = 4       node feature  = [goal_rel_x, goal_rel_y, vx, vy]      -> [N, 4]
  EDGE_IN    = D       edge feature  = relative neighbour position           -> [E, 2]
  HIDDEN     = 64      internal embedding width
  ACTION_DIM = D       expert action = commanded acceleration [ax, ay]       -> [N, 2]

Key shapes (N = robots in a snapshot, E = directed edges, B = graphs in a batch):
  data.x          [N, NODE_IN]     per-node features
  data.edge_index [2, E]           row 0 = source j, row 1 = target i  (message j -> i)
  data.edge_attr  [E, EDGE_IN]     per-edge features
  data.y          [N, ACTION_DIM]  per-node expert action (the label)
  After batching, the loader stacks graphs block-diagonally, so a batch behaves
  like one big graph with N_total = sum of N over the B graphs. Nothing needs to
  be padded and graphs may have different N / E.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing
from lerobot.datasets.lerobot_dataset import LeRobotDataset


# --------------------------------------------------------------------------- #
# 1. One message-passing layer (uses edge features; sum-aggregates neighbours) #
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
# 2. Full model: node encoder -> L message-passing layers -> MLP action head   #
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
# 3. Data interface: build ONE Data object per expert-trajectory timestep.     #
#    Dataset-specific: data/lerobot_dataset_multi_robot_casadi_1786695899      #
#    (NUM_ROBOTS = 2, see meta/info.json "features" for the exact layout).     #
# --------------------------------------------------------------------------- #

NUM_ROBOTS = 2
# Zeroed by systems/multi_robot.py::observe() whenever the true offset exceeds
# inter_robot_visibility_radius; a real offset is astronomically unlikely to
# land exactly on 0.0, so "all-zero" is a safe stand-in for "not visible".
VISIBILITY_ZERO_EPS = 1e-9


def frame_to_graph(frame):
    """Convert ONE LeRobotDataset frame (a single dict of tensors, e.g. ds[i])
    into a torch_geometric graph for the 2-robot casadi dataset.

    frame["observation.environment_state"]  float32 [8]
        per meta/info.json names, laid out per robot as [goal_rel_x, goal_rel_y,
        rel_robot_<other>_x, rel_robot_<other>_y]:
          [0:2] robot_0.goal_rel        (goal_0 - pos_0)
          [2:4] robot_0.rel_robot_1     (pos_1  - pos_0), zeroed if out of range
          [4:6] robot_1.goal_rel        (goal_1 - pos_1)
          [6:8] robot_1.rel_robot_0     (pos_0  - pos_1), zeroed if out of range
    frame["observation.state"]  float32 [4]  -- proprioception, own velocity only
          [0:2] robot_0.vx, vy
          [2:4] robot_1.vx, vy
    frame["action"]  float32 [4]  -- expert (centralized-MPC) acceleration
          [0:2] robot_0.ax, ay
          [2:4] robot_1.ax, ay
    """
    env = frame["observation.environment_state"].numpy().astype("float32")  # [8]
    state = frame["observation.state"].numpy().astype("float32")            # [4]
    act = frame["action"].numpy().astype("float32")                         # [4]

    # concatenate per-robot node features into a single graph snapshot. Each robot is a node.
    goal_rel = [env[0:2], env[4:6]]      # per-robot [2] own offset-to-goal
    vel = [state[0:2], state[2:4]]       # per-robot [2] own velocity

    # concatenate per-robot edge features (zero if edge not exists)
    rel_other = [env[2:4], env[6:8]]     # per-robot [2] offset-to-other-robot (0,0 if not visible)

    # combine into a single node feature vector per robot
    # TODO: could do this directly a few lines above? but this is clearer and more explicit for now
    x = torch.tensor(
        [
            [*goal_rel[i], *vel[i]]
            for i in range(NUM_ROBOTS)
        ],
        dtype=torch.float32,
    )


    # create edge_index and edge_attr for the directed graph.
    # edge only exists if robot j is visible to robot i
    # currently in rel_other[i], which is zeroed out if j is too far away from i.
    src, tgt, attr = [], [], []
    for i in range(NUM_ROBOTS):
        for j in range(NUM_ROBOTS):
            if i == j:
                continue
            offset = rel_other[i]  # pos_j - pos_i, as seen by robot i
            if abs(offset[0]) < VISIBILITY_ZERO_EPS and abs(offset[1]) < VISIBILITY_ZERO_EPS:
                continue  # j not visible to i this frame -> no edge
            src.append(j)
            tgt.append(i)
            attr.append(offset)

    if src:
        edge_index = torch.tensor([src, tgt], dtype=torch.long)             # [2, E]
        edge_attr = torch.from_numpy(np.stack(attr)).to(torch.float32)      # [E, EDGE_IN=2]
    else:
        # no robot pair visible this frame: keep shapes valid for an empty batch.
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 2), dtype=torch.float32)

    # label [NUM_ROBOTS, ACTION_DIM=2] = expert acceleration per robot
    y = torch.tensor(act.reshape(NUM_ROBOTS, 2), dtype=torch.float32)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


def build_dataset_from_lerobot(root, val_split=0.2, seed=0):
    """Load the LeRobot dataset at `root` and turn every frame into a graph.

    Splits by EPISODE (not by frame) so consecutive timesteps of the same
    trajectory never leak across train/val. Returns (train_list, val_list) of
    torch_geometric Data objects, ready for DataLoader.
    """
    ds = LeRobotDataset(repo_id="local", root=root)

    episodes = ds.meta.episodes     # HF Dataset: one row per episode, with the
    num_episodes = len(episodes)    # frame-index range [dataset_from_index, dataset_to_index)

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_episodes, generator=generator).tolist()
    n_val = max(1, int(round(num_episodes * val_split)))
    val_episode_ids = set(perm[:n_val])

    train_list, val_list = [], []
    for ep in episodes:
        frame_range = range(ep["dataset_from_index"], ep["dataset_to_index"])
        target = val_list if ep["episode_index"] in val_episode_ids else train_list
        for frame_idx in frame_range:
            target.append(frame_to_graph(ds[frame_idx]))

    return train_list, val_list


# --------------------------------------------------------------------------- #
# 4. Training / eval loops                                                     #
# --------------------------------------------------------------------------- #
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total, n_nodes = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)                     # [N_total, ACTION_DIM]
        loss = F.mse_loss(pred, batch.y)        # per-node regression; y is [N_total, ACTION_DIM]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * batch.num_nodes  # weight by nodes so epoch avg is per-robot
        n_nodes += batch.num_nodes
    return total / n_nodes


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total, n_nodes = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)
        total += F.mse_loss(pred, batch.y).item() * batch.num_nodes
        n_nodes += batch.num_nodes
    return total / n_nodes


# --------------------------------------------------------------------------- #
# 5. Wiring it together                                                        #
# --------------------------------------------------------------------------- #
def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    NODE_IN = 4     # local node feature: [goal_rel_x, goal_rel_y, vx, vy]
    EDGE_IN = 2     # edge feature: relative offset to neighbour
    HIDDEN = 64     # internal embedding width
    ACTION_DIM = 2  # expert action: commanded acceleration [ax, ay]

    DATASET_ROOT = "data/lerobot_dataset_multi_robot_casadi_1786695899"

    train_set, val_set = build_dataset_from_lerobot(DATASET_ROOT, val_split=0.2, seed=0)

    # DataLoader batches variable-size graphs block-diagonally into one big graph.
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32)

    model = MultiRobotGNN(NODE_IN, EDGE_IN, HIDDEN, ACTION_DIM, num_layers=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, 16):
        tr = train_one_epoch(model, train_loader, optimizer, device)
        va = evaluate(model, val_loader, device)
        print(f"epoch {epoch:2d} | train MSE {tr:.4f} | val MSE {va:.4f}")


if __name__ == "__main__":
    main()