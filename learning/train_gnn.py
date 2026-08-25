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

Deployment: each epoch writes outputs/train_gnn/gnn_checkpoint.pt (last) and
outputs/train_gnn/gnn_checkpoint_best.pt (lowest val MSE). Evaluate a checkpoint in
closed loop against the centralized MPC expert with:

  python3 test/evaluate_policy.py \
    --system multi_robot \
    --policy-type gnn \
    --config test/config/multi_double_integrator_casadi_config.yaml \
    --model-dir outputs/train_gnn/gnn_checkpoint_best.pt \
    --seeds "[[42], [21]]"

Val MSE only measures agreement with expert labels on states the expert visited;
that rollout is what shows whether the imitated policy actually reaches goals.
"""

import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from lerobot.datasets.lerobot_dataset import LeRobotDataset

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# MPLayer / MultiRobotGNN / frame_to_graph live in learning/models/gnn.py so that
# test/evaluate_policy.py rebuilds the exact same architecture AND builds graphs from
# live observations exactly the way training built them from dataset frames.
from learning.models.gnn import (
    MultiRobotGNN,
    frame_to_graph,
    gnn_checkpoint_data,
)


# --------------------------------------------------------------------------- #
# 1. Data interface: build ONE Data object per expert-trajectory timestep.     #
#    Dataset-specific: data/lerobot_dataset_multi_robot_casadi_1786695899      #
#    (NUM_ROBOTS = 2, see meta/info.json "features" for the exact layout:      #
#     observation.environment_state [8], observation.state [4], action [4]).   #
# --------------------------------------------------------------------------- #

NUM_ROBOTS = 2


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
            target.append(frame_to_graph(ds[frame_idx], num_robots=NUM_ROBOTS))

    return train_list, val_list


# --------------------------------------------------------------------------- #
# 2. Training / eval loops                                                     #
# --------------------------------------------------------------------------- #
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total, n_nodes = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)                     # [batch_size, ACTION_DIM]
        loss = F.mse_loss(pred, batch.y)        # y is [batch_size, ACTION_DIM]
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
# 3. Wiring it together                                                        #
# --------------------------------------------------------------------------- #
def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # MODEL PARAMETER
    NODE_IN = 4     # local node feature: [goal_rel_x, goal_rel_y, vx, vy]
    EDGE_IN = 2     # edge feature: relative offset to neighbour
    HIDDEN = 64     # internal embedding width
    ACTION_DIM = 2  # expert action: commanded acceleration [ax, ay]
    NUM_LAYERS = 1  # message-passing rounds

    # TRAINING PARAMETER
    batch_size = 32
    max_epochs = 100

    # CHECKPOINTS of the model
    # 'last' is the final epoch; 'best' is the lowest val MSE seen so far, which is
    # the one to deploy -- see the evaluation command in the module docstring.
    CHECKPOINT_DIR = Path("outputs/train_gnn")
    LAST_CHECKPOINT = CHECKPOINT_DIR / "gnn_checkpoint.pt"
    BEST_CHECKPOINT = CHECKPOINT_DIR / "gnn_checkpoint_best.pt"

    # DATASET
    DATASET_ROOT = "data/lerobot_dataset_multi_robot_casadi_1786695899"
    train_set, val_set = build_dataset_from_lerobot(DATASET_ROOT, val_split=0.2, seed=0)

    # DataLoader batches variable-size graphs block-diagonally into one big graph.
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    gnn_encoder = EncoderFactory.create(
        encoder_type="gnn",
        in_features=EDGE_IN,
        hidden=HIDDEN,
    )

    GNNPolicy = MLPPolicy(
        state_dim=NODE_IN,
        action_dim=ACTION_DIM,
        hidden_dims=HIDDEN,
        prediction_horizon=cfg.prediction_horizon,
        neighbor_feature_dim=neighbor_feature_dim,
        neighbor_slots=neighbor_slots,
        neighbor_encoder=neighbor_encoder,
    ).to(device)

    model = GNNPolicy(NODE_IN, EDGE_IN, HIDDEN, ACTION_DIM, num_layers=NUM_LAYERS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(path, epoch, train_mse, val_mse):
        # Store every hyperparameter the evaluator needs to rebuild this architecture,
        # so a checkpoint stays loadable even if the constants above are edited later.
        torch.save(
            gnn_checkpoint_data(
                model=model,
                node_in=NODE_IN,
                edge_in=EDGE_IN,
                hidden=HIDDEN,
                action_dim=ACTION_DIM,
                num_layers=NUM_LAYERS,
                num_robots=NUM_ROBOTS,
                optimizer_state_dict=optimizer.state_dict(),
                epoch=epoch,
                train_mse=train_mse,
                val_mse=val_mse,
                system="multi_robot",
                dataset_root=DATASET_ROOT,
            ),
            path,
        )

    best_val = float("inf")
    for epoch in range(1, max_epochs + 1):
        tr = train_one_epoch(model, train_loader, optimizer, device)
        va = evaluate(model, val_loader, device)
        print(f"epoch {epoch:2d} | train MSE {tr:.4f} | val MSE {va:.4f}")

        save_checkpoint(LAST_CHECKPOINT, epoch=epoch, train_mse=tr, val_mse=va)
        if va < best_val:
            best_val = va
            save_checkpoint(BEST_CHECKPOINT, epoch=epoch, train_mse=tr, val_mse=va)

    print(f"Saved checkpoints: {LAST_CHECKPOINT} and {BEST_CHECKPOINT} (best val MSE {best_val:.4f})")


if __name__ == "__main__":
    main()