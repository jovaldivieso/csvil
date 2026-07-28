from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from core.types import VectorSpec, as_vector


def _SE2_matrix(x: float, y: float, theta: float) -> np.ndarray:
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    return np.array(
        [
            [cos_theta, -sin_theta, x],
            [sin_theta, cos_theta, y],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


@dataclass(frozen=True, slots=True)
class Euclidean2DState:
    """Immutable typed view over Euclidean 2D state [x, y]."""

    values: np.ndarray

    @classmethod
    def from_array(cls, state: np.ndarray) -> "Euclidean2DState":
        return cls(values=as_vector(state, VectorSpec(name="state", size=2)))

    @property
    def x(self) -> float:
        return float(self.values[0])

    @property
    def y(self) -> float:
        return float(self.values[1])

    def as_numpy(self) -> np.ndarray:
        return self.values


@dataclass(frozen=True, slots=True)
class Euclidean4DState:
    """Immutable typed view over Euclidean 4D state [x, y, vx, vy]."""

    values: np.ndarray

    @classmethod
    def from_array(cls, state: np.ndarray) -> "Euclidean4DState":
        return cls(values=as_vector(state, VectorSpec(name="state", size=4)))

    @property
    def position(self) -> np.ndarray:
        return self.values[0:2]

    @property
    def velocity(self) -> np.ndarray:
        return self.values[2:4]

    def as_numpy(self) -> np.ndarray:
        return self.values


@dataclass(frozen=True, slots=True)
class SE2PoseState:
    """Immutable typed view over an SE(2) pose represented as a homogeneous transform."""

    values: np.ndarray

    @classmethod
    def from_array(cls, pose: np.ndarray) -> "SE2PoseState":
        pose_vector = as_vector(pose, VectorSpec(name="pose", size=3))
        return cls(values=_SE2_matrix(pose_vector[0], pose_vector[1], pose_vector[2]))

    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> "SE2PoseState":
        matrix = np.asarray(matrix, dtype=float)
        if matrix.shape != (3, 3):
            raise ValueError(f"pose matrix must have shape (3, 3), got {matrix.shape}.")
        return cls(values=matrix)

    @property
    def rotation(self) -> np.ndarray:
        return self.values[0:2, 0:2]

    @property
    def translation(self) -> np.ndarray:
        return self.values[0:2, 2]

    @property
    def theta(self) -> float:
        return float(np.arctan2(self.values[1, 0], self.values[0, 0]))

    def as_matrix(self) -> np.ndarray:
        return self.values

    def as_vector(self) -> np.ndarray:
        return np.array([self.translation[0], self.translation[1], self.theta], dtype=float)


@dataclass(frozen=True, slots=True)
class SE2PoseAndEuclidean2DState:
    """Immutable typed view over [SE(2) pose, Euclidean 2D velocity]."""

    values: np.ndarray

    @classmethod
    def from_array(cls, state: np.ndarray) -> "SE2PoseAndEuclidean2DState":
        return cls(values=as_vector(state, VectorSpec(name="state", size=5)))

    @property
    def pose(self) -> SE2PoseState:
        return SE2PoseState.from_array(self.values[0:3])

    @property
    def euclidean_2d(self) -> Euclidean2DState:
        return Euclidean2DState.from_array(self.values[3:5])

    @property
    def theta(self) -> float:
        return self.pose.theta

    @property
    def v(self) -> float:
        return float(self.values[3])

    @property
    def omega(self) -> float:
        return float(self.values[4])

    def as_numpy(self) -> np.ndarray:
        return self.values


@dataclass(frozen=True, slots=True)
class Euclidean2DAction:
    """Immutable typed view over Euclidean 2D action vectors."""

    values: np.ndarray

    @classmethod
    def from_array(cls, action: np.ndarray) -> "Euclidean2DAction":
        return cls(values=as_vector(action, VectorSpec(name="action", size=2)))

    @property
    def first(self) -> float:
        return float(self.values[0])

    @property
    def second(self) -> float:
        return float(self.values[1])

    def clipped(self, bound: float) -> "Euclidean2DAction":
        return Euclidean2DAction(values=np.clip(self.values, -bound, bound))

    def as_numpy(self) -> np.ndarray:
        return self.values

    def as_torch(self) -> torch.Tensor:
        return torch.from_numpy(self.values).float()


@dataclass(frozen=True, slots=True)
class Euclidean4DObservation:
    """Typed view for observations [goal_rel_x, goal_rel_y, v1, v2]."""

    values: np.ndarray

    @classmethod
    def from_array(cls, obs: np.ndarray) -> "Euclidean4DObservation":
        return cls(values=as_vector(obs, VectorSpec(name="observation", size=4)))

    @property
    def goal_relative(self) -> np.ndarray:
        return self.values[0:2]

    @property
    def velocity_like(self) -> np.ndarray:
        return self.values[2:4]

    def as_numpy(self) -> np.ndarray:
        return self.values


@dataclass(frozen=True, slots=True)
class SE2PoseAndEuclidean2DObservation:
    """Typed view for observations [goal_rel_x, goal_rel_y, rel_theta, v, omega]."""

    values: np.ndarray

    @classmethod
    def from_array(cls, obs: np.ndarray) -> "SE2PoseAndEuclidean2DObservation":
        return cls(values=as_vector(obs, VectorSpec(name="observation", size=5)))

    @property
    def SE2_pose(self) -> SE2PoseState:
        return SE2PoseState.from_array(self.values[0:3])

    @property
    def exteroception(self) -> np.ndarray:
        return self.values[0:3]

    @property
    def rel_theta(self) -> float:
        return float(self.values[2])

    @property
    def euclidean_2d(self) -> np.ndarray:
        return self.values[3:5]

    def as_numpy(self) -> np.ndarray:
        return self.values