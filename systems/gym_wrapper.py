import gymnasium as gym
from gymnasium import spaces
import numpy as np


class DynamicsGymWrapper(gym.Env):
    """
    A universal gymnasium wrapper for any DynamicsSimulator subclass.
    Translates the pure physics engine into the standard gymnasium API.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, simulator, max_steps=500):
        super().__init__()
        self.sim = simulator
        self.max_steps = max_steps
        self.current_step = 0

        # Dynamically read the action and observation bounds from the simulator
        self.action_space = spaces.Box(
            low=-self.sim.max_action,
            high=self.sim.max_action,
            shape=(self.sim.nu,),
            dtype=np.float32,
        )

        # Assuming observation is the same size as the state (nx)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.sim.nx,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        """Gymnasium requires reset to handle seeds and return (obs, info)."""
        super().reset(seed=seed)
        self.current_step = 0

        rng = np.random.default_rng(seed)
        initial_state = self.sim.random_initial_state(rng)
        self.sim.reset(initial_state)

        obs = self.sim.observe(self.sim.state).astype(np.float32)
        info = {}

        return obs, info

    def step(self, action):
        """
        Translate the physics step into (obs, reward, terminated, truncated,
        info).
        """
        self.current_step += 1

        # Step the physics simulator
        self.sim.state = self.sim.step(self.sim.state, action)

        # Get the new observation
        obs = self.sim.observe(self.sim.state).astype(np.float32)

        # Check if the goal is reached
        terminated = bool(self.sim.is_done(self.sim.state))

        # Check if we ran out of time
        truncated = bool(self.current_step >= self.max_steps)

        # Calculate a simple sparse reward (1 if goal reached, 0 otherwise)
        # Imitation learning doesn't use this for training, but gym requires it
        reward = 1.0 if terminated else 0.0

        info = {"is_success": terminated}

        return obs, reward, terminated, truncated, info
