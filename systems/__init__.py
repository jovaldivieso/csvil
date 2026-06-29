from gymnasium.envs.registration import register
from .single_integrator import SingleIntegrator
from .double_integrator import DoubleIntegrator
from .gym_wrapper import DynamicsGymWrapper


def make_env(simulator_class, config=None):
    """
    Factory function that instantiates the correct physics simulator
    and automatically wraps it in our universal Gym Wrapper.
    """
    # Default config if LeRobot doesn't provide one
    if config is None:
        config = {
            "dt": 0.05,
            "max_accel": 2.0,
            "max_speed": 1.0,
            "goal": [1.0, 1.0]
        }

    simulator = simulator_class(config)
    return DynamicsGymWrapper(simulator)


register(
    id='SingleIntegrator-v0',
    entry_point=f'{__name__}:make_env',
    kwargs={'simulator_class': SingleIntegrator}
)


register(
    id='DoubleIntegrator-v0',
    entry_point=f'{__name__}:make_env',
    kwargs={'simulator_class': DoubleIntegrator}
)
