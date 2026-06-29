import sys
import os
import argparse
import draccus

from lerobot.scripts.lerobot_train import train
from lerobot.configs.train import TrainPipelineConfig

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Register the gym environment
import systems


def main():
    parser = argparse.ArgumentParser(description="Train LeRobot Diffusion \
    Policy")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the LeRobot YAML config file")
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    print(f"Loading LeRobot config from: {config_path}")

    # Explicitly load the YAML into the LeRobot config object using Draccus
    cfg = draccus.parse(
        config_class=TrainPipelineConfig,
        config_path=config_path
    )

    print("Handing over execution to LeRobot...")
    train(cfg)


if __name__ == "__main__":
    main()
