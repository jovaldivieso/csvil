import os
import argparse
import draccus

from typing import Optional

from lerobot.scripts.lerobot_train import train
from lerobot.configs.train import TrainPipelineConfig


def run_training(config_path: str) -> None:
    lerobot_training_config_path = os.path.abspath(config_path)
    print(f"Loading LeRobot config from: {lerobot_training_config_path}")

    # Explicitly load the YAML into the LeRobot config object using Draccus
    train_pipeline_config = draccus.parse(
        config_class=TrainPipelineConfig,
        config_path=lerobot_training_config_path
    )

    train(train_pipeline_config)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train LeRobot Diffusion Policy")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the LeRobot YAML config file")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    run_training(config_path=args.config)


if __name__ == "__main__":
    main()
