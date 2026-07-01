import os
import argparse
import draccus

from lerobot.scripts.lerobot_train import train
from lerobot.configs.train import TrainPipelineConfig


def main():
    parser = argparse.ArgumentParser(
        description="Train LeRobot Diffusion Policy")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the LeRobot YAML config file")
    args = parser.parse_args()

    lerobot_training_config_path = os.path.abspath(args.config)
    print(f"Loading LeRobot config from: {lerobot_training_config_path}")

    # Explicitly load the YAML into the LeRobot config object using Draccus
    train_pipeline_config = draccus.parse(
        config_class=TrainPipelineConfig,
        config_path=lerobot_training_config_path
    )

    train(train_pipeline_config)


if __name__ == "__main__":
    main()
