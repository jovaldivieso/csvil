import os
import argparse
import subprocess

PATH_TO_EXE = "/home/mambauser/dynoplan/build/main_primitives"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="dynobench model yaml"
    )
    parser.add_argument(
        "--num-primitives",
        "-n",
        type=int,
        default=2000,
        help="number of motion primitives to generate (default: 2000)"
    )
    args = parser.parse_args()

    path_to_config = os.path.join(
        "/workspace",
        "planning",
        "models",
        args.config
    )
    if not os.path.isfile(path_to_config):
        raise FileNotFoundError(path_to_config)

    # uses config filename without .yaml as dynamics name:
    dynamics = args.config.split(".")[0]

    path_to_output = os.path.join("data", "motion_primitives")
    os.makedirs(path_to_output, exist_ok=True)
    path_to_output = os.path.join(path_to_output, f"{dynamics}.bin")
    
    # uses config directory as dynoplan's model directory:
    path_to_models = os.path.dirname(path_to_config) + os.sep

    # Generates motion primitives.
    subprocess.run(
        [
            PATH_TO_EXE,
            "--mode_gen_id", "0",
            "--dynamics", dynamics,
            "--models_base_path", path_to_models,
            "--max_num_primitives", str(args.num_primitives),
            "--out_file", path_to_output,
        ],
        check=True,
        stdout=subprocess.DEVNULL,  
        stderr=subprocess.DEVNULL, 
    )

    # Improves generated primitives.
    improved = f"{path_to_output}.im.bin"
    subprocess.run(
        [
            PATH_TO_EXE,
            "--mode_gen_id", "1",
            "--dynamics", dynamics,
            "--models_base_path", path_to_models,
            "--max_num_primitives", str(args.num_primitives),
            "--in_file", path_to_output,
            "--solver_id", "1",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Splits long primitives.
    final = f"{improved}.sp.bin"
    subprocess.run(
        [
            PATH_TO_EXE,
            "--mode_gen_id", "2",
            "--dynamics", dynamics,
            "--models_base_path", path_to_models,
            "--max_num_primitives", "-1",
            "--max_splits", "1",
            "--min_length_cut", "5",
            "--max_length_cut", "50",
            "--in_file", improved,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    print(f"motion primitives written to {final}")


if __name__ == "__main__":
    main()