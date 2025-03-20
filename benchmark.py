import os
import shutil
import re
import subprocess
import time
import platform
from datetime import datetime
import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt

# Default configuration parameters
DEFAULT_COLORMAPS = ["rainbow", "RdBu", "cividis"]
DEFAULT_OPACITY_STEPS = [1, 3, 5, 7]
DEFAULT_MAX_OPACITY = [1.5]
DEFAULT_MIN_SIZE = [0.0001]
TESTING_COLORMAPS = ["viridis"]
TESTING_OPACITYMAP_OPTIONS = ["linear", "inv_linear", "constant0.01", "constant0.1"]


def run_command(cmd, log_path):
    with open(log_path, "w") as log_file:
        log_file.write(f"Command: {' '.join(cmd)}\n")
        log_file.write(f"Start: {datetime.now().isoformat()}\n\n")

        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )

        while True:
            output = process.stdout.readline()
            if output == "" and process.poll() is not None:
                break
            if output:
                log_file.write(output)
                log_file.flush()

        log_file.write(f"\n\nExit code: {process.poll()}\n")
        log_file.write(f"End: {datetime.now().isoformat()}\n")
        return process.poll()


def get_latest_iteration_ply(model_path):
    pc_dir = os.path.join(model_path, "point_cloud")
    if not os.path.exists(pc_dir):
        return None

    iterations = [d for d in os.listdir(pc_dir) if d.startswith("iteration_")]
    if not iterations:
        return None

    latest_iter = max(iterations, key=lambda x: int(x.split("_")[-1]))
    ply_file = os.path.join(pc_dir, latest_iter, "point_cloud.ply")
    return ply_file if os.path.exists(ply_file) else None


def get_file_size(filepath):
    return os.path.getsize(filepath) if os.path.exists(filepath) else None


def resample_file_to_f32(filepath, dataset_size):
    pattern = r'(float|uint|int)(\d+)'
    match = re.search(pattern, filepath)
    
    if match:
        type_prefix = match.group(1)  # 'float', 'uint', or 'int'
        bit_depth = int(match.group(2))  # 16, 8, 32, etc.
        return (32 / bit_depth) * dataset_size
    else:
        return None


def write_summary(summary_entry, test_type, datetime_string):
    summary_path = os.path.join("output", datetime_string, test_type, "summary.txt")
    with open(summary_path, "a") as f:
        f.write(summary_entry + "\n" + ("-" * 40) + "\n")


def get_system_info():
    info_lines = []
    info_lines.append("System Information")
    info_lines.append("==================")
    info_lines.append(f"Platform: {platform.platform()}")
    info_lines.append(f"Machine: {platform.machine()}")
    info_lines.append(f"Processor: {platform.processor()}")
    info_lines.append(f"Uname: {platform.uname()}\n")

    try:
        lscpu = subprocess.check_output(["lscpu"], text=True)
        info_lines.append("lscpu:\n" + lscpu)
    except Exception:
        info_lines.append("lscpu: Could not retrieve details")

    try:
        gpu_info = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], text=True
        )
        info_lines.append("nvidia-smi:\n" + gpu_info)
    except Exception:
        info_lines.append("nvidia-smi: Not available or no NVIDIA GPU found")

    return "\n".join(info_lines)


def get_original_dataset_filepath(dataset_path):
    if not os.path.exists(dataset_path):
        return None

    for fname in os.listdir(dataset_path):
        if fname.endswith((".raw", ".vtu", ".vtui")):
            return os.path.join(dataset_path, fname)

    return None


def generate_test_configs(args, datasets):
    """
    Generate a list of test configurations based on command-line arguments and provided datasets.
    """
    configs = []

    if args.single_tests:
        for dataset in datasets:
            for cmap in DEFAULT_COLORMAPS:
                config = {
                    "dataset": dataset,
                    "training_colormaps": [cmap],
                    "rendering_colormaps": TESTING_COLORMAPS,
                    "test_type": "single_colormap",
                }
                configs.append(config)

    if args.multi_tests:
        for dataset in datasets:
            for i in range(2, len(DEFAULT_COLORMAPS) + 1):
                config = {
                    "dataset": dataset,
                    "training_colormaps": DEFAULT_COLORMAPS[:i],
                    "rendering_colormaps": TESTING_COLORMAPS,
                    "test_type": "multiple_colormaps",
                }
                configs.append(config)

    if args.opacity_tests:
        for dataset in datasets:
            for step in DEFAULT_OPACITY_STEPS:
                config = {
                    "dataset": dataset,
                    "training_colormaps": ["rainbow"],
                    "rendering_colormaps": TESTING_COLORMAPS,
                    "opacity_steps": step,
                    "test_type": "opacity_steps",
                }
                configs.append(config)

    if args.max_opacity_tests:
        for dataset in datasets:
            for max_op in DEFAULT_MAX_OPACITY:
                config = {
                    "dataset": dataset,
                    "training_colormaps": ["rainbow"],
                    "rendering_colormaps": TESTING_COLORMAPS,
                    "max_opacity": max_op,
                    "test_type": "max_opacity",
                }
                configs.append(config)

    if args.min_size_tests:
        for dataset in datasets:
            for size in DEFAULT_MIN_SIZE:
                config = {
                    "dataset": dataset,
                    "training_colormaps": ["rainbow"],
                    "rendering_colormaps": TESTING_COLORMAPS,
                    "min_size": size,
                    "test_type": "min_gaussian_size",
                }
                configs.append(config)

    if args.combined_tests:
        for dataset in datasets:
            for opacity in DEFAULT_OPACITY_STEPS:
                for max_op in DEFAULT_MAX_OPACITY:
                    for size in DEFAULT_MIN_SIZE:
                        config = {
                            "dataset": dataset,
                            "training_colormaps": ["rainbow"],
                            "rendering_colormaps": TESTING_COLORMAPS,
                            "opacity_steps": opacity,
                            "max_opacity": max_op,
                            "min_size": size,
                            "test_type": "combined_grid",
                        }
                        configs.append(config)

    return configs


def benchmark(args, datasets):
    """
    Run benchmark tests based on generated configurations, logging training, rendering, and metric calculation.
    """
    test_types = set()

    if args.single_tests:
        test_types.add("single_colormap")
    if args.multi_tests:
        test_types.add("multiple_colormaps")
    if args.opacity_tests:
        test_types.add("opacity_steps")
    if args.max_opacity_tests:
        test_types.add("max_opacity")
    if args.min_size_tests:
        test_types.add("min_gaussian_size")
    if args.combined_tests:
        test_types.add("combined_grid")

    if not test_types:
        print(
            "No tests selected. Please specify at least one test flag (e.g. --single-tests)."
        )
        return

    sys_info = get_system_info()
    now = datetime.now()
    datetime_string = now.strftime("%Y-%m-%d_%H-%M-%S")

    for test_type in test_types:
        test_type_dir = os.path.join("output", datetime_string, test_type)
        if os.path.exists(test_type_dir):
            shutil.rmtree(test_type_dir)
        os.makedirs(test_type_dir)
        with open(os.path.join(test_type_dir, "system_info.txt"), "w") as f:
            f.write(sys_info)

    test_configurations = generate_test_configs(args, datasets)
    total_tests = len(test_configurations)

    if total_tests == 0:
        print(
            "No test configurations generated. Check your test flags and dataset inputs."
        )
        return

    print(f"Total test configurations: {total_tests}")

    for idx, config in enumerate(test_configurations, start=1):
        dataset_name = os.path.basename(os.path.normpath(config["dataset"]))
        folder_parts = [dataset_name, "_".join(config["training_colormaps"])]

        if "opacity_steps" in config:
            folder_parts.append(f"opacity{config['opacity_steps']}")
        if "max_opacity" in config:
            folder_parts.append(f"maxOpac{config['max_opacity']}")
        if "min_size" in config:
            folder_parts.append(f"minSize{config['min_size']}")

        main_folder = "_".join(folder_parts)
        test_type = config.get("test_type", "unknown")
        model_path = os.path.join("output", datetime_string, test_type, main_folder)
        os.makedirs(model_path, exist_ok=True)

        print("\n" + "=" * 40)
        print(f"Starting benchmark {idx}/{total_tests}: {test_type}/{main_folder}")
        print("=" * 40)

        # Run training
        train_cmd = [
            "python",
            "train.py",
            "-s",
            config["dataset"],
            "--eval",
            "--colormaps",
            ",".join(config["training_colormaps"]),
            "--model_path",
            model_path,
            "--opacity_steps",
            str(config.get("opacity_steps", 5)),
            "--max_opac_grad",
            str(config.get("max_opacity", 1.5)),
            "--min_gaussian_size",
            str(config.get("min_size", 0.0001)),
        ]
        train_log = os.path.join(model_path, "train.log")
        print("Training started...")
        start_time = time.time()
        exit_code = run_command(train_cmd, train_log)
        train_duration = time.time() - start_time

        if exit_code != 0:
            print(f"Training failed for {main_folder}. Skipping this configuration.")
            continue

        # Run rendering
        render_cmd = [
            "python",
            "render.py",
            "--model_path",
            model_path,
            "--colormaps",
            ",".join(config["rendering_colormaps"]),
            "--opacity_steps",
            str(DEFAULT_OPACITY_STEPS[-1]),
            "--opacitymap_options",
            ",".join(TESTING_OPACITYMAP_OPTIONS)
        ]
        render_log = os.path.join(model_path, "render.log")
        print("Rendering started...")
        start_time = time.time()
        exit_code = run_command(render_cmd, render_log)
        render_duration = time.time() - start_time

        if exit_code != 0:
            print(f"Rendering failed for {main_folder}. Skipping metrics.")
            continue

        # Run metrics calculation
        metrics_cmd = ["python", "metrics.py", "--model_path", model_path]
        metrics_log = os.path.join(model_path, "metrics.log")
        print("Metrics calculation started...")
        start_time = time.time()
        exit_code = run_command(metrics_cmd, metrics_log)
        metrics_duration = time.time() - start_time

        # Calculate compression info if available
        original_file = get_original_dataset_filepath(config["dataset"])
        dataset_size = get_file_size(original_file) if original_file else None
        dataset_size = resample_file_to_f32(original_file, dataset_size)
        ply_file = get_latest_iteration_ply(model_path)
        ply_size = get_file_size(ply_file) if ply_file else None

        if dataset_size and ply_size:
            compression_ratio = dataset_size / ply_size
            compression_info = (
                f"Original Size: {dataset_size} bytes\n"
                f"Compressed Size: {ply_size} bytes\n"
                f"Compression Ratio: {compression_ratio:.2f}\n"
            )
        else:
            compression_info = "Compression info not available.\n"

        # Timing information for this configuration
        timing_info = (
            f"Test Type: {config.get('test_type', 'unknown')}\n"
            f"Training duration: {train_duration:.2f} seconds\n"
            f"Rendering duration: {render_duration:.2f} seconds\n"
            f"Metrics duration: {metrics_duration:.2f} seconds\n"
            f"{compression_info}"
        )

        with open(os.path.join(model_path, "timing.txt"), "w") as f:
            f.write(timing_info)

        # Write a summary entry
        summary_entry = (
            f"Test {idx}/{total_tests}: {test_type}/{main_folder}\n"
            f"Dataset: {config['dataset']}\n"
            f"Training colormaps: {config['training_colormaps']}\n"
            f"Rendering colormaps: {config['rendering_colormaps']}\n"
        )
        if "opacity_steps" in config:
            summary_entry += f"Opacity steps: {config['opacity_steps']}\n"
        if "max_opacity" in config:
            summary_entry += f"Max opacity gradient: {config['max_opacity']}\n"
        if "min_size" in config:
            summary_entry += f"Min Gaussian size: {config['min_size']}\n"
        summary_entry += timing_info

        write_summary(summary_entry, test_type, datetime_string)
        print(f"Completed benchmark {idx}/{total_tests} for {test_type}/{main_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run benchmark tests and generate plots."
    )
    parser.add_argument(
        "--datasets",
        type=str,
        required=True,
        help="Comma separated list of dataset paths (e.g. 'data/skull,data/bluntfin')",
    )
    parser.add_argument(
        "--single-tests", action="store_true", help="Run single colormap tests"
    )
    parser.add_argument(
        "--multi-tests", action="store_true", help="Run multiple colormap tests"
    )
    parser.add_argument(
        "--opacity-tests", action="store_true", help="Run opacity step tests"
    )
    parser.add_argument(
        "--max-opacity-tests",
        action="store_true",
        help="Run max opacity gradient tests",
    )
    parser.add_argument(
        "--min-size-tests", action="store_true", help="Run min Gaussian size tests"
    )
    parser.add_argument(
        "--combined-tests",
        action="store_true",
        help="Run combined grid tests over opacity, max opacity, and min size",
    )

    args = parser.parse_args()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    benchmark(args, datasets)

    print("\nAll benchmarks and plotting completed!")
