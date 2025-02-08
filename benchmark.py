import os
import subprocess
import time
import platform
from datetime import datetime

# Colormaps reserved for training
TRAINING_COLORMAPS = ["viridis", "plasma", "RdBu", "cividis", "turbo"]

# Colormaps reserved for testing, we want them to be different than colormaps used for training
TESTING_COLORMAPS = ["magma", "coolwarm", "twilight", "tab10", "inferno"]


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


def write_summary(summary_entry):
    summary_path = os.path.join("output", "summary.txt")
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

    # Get lscpu info
    try:
        lscpu = subprocess.check_output(["lscpu"], text=True)
        info_lines.append("lscpu:\n" + lscpu)
    except Exception:
        info_lines.append("lscpu: Could not retrieve details")

    # Optionally get GPU info if available
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
        if fname.endswith(".raw") or fname.endswith(".vtu") or fname.endswith(".vtui"):
            return os.path.join(dataset_path, fname)
    return None


def generate_test_configs():
    available_datasets = [
        "data/skull",
        # "data/bluntfin",
        # "data/buckyball",
        # "data/bonsai",
    ]
    configs = []

    # Single-colormap tests
    for dataset in available_datasets:
        for cmap in TRAINING_COLORMAPS:
            config = {
                "dataset": dataset,
                "training_colormaps": [cmap],
                "rendering_colormaps": TESTING_COLORMAPS,
                "test_type": "single",
            }
            configs.append(config)

    # Incremental Multi-Colormap Tests
    for dataset in available_datasets:
        for i in range(2, len(TRAINING_COLORMAPS) + 1):
            config = {
                "dataset": dataset,
                "training_colormaps": TRAINING_COLORMAPS[:i],
                "rendering_colormaps": TESTING_COLORMAPS,
                "test_type": "multiple",
            }
            configs.append(config)

    return configs


def benchmark():
    # Create the parent output directory if it doesn't exist.
    os.makedirs("output", exist_ok=True)

    # Write system info to file.
    sys_info = get_system_info()
    with open(os.path.join("output", "system_info.txt"), "w") as f:
        f.write(sys_info)

    test_configurations = generate_test_configs()
    total_tests = len(test_configurations)

    print(f"Total test configurations: {total_tests}")

    for idx, config in enumerate(test_configurations, start=1):
        dataset_name = os.path.basename(config["dataset"])
        training_colors = "_".join(config["training_colormaps"])
        parent = config.get("test_type", "unknown")
        main_folder = f"{dataset_name}_{training_colors}"
        model_path = os.path.join("output", parent, main_folder)
        os.makedirs(model_path, exist_ok=True)

        print(f"\n{'=' * 40}")
        print(f"Starting benchmark {idx}/{total_tests}: {parent}/{main_folder}")
        print(f"{'=' * 40}")

        # Run train.py
        train_cmd = [
            "python",
            "train.py",
            "-s",
            config["dataset"],
            "--colormaps",
            ",".join(config["training_colormaps"]),
            "--model_path",
            model_path,
        ]
        train_log = os.path.join(model_path, "train.log")
        print("Training started...")
        start_time = time.time()
        exit_code = run_command(train_cmd, train_log)
        train_duration = time.time() - start_time

        if exit_code != 0:
            print(f"Training failed for {main_folder}. Skipping.")
            continue

        # Run render.py
        render_cmd = ["python", "render.py", "--model_path", model_path]
        render_cmd += ["--colormaps", ",".join(config["rendering_colormaps"])]
        render_log = os.path.join(model_path, "render.log")
        print("Rendering started...")
        start_time = time.time()
        exit_code = run_command(render_cmd, render_log)
        render_duration = time.time() - start_time

        if exit_code != 0:
            print(f"Rendering failed for {main_folder}. Skipping metrics.")
            continue

        # Run metrics.py
        metrics_cmd = ["python", "metrics.py", "--model_path", model_path]
        metrics_log = os.path.join(model_path, "metrics.log")
        print("Metrics calculation started...")
        start_time = time.time()
        exit_code = run_command(metrics_cmd, metrics_log)
        metrics_duration = time.time() - start_time

        # Compute compression ratio
        original_file = get_original_dataset_filepath(config["dataset"])
        dataset_size = get_file_size(original_file) if original_file else None
        ply_file = get_latest_iteration_ply(model_path)
        ply_size = get_file_size(ply_file) if ply_file else None

        compression_info = ""
        if dataset_size and ply_size:
            compression_ratio = (1 - (ply_size / dataset_size)) * 100
            compression_info = (
                f"Original Size: {dataset_size} bytes\n"
                f"Compressed Size: {ply_size} bytes\n"
                f"Compression Ratio: {compression_ratio:.2f}%\n"
            )
        else:
            compression_info = "Compression info not available.\n"

        # Create a timing summary for this configuration
        timing_info = (
            f"Test Type: {config.get('test_type', 'unknown')}\n"
            f"Training duration: {train_duration:.2f} seconds\n"
            f"Rendering duration: {render_duration:.2f} seconds\n"
            f"Metrics duration: {metrics_duration:.2f} seconds\n"
            f"{compression_info}"
        )

        with open(os.path.join(model_path, "timing.txt"), "w") as f:
            f.write(timing_info)

        # Append a summary entry for this test.
        summary_entry = (
            f"Test {idx}/{total_tests}: {parent}/{main_folder}\n"
            f"Test Type: {config.get('test_type', 'unknown')}\n"
            f"Dataset: {config['dataset']}\n"
            f"Training colormaps: {config['training_colormaps']}\n"
            f"Rendering colormaps: {config['rendering_colormaps']}\n"
            f"{timing_info}"
        )
        write_summary(summary_entry)

        print(f"Completed benchmark {idx}/{total_tests} for {parent}/{main_folder}")


if __name__ == "__main__":
    benchmark()
    print("\nAll benchmarks completed!")
