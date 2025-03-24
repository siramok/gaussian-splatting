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
TESTING_COLORMAPS = ["rainbow"]
TESTING_OPACITYMAP_OPTIONS = ["linear", "inv_linear", "constant0.01", "constant0.1"]
TESTING_OPACITYSTEPS = 0
TESTING_OPACITYMAP_RANDOMS = 0


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


def write_summary(summary_entry, folder, datetime_string):
    summary_path = os.path.join(folder, datetime_string, "summary.txt")
    if not os.path.exists(os.path.join(folder, datetime_string)):
        os.makedirs(os.path.join(folder, datetime_string))

    with open(summary_path, "a") as f:
        f.write(summary_entry + "\n" + ("-" * 40) + "\n")

def benchmark(args):
    """
    Run benchmark metrics with logging.
    """
    now = datetime.now()
    datetime_string = now.strftime("%Y-%m-%d_%H-%M-%S")

    tests = [item for item in os.listdir(args.folder) 
               if os.path.isdir(os.path.join(args.folder, item))]
    total_tests = len(tests)

    print(f"Total test configurations: {total_tests}")

    for idx, test in enumerate(tests, start=1):

        print("\n" + "=" * 40)
        print(f"Starting benchmark {idx}/{total_tests}: {test}")
        print("=" * 40)

        model_path = os.path.join(args.folder, test)
        print(model_path)

        # Run rendering
        render_cmd = [
            "python",
            "render.py",
            "--model_path",
            model_path,
            "--skip_train",
            "--eval",
            "--colormaps",
            ",".join(TESTING_COLORMAPS),
            "--opacity_steps",
            str(TESTING_OPACITYSTEPS),
            "--opacitymap_options",
            ",".join(TESTING_OPACITYMAP_OPTIONS),
            "--opacitymap_randoms",
            str(TESTING_OPACITYMAP_RANDOMS)
        ]
        render_log = os.path.join(model_path, "render.log")
        print("Rendering started...")
        start_time = time.time()
        exit_code = run_command(render_cmd, render_log)
        render_duration = time.time() - start_time

        if exit_code != 0:
            print(f"Rendering failed for {model_path}. Skipping metrics.")
            continue

        # Run metrics calculation
        metrics_cmd = ["python", "metrics.py", "--model_path", model_path]
        metrics_log = os.path.join(model_path, "metrics.log")
        print("Metrics calculation started...")
        start_time = time.time()
        exit_code = run_command(metrics_cmd, metrics_log)
        metrics_duration = time.time() - start_time

        # Timing information for this configuration
        timing_info = (
            f"Rendering duration: {render_duration:.2f} seconds\n"
            f"Metrics duration: {metrics_duration:.2f} seconds\n"
        )

        # Write a summary entry
        summary_entry = (
            f"Test {idx}/{total_tests}: {test}\n"
            f"Testing colormaps: {TESTING_COLORMAPS}\n"
            f"Testing opacitymap_options: {TESTING_OPACITYMAP_OPTIONS}\n"
            f"Testing opacitymap_randoms: {TESTING_OPACITYMAP_RANDOMS}\n"
            f"Testing opacitymap_steps: {TESTING_OPACITYSTEPS}\n"
        )
        summary_entry += timing_info
        if os.path.exists(metrics_log):
            try:
                with open(metrics_log, 'r') as log_file:
                    metrics_content = log_file.read()
                    summary_entry += "\nMetrics Log:\n"
                    summary_entry += metrics_content
            except Exception as e:
                summary_entry += f"\nFailed to read metrics log: {str(e)}\n"
        else:
            summary_entry += "\nMetrics log file not found.\n"

        write_summary(summary_entry, args.folder, datetime_string)
        print(f"Completed benchmark {idx}/{total_tests} for {test}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run benchmark tests and generate plots."
    )
    parser.add_argument(
        "--folder",
        type=str,
        required=True,
        help="Folder to run metrics for",
    )

    args = parser.parse_args()
    benchmark(args)