import os
import json
import re
import matplotlib.pyplot as plt
import numpy as np


def get_dynamic_width(labels, min_width=10, factor=0.2):
    if not labels:
        return min_width
    max_label_length = max(len(label) for label in labels)
    return max(min_width, max_label_length * factor)


def parse_summary(summary_path):
    experiments_summary = {}
    with open(summary_path, "r") as f:
        content = f.read()

    blocks = content.split("----------------------------------------")

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        lines = block.splitlines()
        header = lines[0]
        parts = header.split(":")
        if len(parts) < 2:
            continue

        experiment_full = parts[1].strip()
        header_parts = experiment_full.split("/")
        if len(header_parts) < 2:
            continue
        experiment_label = header_parts[1].strip()

        summary_data = {}
        for line in lines:
            line = line.strip()
            if line.startswith("Training duration:"):
                m = re.search(r"Training duration:\s*([\d\.]+)", line)
                if m:
                    summary_data["training_duration"] = float(m.group(1))
            elif line.startswith("Rendering duration:"):
                m = re.search(r"Rendering duration:\s*([\d\.]+)", line)
                if m:
                    summary_data["rendering_duration"] = float(m.group(1))
            elif line.startswith("Metrics duration:"):
                m = re.search(r"Metrics duration:\s*([\d\.]+)", line)
                if m:
                    summary_data["metrics_duration"] = float(m.group(1))
            elif line.startswith("Compression Ratio:"):
                m = re.search(r"Compression Ratio:\s*([\d\.]+)%", line)
                if m:
                    summary_data["compression_ratio"] = float(m.group(1))
            elif line.startswith("Original Size:"):
                m = re.search(r"Original Size:\s*([\d\.]+)", line)
                if m:
                    summary_data["original_size"] = float(m.group(1))
            elif line.startswith("Compressed Size:"):
                m = re.search(r"Compressed Size:\s*([\d\.]+)", line)
                if m:
                    summary_data["compressed_size"] = float(m.group(1))
        if (
            "training_duration" in summary_data
            and "rendering_duration" in summary_data
            and "metrics_duration" in summary_data
        ):
            summary_data["total_time"] = (
                summary_data["training_duration"]
                + summary_data["rendering_duration"]
                + summary_data["metrics_duration"]
            )
        experiments_summary[experiment_label] = summary_data
    return experiments_summary


def main():
    output_dir = "output"
    if not os.path.exists(output_dir):
        print(
            f"Error: Directory '{output_dir}' not found. Make sure you're running this script from the correct location."
        )
        return

    global_experiments = []

    # Process each test-type folder
    test_types = [
        d
        for d in os.listdir(output_dir)
        if os.path.isdir(os.path.join(output_dir, d)) and d != "combined_plots"
    ]
    if not test_types:
        print("No test type directories found in 'output'. Nothing to do!")
        return

    for test in test_types:
        test_path = os.path.join(output_dir, test)
        print(f"\nProcessing test type: {test}")

        # Create a "plots" folder inside the test-type directory
        plots_dir = os.path.join(test_path, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        # Parse summary.txt (if it exists) for timing, compression, and file size info
        summary_path = os.path.join(test_path, "summary.txt")
        summary_data = {}
        if os.path.exists(summary_path):
            summary_data = parse_summary(summary_path)
        else:
            print(
                f"Warning: summary.txt not found in {test_path}. Some plots may be incomplete for {test}."
            )

        # Gather experiment data from each subdirectory
        experiments = {}
        for entry in os.listdir(test_path):
            if entry == "plots":
                continue
            entry_path = os.path.join(test_path, entry)
            if os.path.isdir(entry_path):
                results_path = os.path.join(entry_path, "results.json")
                if os.path.exists(results_path):
                    try:
                        with open(results_path, "r") as f:
                            data = json.load(f)
                        key = list(data.keys())[0]
                        metrics = data[key]
                        ssim = metrics.get("SSIM")
                        psnr = metrics.get("PSNR")
                        if ssim is None or psnr is None:
                            print(
                                f"Warning: SSIM or PSNR missing in {results_path}. Skipping experiment '{entry}'."
                            )
                            continue
                        experiments[entry] = experiments.get(entry, {})
                        experiments[entry]["ssim"] = ssim
                        experiments[entry]["psnr"] = psnr
                    except Exception as e:
                        print(
                            f"Error reading {results_path}: {e}. Skipping experiment '{entry}'."
                        )
                else:
                    print(f"Warning: results.json not found in {entry_path}.")

        # Merge summary data into experiments
        for exp, summ in summary_data.items():
            if exp in experiments:
                experiments[exp].update(summ)
            else:
                experiments[exp] = summ

        experiments = {k: v for k, v in experiments.items() if "psnr" in v}
        if not experiments:
            print(
                f"No experiments with valid PSNR data found for test type '{test}'. Skipping."
            )
            continue

        # Sort experiments by PSNR (highest to lowest).
        sorted_experiments = sorted(
            experiments.items(), key=lambda item: item[1]["psnr"], reverse=True
        )
        labels = [item[0] for item in sorted_experiments]
        ssim_values = [item[1]["ssim"] for item in sorted_experiments]
        psnr_values = [item[1]["psnr"] for item in sorted_experiments]
        total_time_values = [
            item[1].get("total_time", 0) for item in sorted_experiments
        ]
        compression_values = [
            item[1].get("compression_ratio", 0) for item in sorted_experiments
        ]

        # Determine dynamic figure height and width for the local plots
        num_exps = len(labels)
        height = max(6, num_exps * 0.3)
        width = get_dynamic_width(labels)

        # --- Produce per-test plots in the test's "plots" folder ---

        # 1. SSIM plot
        ssim_plot_file = os.path.join(plots_dir, f"{test}_ssim.png")
        plt.figure(figsize=(width, height))
        plt.barh(range(num_exps), ssim_values, color="skyblue")
        plt.title(f"SSIM for {test}")
        plt.xlabel("SSIM")
        plt.ylabel("Experiment (parameters)")
        plt.yticks(range(num_exps), labels)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(ssim_plot_file)
        plt.close()
        print(f"Saved SSIM plot for {test} as '{ssim_plot_file}'.")

        # 2. PSNR plot
        psnr_plot_file = os.path.join(plots_dir, f"{test}_psnr.png")
        plt.figure(figsize=(width, height))
        plt.barh(range(num_exps), psnr_values, color="salmon")
        plt.title(f"PSNR for {test}")
        plt.xlabel("PSNR")
        plt.ylabel("Experiment (parameters)")
        plt.yticks(range(num_exps), labels)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(psnr_plot_file)
        plt.close()
        print(f"Saved PSNR plot for {test} as '{psnr_plot_file}'.")

        # 3. Total Time plot
        time_plot_file = os.path.join(plots_dir, f"{test}_time.png")
        if any(total_time_values):
            plt.figure(figsize=(width, height))
            plt.barh(range(num_exps), total_time_values, color="lightgreen")
            plt.title(f"Total Test Duration for {test}")
            plt.xlabel("Time (seconds)")
            plt.ylabel("Experiment (parameters)")
            plt.yticks(range(num_exps), labels)
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(time_plot_file)
            plt.close()
            print(f"Saved Total Time plot for {test} as '{time_plot_file}'.")
        else:
            print(f"No timing data available for {test}. Skipping duration plot.")

        # 4. Compression Ratio plot
        compression_plot_file = os.path.join(plots_dir, f"{test}_compression.png")
        if any(compression_values):
            plt.figure(figsize=(width, height))
            plt.barh(range(num_exps), compression_values, color="violet")
            plt.title(f"Compression Ratio for {test}")
            plt.xlabel("Compression Ratio (%)")
            plt.ylabel("Experiment (parameters)")
            plt.yticks(range(num_exps), labels)
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(compression_plot_file)
            plt.close()
            print(
                f"Saved Compression Ratio plot for {test} as '{compression_plot_file}'."
            )
        else:
            print(
                f"No compression data available for {test}. Skipping compression plot."
            )

        # 5. File Sizes plot
        filesizes_plot_file = os.path.join(plots_dir, f"{test}_filesizes.png")
        original_mb_values = []
        compressed_mb_values = []
        for key, data in sorted_experiments:
            orig = data.get("original_size", 0) / (1024 * 1024)
            comp = data.get("compressed_size", 0) / (1024 * 1024)
            original_mb_values.append(orig)
            compressed_mb_values.append(comp)
        if any(original_mb_values) and any(compressed_mb_values):
            fig, ax = plt.subplots(figsize=(width, height))
            indices = np.arange(num_exps)
            bar_height = 0.4

            ax.barh(
                indices - bar_height / 2,
                original_mb_values,
                height=bar_height,
                color="gray",
                label="Original Size (MB)",
            )
            ax.barh(
                indices + bar_height / 2,
                compressed_mb_values,
                height=bar_height,
                color="orange",
                label="Compressed Size (MB)",
            )

            ax.set_yticks(indices)
            ax.set_yticklabels(labels)
            ax.invert_yaxis()
            ax.set_xlabel("File Size (MB)")
            ax.set_ylabel("Experiment (parameters)")
            ax.set_title(f"File Sizes for {test}")

            ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")

            fig.tight_layout()
            plt.savefig(filesizes_plot_file)
            plt.close()
            print(f"Saved File Sizes plot for {test} as '{filesizes_plot_file}'.")
        else:
            print(
                f"Original or compressed file size data not available for {test}. Skipping file sizes plot."
            )

        for exp_label, metrics in sorted_experiments:
            # Create a unique global label by including the test type
            global_label = f"{test}/{exp_label}"
            global_experiments.append((global_label, metrics))

    if not global_experiments:
        print("No global experiments with valid PSNR data found. Exiting.")
        return

    # Sort global experiments by PSNR descending
    global_experiments.sort(key=lambda item: item[1]["psnr"], reverse=True)

    top_n = 20
    global_experiments = global_experiments[:top_n]

    global_labels = [item[0] for item in global_experiments]
    global_ssim = [item[1]["ssim"] for item in global_experiments]
    global_psnr = [item[1]["psnr"] for item in global_experiments]
    global_total_time = [item[1].get("total_time", 0) for item in global_experiments]
    global_compression = [
        item[1].get("compression_ratio", 0) for item in global_experiments
    ]

    # File sizes: Convert to MB
    global_original_mb = [
        item[1].get("original_size", 0) / (1024 * 1024) for item in global_experiments
    ]
    global_compressed_mb = [
        item[1].get("compressed_size", 0) / (1024 * 1024) for item in global_experiments
    ]

    combined_plots_dir = os.path.join("output", "combined_plots")
    os.makedirs(combined_plots_dir, exist_ok=True)

    num_experiments = len(global_labels)
    height = max(6, num_experiments * 0.3)
    width = get_dynamic_width(global_labels)

    # 1. Global SSIM plot
    combined_ssim_file = os.path.join(combined_plots_dir, "combined_ssim.png")
    plt.figure(figsize=(width, height))
    plt.barh(range(num_experiments), global_ssim, color="skyblue")
    plt.title("Global SSIM Across All Test Types")
    plt.xlabel("SSIM")
    plt.yticks(range(num_experiments), global_labels)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(combined_ssim_file)
    plt.close()
    print(f"Saved global SSIM plot as '{combined_ssim_file}'.")

    # 2. Global PSNR plot
    combined_psnr_file = os.path.join(combined_plots_dir, "combined_psnr.png")
    plt.figure(figsize=(width, height))
    plt.barh(range(num_experiments), global_psnr, color="salmon")
    plt.title("Global PSNR Across All Test Types")
    plt.xlabel("PSNR")
    plt.yticks(range(num_experiments), global_labels)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(combined_psnr_file)
    plt.close()
    print(f"Saved global PSNR plot as '{combined_psnr_file}'.")

    # 3. Global Total Time plot
    if any(global_total_time):
        combined_time_file = os.path.join(combined_plots_dir, "combined_time.png")
        plt.figure(figsize=(width, height))
        plt.barh(range(num_experiments), global_total_time, color="lightgreen")
        plt.title("Global Total Test Duration Across All Test Types")
        plt.xlabel("Time (seconds)")
        plt.yticks(range(num_experiments), global_labels)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(combined_time_file)
        plt.close()
        print(f"Saved global Total Time plot as '{combined_time_file}'.")
    else:
        print("No global timing data available. Skipping global duration plot.")

    # 4. Global Compression Ratio plot
    if any(global_compression):
        combined_compression_file = os.path.join(
            combined_plots_dir, "combined_compression.png"
        )
        plt.figure(figsize=(width, height))
        plt.barh(range(num_experiments), global_compression, color="violet")
        plt.title("Global Compression Ratio Across All Test Types")
        plt.xlabel("Compression Ratio (%)")
        plt.yticks(range(num_experiments), global_labels)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(combined_compression_file)
        plt.close()
        print(f"Saved global Compression Ratio plot as '{combined_compression_file}'.")
    else:
        print("No global compression data available. Skipping global compression plot.")

    # 5. Global File Sizes plot
    if any(global_original_mb) and any(global_compressed_mb):
        combined_filesizes_file = os.path.join(
            combined_plots_dir, "combined_filesizes.png"
        )
        fig, ax = plt.subplots(figsize=(width, height))
        indices = np.arange(num_experiments)
        bar_height = 0.4

        ax.barh(
            indices - bar_height / 2,
            global_original_mb,
            height=bar_height,
            color="gray",
            label="Original Size (MB)",
        )
        ax.barh(
            indices + bar_height / 2,
            global_compressed_mb,
            height=bar_height,
            color="orange",
            label="Compressed Size (MB)",
        )

        ax.set_yticks(indices)
        ax.set_yticklabels(global_labels)
        ax.invert_yaxis()
        ax.set_xlabel("File Size (MB)")
        ax.set_title("Global File Sizes Across All Test Types")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")

        fig.tight_layout()
        plt.savefig(combined_filesizes_file)
        plt.close()
        print(f"Saved global File Sizes plot as '{combined_filesizes_file}'.")
    else:
        print("No global file size data available. Skipping global file sizes plot.")


if __name__ == "__main__":
    main()
