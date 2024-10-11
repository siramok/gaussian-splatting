import os

import matplotlib.pyplot as plt
import numpy as np


def save_debug_image(path, gt_image, image, filename):
    # Create the debug directory if it doesn't exist
    debug_path = os.path.join(path, "debug")
    os.makedirs(debug_path, exist_ok=True)

    # Convert the images to numpy arrays
    gt_image_np = gt_image.permute(1, 2, 0).cpu().numpy()
    image_np = image.permute(1, 2, 0).detach().cpu().numpy()

    # Create the side-by-side plot
    fig, axs = plt.subplots(1, 2, figsize=(12, 7))

    # Display the ground truth image (left)
    axs[0].imshow(gt_image_np)
    axs[0].set_title("Ground Truth Image")
    axs[0].axis("off")

    # Display the rendered image (right)
    axs[1].imshow(image_np)
    axs[1].set_title("Rendered Image")
    axs[1].axis("off")

    # Save the plot to disk
    plt.tight_layout()
    plt.savefig(os.path.join(debug_path, filename))
    plt.close(fig)


def analyze_array(arr):
    arr = np.array(arr).flatten()
    nan_count = np.isnan(arr).sum()
    non_nan_values = arr[~np.isnan(arr)]

    if len(non_nan_values) > 0:
        avg = np.mean(non_nan_values)
        min_val = np.min(non_nan_values)
        max_val = np.max(non_nan_values)
    else:
        avg = min_val = max_val = None

    print(f"Number of NaN values: {nan_count}")
    print(f"Average of non-NaN values: {avg}")
    print(f"Minimum of non-NaN values: {min_val}")
    print(f"Maximum of non-NaN values: {max_val}")
