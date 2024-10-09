import os

import matplotlib.pyplot as plt


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
