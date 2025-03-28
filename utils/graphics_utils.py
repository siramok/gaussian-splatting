#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import math
from typing import NamedTuple

import numpy as np
import torch
import matplotlib.pyplot as plt
import time


class BasicPointCloud(NamedTuple):
    points: np.array
    values: np.array


def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)


def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)


def getWorld2View2(R, t, translate=np.array([0.0, 0.0, 0.0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)


def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


def create_colormaps(names, num_points=256):
    all_colors = []
    all_derivatives = []

    for name in names:
        try:
            cmap = plt.cm.get_cmap(name)
            control_points = np.linspace(0.0, 1.0, num_points)
            colors = cmap(control_points)[:, :3]
            colormap_table = torch.tensor(colors, dtype=torch.float32).to("cuda")

            derivatives = np.zeros_like(colors, dtype=np.float32)
            for i in range(num_points - 1):
                derivatives[i] = (colors[i + 1] - colors[i]) * (num_points - 1)
            colormap_derivatives = torch.tensor(derivatives, dtype=torch.float32).to("cuda")

            all_colors.append(colormap_table)
            all_derivatives.append(colormap_derivatives)

        except Exception as e:
            print(f"Error in create_colormaps for '{name}': {e}")
            raise

    return all_colors, all_derivatives


def create_opacitymaps(options=[], num_points=256, num_steps=5, triangular=True, wrap_around=False, slope=1, num_random=0):
    option_to_func = {
        "inv_linear": np.linspace(1.0, 0.0, num_points),
        "linear": np.linspace(0.0, 1.0, num_points),
        "constant0.1": np.ones(num_points) * 0.1,
        "constant0.01": np.ones(num_points) * 0.01,
        "constant0.005": np.ones(num_points) * 0.005,
    }
    np.random.seed(int(time.time()))
    centers = np.random.random(num_random)
    rand_lengths = np.random.random(num_random)
    for i in range(num_random):
        # num_control_points = 16
        # values = np.zeros(num_points)
        
        # control_indices = np.sort(np.random.choice(
        #     range(num_points), 
        #     size=num_control_points, 
        #     replace=False
        # ))
        # for j, idx in enumerate(control_indices):
        #     if j % 2 == 0:
        #         values[idx] = 0
        #     else:
        #         values[idx] = 1
        # for j in range(num_control_points):
        #     start_idx = control_indices[j]
        #     end_idx = control_indices[(j+1) % num_control_points]  # Wrap around
        #     if end_idx <= start_idx:  # We've wrapped around
        #         segments = [
        #             (start_idx, num_points),  # From start_idx to the end
        #             (-1, end_idx)  # From the beginning to end_idx
        #         ]
        #         total_distance = (num_points - start_idx) + end_idx
        #     else:
        #         segments = [(start_idx, end_idx)]
        #         total_distance = end_idx - start_idx
            
        #     if total_distance > 1:  # Only interpolate if there are points between
        #         start_val = values[start_idx]
        #         end_val = values[end_idx]
                
        #         for segment_start, segment_end in segments:
        #             for k in range(segment_start + 1, segment_end):
        #                 # Calculate position relative to the full wraparound distance
        #                 if end_idx <= start_idx and k < end_idx:
        #                     # We're in the second segment (wrapped around)
        #                     distance_from_start = (num_points - start_idx) + k
        #                 else:
        #                     distance_from_start = k - start_idx
                        
        #                 alpha = distance_from_start / total_distance
        #                 values[k] = start_val * (1 - alpha) + end_val * alpha
        indices = np.linspace(0, 1, num_points)
        center = centers[i]
        rand_length = rand_lengths[i]
        values = np.zeros(num_points)
        
        for j, x in enumerate(indices):
            # Calculate shortest distance considering wrap-around
            if wrap_around:
                dist = min(abs(x - center), abs(x - (center - 1)), abs(x - (center + 1)))
            else:
                dist = abs(x - center)
            # Make opacity 1 at center and 0 at furthest point from center
            values[j] = max(0, 1 - (dist * 2 * slope) / rand_length)   
        option_to_func[f"random{i}"] = values
    options.extend([f"random{i}" for i in range(num_random)])
    opacs = []
    opac_derivatives = []
    for option in options:
        try:
            opac = option_to_func[option]
            opac_table = torch.tensor(opac, dtype=torch.float32).to("cuda")

            # Precompute derivatives
            derivatives = np.zeros_like(opac, dtype=np.float32)
            for i in range(num_points - 1):
                derivatives[i] = (opac[i + 1] - opac[i]) * (num_points - 1)

            # Convert derivatives to float32 and GPU tensor
            opac_derivative = torch.tensor(derivatives, dtype=torch.float32).to("cuda")

            opacs.append(opac_table)
            opac_derivatives.append(opac_derivative)

        except Exception as e:
            print(f"Error in create_opacitymaps: {e}")
            raise
    try:
        if num_steps > 0:
            if not triangular:
                indices = np.arange(num_points)
                bins = np.linspace(0, num_points, num_steps+1).astype(int)
                
                for arr in [((indices >= start - 1) & (indices < end + 1)).astype(np.float32) for start, end in zip(bins[:-1], bins[1:])]:
                    # arr = arr * 0.5
                    opac_table = torch.tensor(arr, dtype=torch.float32).to("cuda")

                    # Precompute derivatives
                    derivatives = np.zeros_like(arr, dtype=np.float32)
                    for i in range(num_points - 1):
                        derivatives[i] = (arr[i + 1] - arr[i]) * (num_points - 1)

                    # Convert derivatives to float32 and GPU tensor
                    opac_derivative = torch.tensor(derivatives, dtype=torch.float32).to("cuda")

                    opacs.append(opac_table)
                    opac_derivatives.append(opac_derivative)
            else: 
                indices = np.linspace(0, 1, num_points)
                step_size = 1.0 / num_steps
                
                for step in range(num_steps):
                    center = step * step_size + step_size / 2
                    arr = np.zeros(num_points, dtype=np.float32)
                    
                    for i, x in enumerate(indices):
                        # Calculate shortest distance considering wrap-around
                        if wrap_around:
                            dist = min(abs(x - center), abs(x - (center - 1)), abs(x - (center + 1)))
                        else:
                            dist = abs(x - center)
                        # Make opacity 1 at center and 0 at furthest point from center
                        arr[i] = max(0, 1 - (dist * 2 * slope * (num_steps / 2)))
                    
                    opac_table = torch.tensor(arr, dtype=torch.float32).to("cuda")
                    
                    # Compute derivatives
                    derivatives = np.zeros_like(arr, dtype=np.float32)
                    for i in range(num_points - 1):
                        derivatives[i] = (arr[i + 1] - arr[i]) * (num_points - 1)
                        
                    opac_derivative = torch.tensor(derivatives, dtype=torch.float32).to("cuda")
                    
                    opacs.append(opac_table)
                    opac_derivatives.append(opac_derivative)
            
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot each opacity map
        x = np.linspace(0, 1, num_points)
        for i, opac in enumerate(opacs):
            # Convert from GPU tensor to numpy array
            opac_np = opac.cpu().numpy()
            plt.plot(x, opac_np, label=f'Step {i+1}', alpha=0.7)
        
        # Add combined plot
        combined = np.zeros(num_points)
        for opac in opacs:
            combined += opac.cpu().numpy()
        plt.plot(x, combined, '--', label='Combined', color='black', linewidth=2)
        
        # Customize plot
        plt.xlabel('Position')
        plt.ylabel('Opacity')
        plt.title(f'Opacity Maps (n_steps={num_steps})')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save plot
        plt.savefig("opac.png")
        plt.close()
            
    except Exception as e:
        print(f"Error in create_opacitymaps: {e}")
        raise

    return opacs, opac_derivatives
