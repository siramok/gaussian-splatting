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


def create_opacitymaps(options=["inv_linear", "linear"], num_points=256, num_steps=0):
    option_to_func = {
        "inv_linear": np.linspace(1.0, 0.0, num_points),
        "linear": np.linspace(0.0, 1.0, num_points),
        "constant": np.ones(num_points) * 0.01
    }
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
        indices = np.arange(num_points)
        bins = np.linspace(0, num_points, num_steps+1).astype(int)
        
        for arr in [((indices >= start) & (indices < end)).astype(np.float32) for start, end in zip(bins[:-1], bins[1:])]:
            opac_table = torch.tensor(arr, dtype=torch.float32).to("cuda")

            # Precompute derivatives
            derivatives = np.zeros_like(arr, dtype=np.float32)
            for i in range(num_points - 1):
                derivatives[i] = (arr[i + 1] - arr[i]) * (num_points - 1)

            # Convert derivatives to float32 and GPU tensor
            opac_derivative = torch.tensor(derivatives, dtype=torch.float32).to("cuda")

            opacs.append(opac_table)
            opac_derivatives.append(opac_derivative)
    except Exception as e:
        print(f"Error in create_opacitymaps: {e}")
        raise

    # opac = np.zeros(num_points)
    # opac[-1] = 1
    # opac_table = torch.tensor(opac, dtype=torch.float32).to("cuda")
    # derivatives = np.zeros_like(opac, dtype=np.float32)
    # for i in range(num_points - 1):
    #     derivatives[i] = (opac[i + 1] - opac[i]) * (num_points - 1)
    # opac_derivative = torch.tensor(derivatives, dtype=torch.float32).to("cuda")
    # opacs.append(opac_table)
    # opac_derivatives.append(opac_derivative)

    return opacs, opac_derivatives
