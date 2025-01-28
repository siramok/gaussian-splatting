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


def create_colormap(name="viridis", num_points=32):
    try:
        # Get the colormap from matplotlib
        cmap = plt.cm.get_cmap(name)

        # Generate control points in the range [0, 1]
        control_points = np.linspace(0.0, 1.0, num_points)

        # Get RGBA colors for each control point
        colors = cmap(control_points)[:, :3]  # Only take RGB (discard alpha)

        # Convert to float32 for GPU efficiency
        colormap_table = torch.tensor(colors, dtype=torch.float32).to("cuda")

        # Precompute derivatives
        derivatives = np.zeros_like(colors, dtype=np.float32)
        for i in range(num_points - 1):
            derivatives[i] = (colors[i + 1] - colors[i]) * (num_points - 1)

        # Only n-1 intervals when n points between [0,1]
        derivatives[-1] = 0

        # Convert derivatives to float32 and GPU tensor
        colormap_derivatives = torch.tensor(derivatives, dtype=torch.float32).to("cuda")

        return colormap_table, colormap_derivatives

    except Exception as e:
        print(f"Error in create_colormap: {e}")
        raise


def create_opacitymap(num_points=256):
    try:
        # Generate control points in the range [0, 1]
        linear_opac = np.linspace(0.0, 1.0, num_points)

        # Convert to float32 for GPU efficiency
        opac_table = torch.tensor(linear_opac, dtype=torch.float32).to("cuda")

        # Precompute derivatives
        derivatives = np.zeros_like(linear_opac, dtype=np.float32)
        for i in range(num_points - 1):
            derivatives[i] = (linear_opac[i + 1] - linear_opac[i]) * (num_points - 1)

        # Only n-1 intervals when n points between [0,1]
        derivatives[-1] = 0

        # Convert derivatives to float32 and GPU tensor
        opac_derivatives = torch.tensor(derivatives, dtype=torch.float32).to("cuda")

        return opac_table, opac_derivatives

    except Exception as e:
        print(f"Error in create_colormap: {e}")
        raise