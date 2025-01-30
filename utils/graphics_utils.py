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

            derivatives = np.zeros_like(colors, dtype=np.float32)
            for i in range(num_points - 1):
                derivatives[i] = (colors[i + 1] - colors[i]) * (num_points - 1)
            derivatives[-1] = 0

            all_colors.append(colors)
            all_derivatives.append(derivatives)

        except Exception as e:
            print(f"Error in create_colormaps for '{name}': {e}")
            raise

    all_colors_np = np.stack(all_colors, axis=0).astype(np.float32)
    all_colors_np = all_colors_np.reshape(-1, num_points * 3)
    all_derivatives_np = np.stack(all_derivatives, axis=0).astype(np.float32)
    all_derivatives_np = all_derivatives_np.reshape(-1, num_points * 3)

    colormap_tensor = torch.from_numpy(all_colors_np).to("cuda")
    derivative_tensor = torch.from_numpy(all_derivatives_np).to("cuda")

    return colormap_tensor, derivative_tensor
