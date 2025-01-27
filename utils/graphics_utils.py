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


def validate_colormaps(colormaps):
    validated_maps = []
    for cmap in colormaps.split(","):
        if cmap not in plt.colormaps():
            raise ValueError(f"Invalid colormap: {cmap}")
        validated_maps.append(cmap)

    if not validated_maps:
        print("No valid colormaps found")
        raise

    return validated_maps


def create_colormaps(cmap_names, num_points=32):
    try:
        # Preallocate numpy arrays
        num_colormaps = len(cmap_names)
        tables = np.zeros((num_colormaps, num_points, 3), dtype=np.float32)
        derivatives = np.zeros((num_colormaps, num_points, 3), dtype=np.float32)

        for i, name in enumerate(cmap_names):
            cmap = plt.cm.get_cmap(name)
            control_points = np.linspace(0.0, 1.0, num_points)

            # Get colors
            colors = cmap(control_points)[:, :3]
            tables[i] = colors

            # Compute derivatives
            derivatives[i, :-1] = (colors[1:] - colors[:-1]) * (num_points - 1)
            derivatives[i, -1] = 0

        return (
            torch.as_tensor(tables, dtype=torch.float32, device="cuda"),
            torch.as_tensor(derivatives, dtype=torch.float32, device="cuda"),
        )

    except Exception as e:
        print(f"Error creating {len(cmap_names)} colormaps: {e}")
        raise
