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

import torch
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)

from scene.gaussian_model import GaussianModel


def render(
    pc: GaussianModel,
    pipe,
    bg=-1.0,
    scaling_modifier=1.0,
):
    """
    Render the scene.

    """

    raster_settings = GaussianRasterizationSettings(
        volume_mins=[0.0, 0.0, 0.0],
        volume_maxes=[1.0, 1.0, 1.0],
        cell_size=0.02,
        bg=bg,
        scale_modifier=scaling_modifier,
        debug=pipe.debug,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    scales = pc.get_scaling
    rotations = pc.get_rotation

    values = pc.get_values

    # Rasterize visible Gaussians to cells, obtain their radii
    out_cells, radii = rasterizer(
        means3D=means3D,
        scales=scales,
        rotations=rotations,
        values=values,
    )

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    out = {
        "cells": out_cells,
        "visibility_filter": (radii > 0).nonzero(),
        "radii": radii,
    }

    return out
