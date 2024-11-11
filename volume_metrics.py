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

import os
import sys
import uuid
from argparse import ArgumentParser, Namespace
from random import randint
import numpy as np

import torch
from tqdm import tqdm
import piq

from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import network_gui, render
from scene import GaussianModel, Scene
from utils.debug_utils import save_debug_image, tensor_to_vtk
from utils.general_utils import get_expon_lr_func, safe_state
from utils.image_utils import psnr
from utils.loss_utils import bounding_box_regularization, create_window, l1_loss, l2_loss

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

DEBUG = True


def training(
    dataset,
    opt,
    pipe,
):
    first_iter = 0
    gaussians = GaussianModel()
    scene = Scene(dataset, gaussians, load_iteration=-1)

    # Make ground truth
    v = np.linspace(0.01, 0.99, 50)
    x, y, z = np.meshgrid(v, v, v, indexing='ij')
    samples = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T
    gt_cells = gaussians.interpolator(x.ravel(), y.ravel(), z.ravel()).reshape(50, 50, 50)
    flipped_tensor = np.flip(gt_cells, axis=1)
    rotated_tensor = np.rot90(flipped_tensor, k=1, axes=(2, 0))
    tensor_to_vtk(rotated_tensor, "test_gt.vtk")
    gt = torch.tensor(rotated_tensor.copy()).cuda()

    render_pkg = render(
        gaussians,
        pipe
    )
    cells, visibility_filter, radii = (
        render_pkg["cells"],
        render_pkg["visibility_filter"],
        render_pkg["radii"],
    )

    l1_l = l1_loss(cells, gt)
    mse = torch.mean((cells - gt) ** 2)
    psnr = 20 * torch.log10(torch.tensor(1.0)) - 10 * torch.log10(mse + 1e-8)
    print(f"L1 loss: {l1_l.item()}")
    print(f"L2 loss: {mse}")
    print(f"PSNR: {psnr}")

if __name__ == "__main__":
    window = create_window()
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    args = parser.parse_args(sys.argv[1:])
    print(args.model_path)

    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
    )