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
    scene = Scene(dataset, gaussians, load_iteration=-1, normalize=True)

    # Make ground truth
    cell_size = 0.02
    num_cells = [
        int(np.ceil((gaussians.maxes[0] - gaussians.mins[0]) / cell_size)),
        int(np.ceil((gaussians.maxes[1] - gaussians.mins[1]) / cell_size)),
        int(np.ceil((gaussians.maxes[2] - gaussians.mins[2]) / cell_size))
    ]
    v1 = np.flip(np.arange(gaussians.mins[0] + cell_size/2, 
                      gaussians.mins[0] + cell_size * num_cells[0],
                      cell_size))
    v2 = np.flip(np.arange(gaussians.mins[1] + cell_size/2, 
                      gaussians.mins[1] + cell_size * num_cells[1],
                      cell_size))
    v3 = np.arange(gaussians.mins[2] + cell_size/2, 
                      gaussians.mins[2] + cell_size * num_cells[2],
                      cell_size)
    print(gaussians._xyz.detach().cpu().numpy().shape)
    print(v1.shape)
    print(num_cells)
    z, y, x = np.meshgrid(v3, v2, v1, indexing='ij')
    samples = np.stack((x.ravel(), y.ravel(), z.ravel()), axis=1)
    # print(x)
    # print(samples.shape)
    # print(samples)
    gt_cells = gaussians.interpolator(samples).reshape(num_cells)
    # print(gt_cells.shape)
    # print(gt_cells)
    # flipped_tensor = np.flip(gt_cells, axis=1)
    # rotated_tensor = np.rot90(gt_cells, k=1, axes=(2, 0))
    tensor_to_vtk(gt_cells, "test_gt.vtk")
    gt = torch.tensor(gt_cells.copy()).cuda()

    pipe.debug = True
    render_pkg = render(
        gaussians,
        pipe,
        cell_size
    )
    cells, visibility_filter, radii = (
        render_pkg["cells"],
        render_pkg["visibility_filter"],
        render_pkg["radii"],
    )
    print(cells.shape)

    l1_l = l1_loss(cells, gt)
    mse = torch.mean((cells - gt) ** 2)
    psnr = 20 * torch.log10(torch.tensor(1.0)) - 10 * torch.log10(mse + 1e-8)
    print(f"L1 loss: {l1_l.item()}")
    print(f"L2 loss: {mse}")
    print(f"PSNR: {psnr}")
    tensor_to_vtk(cells.detach().cpu().numpy(), f"test.vtk")

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