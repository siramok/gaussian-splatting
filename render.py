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
from argparse import ArgumentParser
from itertools import islice
from os import makedirs
from turtle import st

import torch
import torchvision
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render
from scene import Scene
from utils.general_utils import safe_state
from utils.graphics_utils import create_colormaps
from utils.validate_args import (
    validate_colormaps,
    validate_resolution,
    validate_spacing,
)


def render_set(
    model_path,
    name,
    iteration,
    views,
    gaussians,
    pipeline,
    background,
    colormap_tables,
    derivative_tables,
    train_test_exp,
):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in islice(enumerate(tqdm(views, desc="Rendering progress")), 10):
        rendering = render(
            view,
            gaussians,
            pipeline,
            background,
            view.colormap_id,
            colormap_tables,
            derivative_tables,
            use_trained_exp=train_test_exp,
        )["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(
            rendering, os.path.join(render_path, "{0:05d}".format(idx) + ".png")
        )
        torchvision.utils.save_image(
            gt, os.path.join(gts_path, "{0:05d}".format(idx) + ".png")
        )


def render_sets(
    dataset: ModelParams,
    iteration: int,
    pipeline: PipelineParams,
    skip_train: bool,
    skip_test: bool,
):
    with torch.no_grad():
        gaussians = GaussianModel()
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        colormap_tables, derivative_tables = create_colormaps(dataset.colormaps)

        if not skip_train:
            render_set(
                dataset.model_path,
                "train",
                scene.loaded_iter,
                scene.getTrainCameras(),
                gaussians,
                pipeline,
                background,
                colormap_tables,
                derivative_tables,
                dataset.train_test_exp,
            )

        if not skip_test:
            render_set(
                dataset.model_path,
                "test",
                scene.loaded_iter,
                scene.getTestCameras(),
                gaussians,
                pipeline,
                background,
                colormap_tables,
                derivative_tables,
                dataset.train_test_exp,
            )


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--num_control_points", type=int, default=256)
    parser.add_argument(
        "--resolution",
        type=validate_resolution,
        default="medium",
    )
    parser.add_argument(
        "--spacing",
        type=validate_spacing,
        default=(1, 1, 1),
    )
    args = get_combined_args(parser)
    print(f"Colormaps: {args.colormaps}")
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    dataset = model.extract(args)
    if isinstance(args.colormaps, list):
        dataset.colormaps = args.colormaps
    else:
        dataset.colormaps = validate_colormaps(dataset.colormaps)

    print(f"Colormaps: {dataset.colormaps}")
    dataset.num_control_points = args.num_control_points
    dataset.resolution = args.resolution
    dataset.spacing = args.spacing
    render_sets(
        dataset,
        args.iteration,
        pipeline.extract(args),
        args.skip_train,
        args.skip_test,
    )
