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
import shutil
import time
import numpy as np
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
from utils.graphics_utils import create_colormaps, create_opacitymaps
from utils.validate_args import validate_colormaps, validate_opacitymaps, validate_resolution, validate_spacing, validate_dropout


def render_set(
    model_path, name, iteration, views, gaussians, pipeline, background, colormap_tables, derivatives, opacity_tables, opac_derivatives, train_test_exp
):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    if os.path.exists(render_path):
        shutil.rmtree(render_path)
    if os.path.exists(gts_path):  
        shutil.rmtree(gts_path)
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    time_iters = []

    for idx, view in islice(enumerate(tqdm(views, desc="Rendering progress")), len(views)):
        torch.cuda.synchronize()
        start = time.time()
        rendering = render(
            view, gaussians, pipeline, background, colormap_tables[view.colormap_id], derivatives[view.colormap_id], opacity_tables[view.opacitymap_id], opac_derivatives[view.opacitymap_id], use_trained_exp=train_test_exp
        )["render"]
        # gt = view.original_image[0:3, :, :]
        torch.cuda.synchronize()
        time_iters.append(time.time() - start)
        torchvision.utils.save_image(
            rendering, os.path.join(render_path, "{0:05d}".format(idx) + ".png")
        )
        # torchvision.utils.save_image(
        #     gt, os.path.join(gts_path, "{0:05d}".format(idx) + ".png")
        # )
    print(f"Mean time to render a frame: {np.mean(time_iters[20:])}")


def render_sets(
    dataset: ModelParams,
    iteration: int,
    pipeline: PipelineParams,
    skip_train: bool,
    skip_test: bool,
):
    with torch.no_grad():
        opacity_tables, opac_derivatives = create_opacitymaps(options=dataset.opacitymap_options, num_steps=dataset.opacity_steps, num_random=dataset.opacitymap_randoms)
        gaussians = GaussianModel()
        scene = Scene(dataset, gaussians, opacity_tables, load_iteration=iteration, shuffle=False, skip_train=skip_train)

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
                opacity_tables,
                opac_derivatives,
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
                opacity_tables,
                opac_derivatives,
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
        "--opacity_steps",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--opacitymap_randoms",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--dropout",
        type=validate_dropout,
        default=0.01
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
    dataset.opacitymap_options = validate_opacitymaps(dataset.opacitymap_options)

    print(f"Colormaps: {dataset.colormaps}")
    dataset.num_control_points = args.num_control_points
    dataset.resolution = args.resolution
    dataset.spacing = args.spacing
    dataset.dropout = args.dropout
    dataset.opacity_steps = args.opacity_steps
    dataset.opacitymap_randoms = args.opacitymap_randoms
    render_sets(
        dataset,
        args.iteration,
        pipeline.extract(args),
        args.skip_train,
        args.skip_test,
    )
