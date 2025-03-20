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

import json
import os
import sys
import uuid
import time
from argparse import ArgumentParser, Namespace
from random import randint

import torch
from tqdm import tqdm
import piq
import numpy as np

from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import network_gui, render
from scene import GaussianModel, Scene
from utils.debug_utils import save_debug_image
from utils.general_utils import get_expon_lr_func, safe_state
from utils.graphics_utils import create_colormaps, create_opacitymaps
from utils.image_utils import psnr
from utils.loss_utils import bounding_box_regularization, create_window, l1_loss
from utils.validate_args import (
    validate_colormaps,
    validate_dropout,
    validate_resolution,
    validate_spacing,
)

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
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    debug_from,
    interpolate_until,
):
    first_iter = 0
    colormap_tables, derivatives = create_colormaps(
        dataset.colormaps, dataset.num_control_points
    )
    opacity_tables, opac_derivatives = create_opacitymaps(
        num_steps=dataset.opacity_steps
    )
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(opt.train_opacity, opt.train_values)
    scene = Scene(dataset, gaussians, opacity_tables, train_values=opt.train_values)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    if opt.random_background:
        background = torch.rand((3), device="cuda")
    else:
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    depth_l1_weight = get_expon_lr_func(
        opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations
    )

    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    start = time.time()
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn is None:
            network_gui.try_connect()
        while network_gui.conn is not None:
            try:
                net_image_bytes = None
                (
                    custom_cam,
                    do_training,
                    pipe.compute_cov3D_python,
                    keep_alive,
                    scaling_modifer,
                ) = network_gui.receive()
                if custom_cam is not None:
                    net_image = render(
                        custom_cam, gaussians, pipe, background, scaling_modifer
                    )["render"]
                    net_image_bytes = memoryview(
                        (torch.clamp(net_image, min=0, max=1.0) * 255)
                        .byte()
                        .permute(1, 2, 0)
                        .contiguous()
                        .cpu()
                        .numpy()
                    )
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and (
                    (iteration < int(opt.iterations)) or not keep_alive
                ):
                    break
            except Exception:
                network_gui.conn = None

        if iteration <= interpolate_until or not opt.train_values:
            gaussians.interpolate_new_values()

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Pick a random Camera
        viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        render_pkg = render(
            viewpoint_cam,
            gaussians,
            pipe,
            background,
            colormap_tables[viewpoint_cam.colormap_id],
            derivatives[viewpoint_cam.colormap_id],
            opacity_tables[viewpoint_cam.opacitymap_id],
            opac_derivatives[viewpoint_cam.opacitymap_id],
            use_trained_exp=dataset.train_test_exp,
        )
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        ssim_value = piq.multi_scale_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        scaling_loss = (
            torch.mean(torch.norm(1.0 / gaussians._scaling, p=2)) * opt.lambda_scaling
        )
        bound_loss = bounding_box_regularization(gaussians)

        # Combine all losses
        loss = (
            (1.0 - opt.lambda_dssim) * Ll1
            + opt.lambda_dssim * (1.0 - ssim_value)
            + scaling_loss
            + bound_loss
        )

        # Side-by-side debug images
        if DEBUG:
            capture_frequency = 500
            if iteration % capture_frequency == 0:
                save_debug_image(
                    dataset.model_path,
                    gt_image,
                    image,
                    f"debug_{iteration}_cmap_{viewpoint_cam.colormap_id}_omap_{viewpoint_cam.opacitymap_id}.png",
                )

        # Depth regularization
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            Ll1depth_pure = torch.abs((invDepth - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 500 == 0:
                progress_bar.set_postfix(
                    {
                        "Loss": f"{ema_loss_for_log:.{7}f}",
                        "SSIM": f"{ssim_value.item():.{7}f}",
                        "Scaling": f"{scaling_loss.item():.{7}f}",
                        "Bound": f"{bound_loss.item():.{7}f}",
                    }
                )
                progress_bar.update(500)
                print(f"\n Number of Gaussians: {gaussians.get_values.shape[0]}")
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(
                tb_writer,
                iteration,
                Ll1,
                loss,
                l1_loss,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                render,
                (
                    pipe,
                    background,
                    colormap_tables[viewpoint_cam.colormap_id],
                    derivatives[viewpoint_cam.colormap_id],
                    opacity_tables[viewpoint_cam.opacitymap_id],
                    opac_derivatives[viewpoint_cam.opacitymap_id],
                ),
                dataset.train_test_exp,
            )
            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                )
                gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter
                )
                if (
                    iteration > opt.densify_from_iter
                    and iteration % opt.densification_interval == 0
                ):
                    size_threshold = (
                        20 if iteration > opt.opacity_reset_interval else None
                    )
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        0.005,
                        scene.cameras_extent,
                        size_threshold,
                        dataset_args.max_opac_grad,
                        dataset_args.min_gaussian_size,
                    )

                if (
                    iteration % opt.opacity_reset_interval == 0 and opt.train_opacity
                ) or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none=True)
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if iteration in checkpoint_iterations:
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                torch.save(
                    (gaussians.capture(), iteration),
                    os.path.join(scene.model_path, "/chkpnt{iteration}.pth"),
                )
    end = time.time()
    print(f"Total training time: {(end - start):.2f} seconds")
    print(f"Final number of Gaussians: {gaussians.get_values.shape[0]}")


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv("OAR_JOB_ID"):
            unique_str = os.getenv("OAR_JOB_ID")
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(
    tb_writer,
    iteration,
    Ll1,
    loss,
    l1_loss,
    elapsed,
    testing_iterations,
    scene: Scene,
    renderFunc,
    renderArgs,
    train_test_exp,
):
    if tb_writer:
        tb_writer.add_scalar("train_loss_patches/l1_loss", Ll1.item(), iteration)
        tb_writer.add_scalar("train_loss_patches/total_loss", loss.item(), iteration)
        tb_writer.add_scalar("iter_time", elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {"name": "test", "cameras": scene.getTestCameras()},
            {
                "name": "train",
                "cameras": [
                    scene.getTrainCameras()[idx % len(scene.getTrainCameras())]
                    for idx in range(5, 30, 5)
                ],
            },
        )

        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config["cameras"]):
                    image = torch.clamp(
                        renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"],
                        0.0,
                        1.0,
                    )
                    gt_image = torch.clamp(
                        viewpoint.original_image.to("cuda"), 0.0, 1.0
                    )
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2 :]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2 :]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(
                            config["name"]
                            + "_view_{}/render".format(viewpoint.image_name),
                            image[None],
                            global_step=iteration,
                        )
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(
                                config["name"]
                                + "_view_{}/ground_truth".format(viewpoint.image_name),
                                gt_image[None],
                                global_step=iteration,
                            )
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config["cameras"])
                l1_test /= len(config["cameras"])
                print(
                    "\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(
                        iteration, config["name"], l1_test, psnr_test
                    )
                )
                if tb_writer:
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - l1_loss", l1_test, iteration
                    )
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - psnr", psnr_test, iteration
                    )

        if tb_writer:
            tb_writer.add_histogram(
                "scene/opacity_histogram", scene.gaussians.get_opacity, iteration
            )
            tb_writer.add_scalar(
                "total_points", scene.gaussians.get_xyz.shape[0], iteration
            )
        torch.cuda.empty_cache()


def save_args_to_json(args, dataset_args, opt_args, pipe_args):
    args_dict = vars(args)

    # Ensure dataset has a valid model path
    model_path = dataset_args.model_path
    if not model_path:
        print("Warning: model_path is not set. Arguments will not be saved.")
        return

    os.makedirs(model_path, exist_ok=True)

    op_args_dict = vars(opt_args) if hasattr(opt_args, "__dict__") else opt_args
    pp_args_dict = vars(pipe_args) if hasattr(pipe_args, "__dict__") else pipe_args

    dataset_dict = {
        k: v
        for k, v in vars(dataset_args).items()
        if not callable(v) and not k.startswith("__")
    }

    # Merge all argument dictionaries
    all_args = {
        "main_args": args_dict,
        "dataset_args": dataset_dict,
        "optimization_args": op_args_dict,
        "pipeline_args": pp_args_dict,
    }

    # Save merged arguments to JSON
    args_dump_path = os.path.join(model_path, "all_args.json")
    with open(args_dump_path, "w") as f:
        json.dump(all_args, f, indent=4)

    print(f"Arguments saved to {args_dump_path}")


if __name__ == "__main__":
    window = create_window()
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument(
        "--test_iterations",
        nargs="+",
        type=int,
        default=list(range(1000, 30001, 1000)),
    )
    parser.add_argument(
        "--save_iterations",
        nargs="+",
        type=int,
        default=[
            1,
            10_000,
            20_000,
            30_000,
        ],
    )
    parser.add_argument(
        "--interpolate_until",
        type=int,
        default=0,
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--disable_viewer", action="store_true", default=True)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
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
    parser.add_argument(
        "--opacity_steps",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--max_opac_grad",
        type=float,
        default=1,
    )
    parser.add_argument(
        "--min_gaussian_size",
        type=float,
        default=0.001,
    )
    parser.add_argument(
        "--dropout",
        type=validate_dropout,
        default=0.99
    )
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # TODO: Can we remove this GUI server?
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    dataset_args = lp.extract(args)
    dataset_args.colormaps = validate_colormaps(dataset_args.colormaps)
    dataset_args.num_control_points = args.num_control_points
    dataset_args.resolution = args.resolution
    dataset_args.spacing = args.spacing
    dataset_args.opacity_steps = args.opacity_steps
    dataset_args.max_opac_grad = args.max_opac_grad
    dataset_args.min_gaussian_size = args.min_gaussian_size
    dataset_args.dropout = args.dropout
    opt_args = op.extract(args)
    pipe_args = pp.extract(args)
    training(
        dataset_args,
        opt_args,
        pipe_args,
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
        args.interpolate_until,
    )

    # All done
    save_args_to_json(args, dataset_args, opt_args, pipe_args)
    print("\nTraining complete.")
