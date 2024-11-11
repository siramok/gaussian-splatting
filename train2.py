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
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    debug_from,
):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel()
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    scene.save(0)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    depth_l1_weight = get_expon_lr_func(
        opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations
    )

    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    # Make ground truth
    v = np.linspace(0.01, 0.99, 50)
    x, y, z = np.meshgrid(v, v, v, indexing='ij')
    samples = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T
    gt_cells = gaussians.interpolator(x.ravel(), y.ravel(), z.ravel()).reshape(50, 50, 50)
    flipped_tensor = np.flip(gt_cells, axis=1)
    rotated_tensor = np.rot90(flipped_tensor, k=1, axes=(2, 0))
    tensor_to_vtk(rotated_tensor, "test_gt.vtk")
    gt = torch.tensor(rotated_tensor.copy()).cuda()

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

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

        # ssim_value = piq.multi_scale_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))

        # scaling_loss = torch.mean(1.0 / torch.exp(gaussians._scaling))
        # scaling_modifier = opt.lambda_scaling * torch.sigmoid(scaling_loss - 450).item()

        # bound_loss = bounding_box_regularization(gaussians)

        # Combine all losses
        # loss = (
        #     (1.0 - opt.lambda_dssim) * Ll1
        #     + opt.lambda_dssim * (1.0 - ssim_value)
        #     + scaling_modifier * scaling_loss
        #     + bound_loss
        # )
        loss = l1_l
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

            if iteration % 500 == 0:
                progress_bar.set_postfix(
                    {
                        "Loss": f"{ema_loss_for_log:.{7}f}",
                        # "SSIM": f"{ssim_value.item():.{7}f}",
                        # "Scaling": f"{scaling_modifier * scaling_loss.item():.{7}f}",
                        # "Bound": f"{bound_loss.item():.{7}f}",
                    }
                )
                progress_bar.update(500)
                print("")
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            # training_report(
            #     tb_writer,
            #     iteration,
            #     Ll1,
            #     loss,
            #     l1_loss,
            #     iter_start.elapsed_time(iter_end),
            #     testing_iterations,
            #     scene,
            #     render,
            #     (pipe, background),
            #     dataset.train_test_exp,
            # )
            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                tensor_to_vtk(cells.cpu().numpy(), f"test_{iteration}.vtk")
                
            # Densification
            # if iteration < opt.densify_until_iter:
            #     # Keep track of max radii in image-space for pruning
            #     gaussians.max_radii2D[visibility_filter] = torch.max(
            #         gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
            #     )
            #     gaussians.add_densification_stats(
            #         viewspace_point_tensor, visibility_filter
            #     )

            #     if (
            #         iteration > opt.densify_from_iter
            #         and iteration % opt.densification_interval == 0
            #     ):
            #         size_threshold = (
            #             20 if iteration > opt.opacity_reset_interval else None
            #         )
            #         gaussians.densify_and_prune(
            #             opt.densify_grad_threshold,
            #             0.005,
            #             scene.cameras_extent,
            #             size_threshold,
                    # )

                # if iteration % opt.opacity_reset_interval == 0 or (
                #     dataset.white_background and iteration == opt.densify_from_iter
                # ):
                #     gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            # gaussians.interpolate_new_values()

            if iteration in checkpoint_iterations:
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                torch.save(
                    (gaussians.capture(), iteration),
                    os.path.join(scene.model_path, "/chkpnt{iteration}.pth"),
                )


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
        "--test_iterations", nargs="+", type=int, default=[7_000, 30_000]
    )
    parser.add_argument(
        "--save_iterations", nargs="+", type=int, default=[2_000, 4_000, 6_000, 8_000, 10_000]
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--disable_viewer", action="store_true", default=True)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
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
    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
    )

    # All done
    print("\nTraining complete.")
