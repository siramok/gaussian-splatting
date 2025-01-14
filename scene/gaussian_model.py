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

from utils.system_utils import mkdir_p
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import (
    build_rotation,
    build_scaling_rotation,
    get_expon_lr_func,
    inverse_sigmoid,
    strip_symmetric,
)
import json
import os

import numpy as np
import pyvista as pv
import torch
from plyfile import PlyData, PlyElement
from scipy.interpolate import NearestNDInterpolator
from simple_knn._C import distCUDA2
from torch import nn


class GaussianModel:
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self):
        self._xyz = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._values = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.interpolator = None
        self.interpolation_threshold = 0.05
        self.interpolation_mask = None
        self.last_interpolated_xyz = None
        self.should_interpolate = False
        self.bounding_box = None
        self.setup_functions()

    def capture(self):
        return (
            self._xyz,
            self._scaling,
            self._rotation,
            self._opacity,
            self._values,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        (
            self._xyz,
            self._scaling,
            self._rotation,
            self._opacity,
            self._values,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale,
        ) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
        self.last_interpolated_xyz = self._xyz.clone()
        self.interpolation_mask = np.full(self._xyz.shape[0], True)
        self.should_interpolate = True

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_values(self):
        return self._values

    @property
    def get_exposure(self):
        return self._exposure

    def get_exposure_from_name(self, image_name):
        if self.pretrained_exposures is None:
            return self._exposure[self.exposure_mapping[image_name]]
        else:
            return self.pretrained_exposures[image_name]

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self._rotation
        )

    def create_from_pcd(
        self,
        pcd: BasicPointCloud,
        cam_infos: int,
        spatial_lr_scale: float,
        mesh: pv.PolyData,
    ):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()

        print(f"Number of points at initialisation : {fused_point_cloud.shape[0]}")

        dist2 = torch.clamp_min(
            distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()),
            0.0000001,
        )
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(
            (0.01)
            * torch.ones(
                (fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"
            )
        )

        values = torch.tensor(pcd.values, dtype=torch.float, device="cuda").reshape(
            -1, 1
        )

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._values = nn.Parameter(values.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.exposure_mapping = {
            cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)
        }
        self.pretrained_exposures = None
        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))

        self.process_mesh(mesh)
        self.last_interpolated_xyz = self._xyz.clone()
        self.interpolation_mask = np.full(self._xyz.shape[0], True)
        self.should_interpolate = True

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        optimizer_params = [
            {
                "params": [self._xyz],
                "lr": training_args.position_lr_init * self.spatial_lr_scale,
                "name": "xyz",
            },
            {
                "params": [self._opacity],
                "lr": training_args.opacity_lr,
                "name": "opacity",
            },
            {
                "params": [self._scaling],
                "lr": training_args.scaling_lr,
                "name": "scaling",
            },
            {
                "params": [self._rotation],
                "lr": training_args.rotation_lr,
                "name": "rotation",
            },
                        {
                "params": [self._values],
                "lr": training_args.values_lr,
                "name": "value",
            },
        ]

        self.optimizer = torch.optim.Adam(optimizer_params, lr=0.0, eps=1e-15)
        if self.pretrained_exposures is None:
            self.exposure_optimizer = torch.optim.Adam([self._exposure])

        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )

        self.exposure_scheduler_args = get_expon_lr_func(
            training_args.exposure_lr_init,
            training_args.exposure_lr_final,
            lr_delay_steps=training_args.exposure_lr_delay_steps,
            lr_delay_mult=training_args.exposure_lr_delay_mult,
            max_steps=training_args.iterations,
        )

    def update_learning_rate(self, iteration):
        """Learning rate scheduling per step"""
        if self.pretrained_exposures is None:
            for param_group in self.exposure_optimizer.param_groups:
                param_group["lr"] = self.exposure_scheduler_args(iteration)

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group["lr"] = lr
                return lr

    def construct_list_of_attributes(self):
        attributes = ["x", "y", "z", "value", "opacity"]
        for i in range(self._scaling.shape[1]):
            attributes.append("scale_{}".format(i))
        for i in range(self._rotation.shape[1]):
            attributes.append("rot_{}".format(i))
        return attributes

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        values = self._values.detach().cpu().numpy()

        dtype_full = [
            (attribute, "f4") for attribute in self.construct_list_of_attributes()
        ]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, values, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

        # Also produce an ascii version of the .ply file
        self.convert_ply_to_ascii(path)

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(
            torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01)
        )
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, mesh, use_train_test_exp=False):
        plydata = PlyData.read(path)
        if use_train_test_exp:
            exposure_file = os.path.join(
                os.path.dirname(path), os.pardir, os.pardir, "exposure.json"
            )
            if os.path.exists(exposure_file):
                with open(exposure_file, "r") as f:
                    exposures = json.load(f)
                self.pretrained_exposures = {
                    image_name: torch.FloatTensor(exposures[image_name])
                    .requires_grad_(False)
                    .cuda()
                    for image_name in exposures
                }
                print("Pretrained exposures loaded.")
            else:
                print(f"No exposure to be loaded at {exposure_file}")
                self.pretrained_exposures = None

        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        scale_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("scale_")
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("rot")
        ]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        values = np.asarray(plydata.elements[0]["value"])[..., np.newaxis]

        self._xyz = nn.Parameter(
            torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._opacity = nn.Parameter(
            torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(
                True
            )
        )
        self._scaling = nn.Parameter(
            torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._rotation = nn.Parameter(
            torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._values = nn.Parameter(
            torch.tensor(values, dtype=torch.float, device="cuda").requires_grad_(True)
        )

        self.process_mesh(mesh)
        self.last_interpolated_xyz = self._xyz.clone()
        self.interpolation_mask = np.full(len(self._values), True)

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group["params"][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    (group["params"][0][mask].requires_grad_(True))
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    group["params"][0][mask].requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._values = optimizable_tensors["value"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

        self.last_interpolated_xyz = self.last_interpolated_xyz[valid_points_mask]
        self.interpolation_mask = self.interpolation_mask[
            valid_points_mask.detach().cpu().numpy()
        ]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                    dim=0,
                )

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(
        self,
        new_xyz,
        new_opacities,
        new_scaling,
        new_rotation,
        new_values
    ):
        d = {
            "xyz": new_xyz,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
            "value": new_values
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)

        # The sizes may not be the same, which necessitates extending the arrays
        # used for interpolation
        new_size = optimizable_tensors["xyz"].shape[0]
        old_size = self._xyz.shape[0]

        # We always want the interpolation mask to be the same size as the incoming xyz tensor
        interpolation_mask = np.full(new_size, False)

        if new_size > old_size:
            # Always interpolate the new points
            interpolation_mask[old_size:] = True

            # Extend these tensors to avoid size mismatches during interpolation
            self.last_interpolated_xyz = torch.cat(
                (self.last_interpolated_xyz, optimizable_tensors["xyz"][old_size:]),
                dim=0,
            )
            self._values = torch.cat(
                (
                    self._values,
                    torch.tensor(
                        np.zeros((new_size - old_size, 1)),
                        dtype=torch.float,
                        device="cuda",
                    ),
                ),
                dim=0,
            )

        # Compute the distances between the new points and the last interpolated points
        new_xyz = optimizable_tensors["xyz"][:old_size]
        diff = new_xyz - self.last_interpolated_xyz[:old_size]
        distances = torch.norm(diff, dim=1)

        # If a Gaussian's position has moved more than the threshold, re-interpolate its value
        interpolation_mask[:old_size] = (
            (distances > self.interpolation_threshold).detach().cpu().numpy()
        )

        self.interpolation_mask = interpolation_mask
        # Only bother interpolating if there are any points that need updating
        self.should_interpolate = np.any(interpolation_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._values = optimizable_tensors["value"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[: grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values
            > self.percent_dense * scene_extent,
        )

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[
            selected_pts_mask
        ].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(
            self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
        )
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        new_values = self._values[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(
            new_xyz,
            new_opacity,
            new_scaling,
            new_rotation,
            new_values
        )

        prune_filter = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool),
            )
        )
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(
            torch.norm(grads, dim=-1) >= grad_threshold, True, False
        )
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values
            <= self.percent_dense * scene_extent,
        )

        new_xyz = self._xyz[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_values = self._values[selected_pts_mask]

        self.densification_postfix(
            new_xyz,
            new_opacities,
            new_scaling,
            new_rotation,
            new_values,
        )

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(
                torch.logical_or(prune_mask, big_points_vs), big_points_ws
            )
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True
        )
        self.denom[update_filter] += 1

    def process_mesh(self, mesh):
        if mesh is None:
            return

        points = mesh.points
        values = mesh.get_array("value").reshape(-1, 1)
        self.interpolator = NearestNDInterpolator(
            points, values, tree_options={"leafsize": 30}
        )

        min_x, max_x = points[:, 0].min(), points[:, 0].max()
        min_y, max_y = points[:, 1].min(), points[:, 1].max()
        min_z, max_z = points[:, 2].min(), points[:, 2].max()
        self.bounding_box = (min_x, max_x), (min_y, max_y), (min_z, max_z)

    def interpolate_new_values(self):
        # Return early if there are no new points to interpolate
        if self.interpolator is None or not self.should_interpolate:
            return

        # Filter out the positions that need to be interpolated
        gaussian_positions = self._xyz.detach().cpu().numpy()
        gaussian_positions = gaussian_positions[self.interpolation_mask]

        interpolated_values = self._values.detach().cpu().numpy()
        interpolated_values[self.interpolation_mask] = self.interpolator(
            gaussian_positions
        )
        interpolated_values = np.nan_to_num(interpolated_values, nan=0.0)

        new_values = torch.tensor(
            interpolated_values, dtype=torch.float, device="cuda"
        ).reshape(-1, 1)

        self._values = nn.Parameter(new_values.requires_grad_(False))

        # Update the last interpolated positions, and reset the interpolation mask
        self.last_interpolated_xyz[self.interpolation_mask] = self._xyz[
            self.interpolation_mask
        ]
        self.interpolation_mask = np.full(self._xyz.shape[0], False)
        self.should_interpolate = False

    def convert_ply_to_ascii(self, binary_ply_file_path):
        ascii_ply_file_path = binary_ply_file_path.replace(".ply", "_ascii.ply")

        ply_data = PlyData.read(binary_ply_file_path)

        with open(ascii_ply_file_path, "w") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")

            for element in ply_data.elements:
                f.write(f"element {element.name} {element.count}\n")
                for prop in element.properties:
                    f.write(f"property float {prop.name}\n")

            f.write("end_header\n")

            for element in ply_data.elements:
                for row in element.data:
                    f.write(" ".join(str(val) for val in row) + "\n")
