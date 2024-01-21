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

import torch
import numpy as np
from arguments import PipelineParams
from gaussian_renderer import render
from scene.cameras import Camera
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

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

    def __init__(self, sh_degree : int):
        '''
        Initializes the Gaussian model's tensors

        Parameters:
        sh_degree (int): The degree of the spherical harmonics used to represent the model
        '''
        # Added .cuda(), previously was not there
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0).cuda()
        self._features_dc = torch.empty(0).cuda()
        self._features_rest = torch.empty(0).cuda()
        self._scaling = torch.empty(0).cuda()
        self._rotation = torch.empty(0).cuda()
        self._opacity = torch.empty(0).cuda()
        self.max_radii2D = torch.empty(0).cuda()
        self.xyz_gradient_accum = torch.empty(0).cuda()
        self.denom = torch.empty(0).cuda()
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

        # PROFILING
        self.iteration = 0
        self.stats = [] # For logging iterations and number of points

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

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
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        self.iteration = iteration
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01)) # Reset opacity to 0.01
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity") # Replace the opacity tensor in the optimizer
        self._opacity = optimizable_tensors["opacity"] # Update the opacity tensor

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        """
        Replaces a specified parameter tensor in the model's optimizer with a new tensor.

        This function is essential when the parameters of a model are dynamically modified during training. 
        It ensures that the optimizer's internal state is updated to reflect changes in the model's parameters. 
        Specifically, it resets the optimizer's state regarding the exponential moving average (exp_avg) and the 
        squared exponential moving average (exp_avg_sq) of the gradients for the new tensor, as the historical 
        gradient information of the previous tensor is no longer relevant.

        Parameters:
        tensor (torch.Tensor): The new tensor that is to replace an existing parameter in the optimizer.
        name (str): The name identifier of the parameter to be replaced. This should match the 'name' key in one of 
                    the optimizer's parameter groups.

        Returns:
        dict: A dictionary containing the updated parameter tensor, keyed by the parameter's name.

        Note:
        This method directly modifies the optimizer attached to the instance of the class it is a part of. It is 
        assumed that the optimizer has an attribute `param_groups`, a common attribute in PyTorch optimizers.

        Example Usage:
        model.replace_tensor_to_optimizer(new_tensor, 'weight')
        """
        optimizable_tensors = {}
        for group in self.optimizer.param_groups: # For each group in the optimizer
            if group["name"] == name: # If the name of the group is the same as the name of the tensor we want to replace
                stored_state = self.optimizer.state.get(group['params'][0], None) # Get the state of the optimizer for this group
                stored_state["exp_avg"] = torch.zeros_like(tensor) # Set the exp_avg to a tensor of zeros with the same shape as the tensor we want to replace
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor) # Set the exp_avg_sq to a tensor of zeros with the same shape as the tensor we want to replace

                del self.optimizer.state[group['params'][0]] # Delete the state of the optimizer for this group
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True)) # Set the parameter of the group to the tensor we want to replace
                self.optimizer.state[group['params'][0]] = stored_state # Set the state of the optimizer for this group to the stored state

                optimizable_tensors[group["name"]] = group["params"][0] # Add the parameter of the group to the optimizable tensors
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False) # If the norm of the gradient is greater than the threshold, we select the point
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent) # If the max scaling of the point is less than the percent dense, we select the point
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        print("Densifying and pruning...")
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        self.append_stats()

        torch.cuda.empty_cache()

    def random_subsample(self, num_points):
        # Assuming self.point_cloud is a tensor representing your point cloud.
        total_points = self.get_xyz.shape[0]

        # Ensure num_points is not greater than the total number of points.
        num_points = min(num_points, total_points)

        # Randomly select indices to keep. torch.randperm generates a permutation of indices.
        keep_indices = torch.randperm(total_points)[:num_points]

        # Create a mask of True (for keeping points) and False (for removing points).
        keep_mask = torch.zeros(total_points, dtype=torch.bool)
        keep_mask[keep_indices] = True

        # Prune points that are not in keep_mask.
        self.prune_points(~keep_mask)

    def get_stats(self):
        return {
            "iter": self.iteration,
            "num_points": self.get_xyz.shape[0],
            "memory": torch.cuda.memory_allocated() / 1024 / 1024 # in MB
        }

    def __len__(self):
        return self.get_xyz.shape[0]

    def append_stats(self):
        self.stats.append(self.get_stats())

    def get_camera_visbility_mask(self, camera: Camera):
        with torch.no_grad():
            # Alternatively, we could do another implementation for CPU, but now its not required yet.
            pp = PipelineParams()
            render_pkg = render(camera, self, pp, torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"))
            return render_pkg["visibility_filter"]
    
    def calculate_bounding_box(self):
        with torch.no_grad():
            return torch.min(self.get_xyz, dim=0).values, torch.max(self.get_xyz, dim=0).values
        
    def calculate_occupied_grids(self, side_length=10.):
        with torch.no_grad():
            min_point, max_point = self.calculate_bounding_box()

            # Convert coordinates to grid indices
            min_indices = torch.floor(min_point / side_length).int()
            max_indices = torch.ceil(max_point / side_length).int()

            # Generate grid cell bounding boxes
            occupied_grids = []
            for x in range(min_indices[0], max_indices[0]):
                for y in range(min_indices[1], max_indices[1]):
                    for z in range(min_indices[2], max_indices[2]):
                        grid_min = torch.tensor([x, y, z], dtype=torch.float32) * side_length
                        grid_max = grid_min + side_length
                        occupied_grids.append((grid_min, grid_max))

            return occupied_grids
        
    def split_to_grid(self, side_length=10.):
        occupied_grids = self.calculate_occupied_grids(side_length)
        sub_models = []

        for grid_min, grid_max in occupied_grids:
            grid_min = grid_min.cuda()
            grid_max = grid_max.cuda()
            # Determine points within this grid cell
            in_grid = (self._xyz >= grid_min) & (self._xyz < grid_max)
            in_grid_mask = in_grid.all(dim=1)

            if in_grid_mask.any():
                # Create a new GaussianModel for this grid cell
                sub_model = GaussianModel(self.max_sh_degree)
                sub_model._xyz = nn.Parameter(self._xyz[in_grid_mask])
                sub_model._features_dc = nn.Parameter(self._features_dc[in_grid_mask])
                sub_model._features_rest = nn.Parameter(self._features_rest[in_grid_mask])
                sub_model._scaling = nn.Parameter(self._scaling[in_grid_mask])
                sub_model._rotation = nn.Parameter(self._rotation[in_grid_mask])
                sub_model._opacity = nn.Parameter(self._opacity[in_grid_mask])
                # max_radii2D, xyz_gradient_accum, and denom
                sub_model.max_radii2D = self.max_radii2D[in_grid_mask].cuda()
                sub_model.xyz_gradient_accum = self.xyz_gradient_accum[in_grid_mask].cuda()
                sub_model.denom = self.denom[in_grid_mask].cuda()
                
                sub_model.active_sh_degree = self.active_sh_degree
                sub_model.max_sh_degree = self.max_sh_degree
                # Copy other necessary properties and setup functions
                sub_model.setup_functions()
                # Add to the list of sub-models
                sub_models.append(sub_model)

        return sub_models
    
    def append(self, other_model):
        """
        Appends the contents of another GaussianModel to this one.

        Parameters:
        other_model (GaussianModel): The model to be appended to this one.
        """
        if not isinstance(other_model, GaussianModel):
            raise ValueError("other_model must be an instance of GaussianModel")

        # Concatenate the properties of both models
        self._xyz = torch.cat([self._xyz, other_model._xyz], dim=0)
        self._features_dc = torch.cat([self._features_dc, other_model._features_dc], dim=0)
        self._features_rest = torch.cat([self._features_rest, other_model._features_rest], dim=0)
        self._scaling = torch.cat([self._scaling, other_model._scaling], dim=0)
        self._rotation = torch.cat([self._rotation, other_model._rotation], dim=0)
        self._opacity = torch.cat([self._opacity, other_model._opacity], dim=0)

        # Update max_radii2D, xyz_gradient_accum, and denom if they are not empty
        if self.max_radii2D.numel() > 0 and other_model.max_radii2D.numel() > 0:
            self.max_radii2D = torch.cat([self.max_radii2D, other_model.max_radii2D], dim=0).cuda()

        if self.xyz_gradient_accum.numel() > 0 and other_model.xyz_gradient_accum.numel() > 0:
            self.xyz_gradient_accum = torch.cat([self.xyz_gradient_accum, other_model.xyz_gradient_accum], dim=0).cuda()

        if self.denom.numel() > 0 and other_model.denom.numel() > 0:
            self.denom = torch.cat([self.denom, other_model.denom], dim=0).cuda()

        # Ensure the active_sh_degree is set to the maximum of the two models
        self.active_sh_degree = max(self.active_sh_degree, other_model.active_sh_degree)

    def append_multiple(self, other_models):
        """
        Appends the contents of multiple GaussianModels to this one.

        Parameters:
        other_models (list): A list of GaussianModels to be appended to this one.
        """
        for other_model in other_models:
            self.append(other_model)

    def add_densification_stats(
            self, 
            viewspace_point_tensor, # Points in 3D space
            update_filter # Mask that determines which points are updated
        ):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        # Add length of gradient, gets first two components of gradient (x,y) in VIEW space. Z (depth) may be irrelevant.
        self.denom[update_filter] += 1 # Add 1 to denominator for each point, so we can average the gradient later