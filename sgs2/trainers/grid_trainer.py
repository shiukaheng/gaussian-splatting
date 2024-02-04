from dataclasses import dataclass
from random import randint
import PIL
from matplotlib import pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from gaussian_renderer import network_gui, render
from scene.cameras import Camera
from sgs2.gaussian import GaussianModel
from sgs2.helpers import PipelineParams
from sgs2.scene import Scene
from sgs2.trainers.simple_trainer import SimpleTrainer
from typing import Callable, List

from utils.loss_utils import l1_loss, ssim

@dataclass
class GridTrainer:

    # SimpleTrainerParams; in the future when migrating to 3.11, we can inherit from SimpleTrainerParams
    white_background: bool = False
    iterations: int = 3000
    lambda_dssim: float = 0.2
    densification_interval: int = 100
    opacity_reset_interval: int = 3000
    densify_from_iter: int = 500
    densify_until_iter: int = 15_000
    densify_grad_threshold: float = 0.0002
    random_background: bool = False
    iteration_callback: Callable[[int, int, int], None] = None
    pipeline_params: PipelineParams = None
    network_gui: bool = True
    clean_chunk_edges: bool = True

    chunk_side_length: float = 1000
    chunk_loss_masking: bool = True# Color points within the grid cell white
    draft_iterations: int = 1000
    visibility_threshold: float = 0.05

    # Interior iteration callback passed to SimpleTrainer
    def _iteration_callback(self, iteration, num_gaussians, memory):
        self.last_recorded_iteration = iteration
        self.num_gaussians_per_model[self.active_model] = num_gaussians
        total_num_gaussians = sum(self.num_gaussians_per_model)
        if self.iteration_callback:
            self.iteration_callback(int((iteration + self.iteration_offset) / self.num_models), total_num_gaussians, memory)
    
    def __post_init__(self):
        # Pass in all dataclass parameters to SimpleTrainer dynamically
        # Create new __dict__ that replaces iteration_callback with our own
        new_dict = self.__dict__.copy()
        del new_dict["iteration_callback"]
        # Remove chunk_side_length, chunk_loss_masking, draft_iterations
        del new_dict["chunk_side_length"]
        del new_dict["chunk_loss_masking"]
        del new_dict["draft_iterations"]
        del new_dict["visibility_threshold"]
        self.simple_trainer = SimpleTrainer(**new_dict)
        self.iteration_offset = 0
        self.last_recorded_iteration = 0
        self.num_models = 1
        self.num_gaussians_per_model = []
        self.active_model = 0
        if self.pipeline_params is None:
            self.pipeline_params = PipelineParams()

    # For progress reporting only
    def record_offset(self):
        self.iteration_offset += self.last_recorded_iteration

    def train(self, scene: Scene, gaussian_model: GaussianModel):

        # Stage 1: Pre-train gaussians
        print("Pre-training gaussians...")
        self.simple_trainer.iterations = self.draft_iterations
        self.simple_trainer.train(scene, gaussian_model)
        self.simple_trainer.iteration_callback = self._iteration_callback

        # Stage 2: Compute visibility mask and split gaussians
        with torch.no_grad():
            occupied_grids = gaussian_model.calculate_occupied_grids(self.chunk_side_length)
            cameras = scene.get_train_cameras()
            visibility = {}
            visibility_gaussian = gaussian_model.clone()
            print("Computing visibility...")
            for i_grid, (min, max) in enumerate(tqdm(occupied_grids)):
                visibility[i_grid] = {}
                # Color according to grid.
                visibility_gaussian.color_segment_cell(min, max)
                # Render for each camera
                for camera in cameras:
                    # TODO: Frustum culling to early exit
                    render_results = render(camera, visibility_gaussian, self.pipeline_params)
                    # Average of RGB channels
                    render_intensity = torch.mean(render_results["render"], dim=0)
                    # Calculate mean of whole image (render_results["render"]) and average intensity
                    percent_visible = torch.mean(render_results["render"]).item()
                    visibility[i_grid][camera.image_name] = {
                        "percent_visible": percent_visible,
                        "visibility_mask": render_intensity
                    }

            split_gaussians = gaussian_model.split_with_grids(occupied_grids)

        final_models = []

        # Stage 3: Train split gaussians
        for i, (sub_gaussians, (min, max)) in enumerate(split_gaussians):
            print(f"Training submodel {i+1}/{len(split_gaussians)}...")
            model = self.train_submodel(scene, cameras, sub_gaussians, visibility[i])
            model.to_cpu()
            if self.clean_chunk_edges:
                model.cull_outside_box(min, max)
            final_models.append(model)

        # Stage 4: Combine gaussians
        print("Combining gaussians...")

        combined = GaussianModel(gaussian_model.sh_degree)
        combined.to_cpu() # Since the sub-models are on the CPU, we need to move the combined model to the CPU as well
        # combined.append_multiple(final_models)
        combined.to_gpu() # For inference, we need to move the combined model back to the GPU

        print("Done.")
        return combined

    def _create_bg(self):
        bg_color = [1, 1, 1] if self.white_background else [0, 0, 0]
        bg = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        return bg
    
    def train_submodel(self, scene: Scene, cameras: List[Camera], gaussian_model: GaussianModel, visibility: dict):
        torch.cuda.empty_cache()
        gaussian_model.to_gpu()
        # Let's first filter out cameras that do not meet the visibility threshold
        filtered_cameras = [camera for camera in cameras if visibility[camera.image_name]["percent_visible"] >= self.visibility_threshold]
        if len(filtered_cameras) == 0:
            print(f'No cameras meet the visibility threshold of {self.visibility_threshold}')
            return gaussian_model
        print(f'Filtered cameras from {len(cameras)} to {len(filtered_cameras)}')
        # Now, we can train the model
        bg = self._create_bg()
        viewpoint_stack = None
        for iteration in tqdm(range(1, self.iterations + 1)):
            if self.network_gui:
                self.update_network_viewer(scene, gaussian_model, bg, iteration)
            torch.cuda.empty_cache()
            gaussian_model.update_learning_rate(iteration)
            if iteration % 1000 == 0:
                gaussian_model.oneupSHdegree()
            if not viewpoint_stack:
               viewpoint_stack = filtered_cameras.copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
            # Now we render the image
            render_pkg = render(viewpoint_cam, gaussian_model, self.pipeline_params, bg)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            ground_truth = viewpoint_cam.original_image.cuda()
            # Let's obtain the mask
            mask = visibility[viewpoint_cam.image_name]["visibility_mask"]
            # print the dimensions of the mask and raise error to debug
            l1_val = l1_loss(image, ground_truth, mask=mask)
            ssim_val = ssim(image, ground_truth, mask=mask)
            loss = (1.0 - self.lambda_dssim) * l1_val + self.lambda_dssim * (1.0 - ssim_val)
            loss.backward()

            # Densification. Pretty much the same from this point on
            with torch.no_grad():
                # Densification
                if iteration < self.densify_until_iter:
                    # Keep track of max radii in image-space for pruning
                    gaussian_model.max_radii2D[visibility_filter] = torch.max(gaussian_model.max_radii2D[visibility_filter], radii[visibility_filter]) # Update max radii
                    gaussian_model.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if iteration > self.densify_from_iter and iteration % self.densification_interval == 0: # Densify every densification_interval iterations
                        size_threshold = 20 if iteration > self.opacity_reset_interval else None
                        gaussian_model.densify_and_prune(self.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                    
                    if iteration % self.opacity_reset_interval == 0 or (self.white_background and iteration == self.densify_from_iter):
                        gaussian_model.reset_opacity()

                # Optimizer step
                if iteration < self.iterations:
                    gaussian_model.optimizer.step() # Optimizer step
                    gaussian_model.optimizer.zero_grad(set_to_none = True)

            if self.iteration_callback:
                self.iteration_callback(iteration, gaussian_model._xyz.shape[0], torch.cuda.memory_allocated() / 1024 / 1024)

            torch.cuda.empty_cache()

        return gaussian_model
            
    def update_network_viewer(self, scene: Scene, gaussian_model, bg, iteration):
        if network_gui.conn == None: # If no GUI connection, we just train
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, self.pipeline_params.convert_SHs_python, self.pipeline_params.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussian_model, self.pipeline_params, bg, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, scene.source_path)
                if do_training and ((iteration < int(self.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None