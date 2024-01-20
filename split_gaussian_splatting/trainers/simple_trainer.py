from random import randint
import torch
from gaussian_renderer import render
from scene.gaussian_model import GaussianModel
from split_gaussian_splatting.scene import Scene
from split_gaussian_splatting.trainers.base_trainer import BaseTrainer
from split_gaussian_splatting.training_task import Task
from utils.loss_utils import l1_loss, ssim
from tqdm import tqdm
from typing import Callable

# Most basic trainer that wraps the original implementation to implement the base signature.
class SimpleTrainer(BaseTrainer):
    
    def __init__(self, iteration_callback: Callable[[int, int, int], None] = None):
        super().__init__(iteration_callback)

    def train(self, task: Task, scene: Scene = None, gaussian_model: GaussianModel = None):

        if not scene:
            scene = task.load_scene()

        if not gaussian_model:
            gaussian_model = scene.create_gaussians()

        bg = self.create_bg(task)

        viewpoint_stack = None

        for iteration in tqdm(range(1, task.iterations + 1)):

            torch.cuda.empty_cache()

            gaussian_model.update_learning_rate(iteration)

            if iteration % 1000 == 0:
                gaussian_model.oneupSHdegree()

            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy() # Get training cameras
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1)) # Pick a random camera

            render_pkg = render(viewpoint_cam, gaussian_model, task, bg)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

            ground_truth = viewpoint_cam.original_image.cuda()
            l1 = l1_loss(image, ground_truth)
            loss = (1.0 - task.lambda_dssim) * l1 + task.lambda_dssim * (1.0 - ssim(image, ground_truth)) # SSIM loss (per pixel)
            loss.backward() # Backpropagate loss

            with torch.no_grad():

                # Densification
                if iteration < task.densify_until_iter:
                    # Keep track of max radii in image-space for pruning
                    gaussian_model.max_radii2D[visibility_filter] = torch.max(gaussian_model.max_radii2D[visibility_filter], radii[visibility_filter]) # Update max radii
                    gaussian_model.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if iteration > task.densify_from_iter and iteration % task.densification_interval == 0: # Densify every densification_interval iterations
                        size_threshold = 20 if iteration > task.opacity_reset_interval else None
                        gaussian_model.densify_and_prune(task.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                    
                    if iteration % task.opacity_reset_interval == 0 or (task.white_background and iteration == task.densify_from_iter):
                        gaussian_model.reset_opacity()

                # Optimizer step
                if iteration < task.iterations:
                    gaussian_model.optimizer.step() # Optimizer step
                    gaussian_model.optimizer.zero_grad(set_to_none = True)

            if self.iteration_callback:
                self.iteration_callback(iteration, gaussian_model._xyz.shape[0], torch.cuda.memory_allocated() / 1024 / 1024)

            torch.cuda.empty_cache()

        return scene, gaussian_model

    def create_bg(self, task):
        bg_color = [1, 1, 1] if task.white_background else [0, 0, 0]
        bg = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        return bg
