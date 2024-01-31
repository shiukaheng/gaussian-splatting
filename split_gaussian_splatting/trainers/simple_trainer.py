from random import randint
import torch
from gaussian_renderer import network_gui, render
from scene.gaussian_model import GaussianModel
from split_gaussian_splatting.scene import Scene
from split_gaussian_splatting.trainers.base_trainer import BaseTrainer
from split_gaussian_splatting.training_task import SimpleTrainerParams
from utils.loss_utils import l1_loss, ssim
from tqdm import tqdm
from typing import Callable

# Most basic trainer that wraps the original implementation to implement the base signature.
class SimpleTrainer(BaseTrainer):
    
    def __init__(self, iteration_callback: Callable[[int, int, int], None] = None):
        print(iteration_callback)
        super().__init__(iteration_callback)

    def train(self, train_params: SimpleTrainerParams, scene: Scene = None, gaussian_model: GaussianModel = None):

        if not scene:
            scene = train_params.load_scene()

        if not gaussian_model:
            gaussian_model = scene.create_gaussians()

        bg = self.create_bg(train_params)

        viewpoint_stack = None

        for iteration in range(1, train_params.iterations + 1):

            self.update_network_viewer(train_params, gaussian_model, bg, iteration)

            torch.cuda.empty_cache()

            gaussian_model.update_learning_rate(iteration)

            if iteration % 1000 == 0:
                gaussian_model.oneupSHdegree()

            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy() # Get training cameras
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1)) # Pick a random camera

            render_pkg = render(viewpoint_cam, gaussian_model, train_params, bg)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

            ground_truth = viewpoint_cam.original_image.cuda()
            l1 = l1_loss(image, ground_truth)
            loss = (1.0 - train_params.lambda_dssim) * l1 + train_params.lambda_dssim * (1.0 - ssim(image, ground_truth)) # SSIM loss (per pixel)
            loss.backward() # Backpropagate loss

            with torch.no_grad():

                # Densification
                if iteration < train_params.densify_until_iter:
                    # Keep track of max radii in image-space for pruning
                    gaussian_model.max_radii2D[visibility_filter] = torch.max(gaussian_model.max_radii2D[visibility_filter], radii[visibility_filter]) # Update max radii
                    gaussian_model.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if iteration > train_params.densify_from_iter and iteration % train_params.densification_interval == 0: # Densify every densification_interval iterations
                        size_threshold = 20 if iteration > train_params.opacity_reset_interval else None
                        gaussian_model.densify_and_prune(train_params.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                    
                    if iteration % train_params.opacity_reset_interval == 0 or (train_params.white_background and iteration == train_params.densify_from_iter):
                        gaussian_model.reset_opacity()

                # Optimizer step
                if iteration < train_params.iterations:
                    gaussian_model.optimizer.step() # Optimizer step
                    gaussian_model.optimizer.zero_grad(set_to_none = True)

            if self.iteration_callback:
                self.iteration_callback(iteration, gaussian_model._xyz.shape[0], torch.cuda.memory_allocated() / 1024 / 1024)

            torch.cuda.empty_cache()

        return scene, gaussian_model

    def update_network_viewer(self, task, gaussian_model, bg, iteration):
        if network_gui.conn == None: # If no GUI connection, we just train
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, task.convert_SHs_python, task.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussian_model, task, bg, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, task.source_path)
                if do_training and ((iteration < int(task.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

    def create_bg(self, task):
        bg_color = [1, 1, 1] if task.white_background else [0, 0, 0]
        bg = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        return bg
