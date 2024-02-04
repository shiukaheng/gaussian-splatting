from dataclasses import dataclass
from random import randint
import torch
from tqdm import tqdm
from gaussian_renderer import network_gui, render
from scene.cameras import Camera
from sgs2.gaussian import GaussianModel
from sgs2.helpers import PipelineParams
from sgs2.scene import Scene
from utils.loss_utils import l1_loss, ssim
from typing import Callable, List

def train(
    scene: Scene,
    gaussian_model: GaussianModel,
    white_background: bool = False,
    iterations: int = 30_000,
    lambda_dssim: float = 0.2,
    densification_interval: int = 100,
    opacity_reset_interval: int = 3000,
    densify_from_iter: int = 500,
    densify_until_iter: int = 15_000,
    densify_grad_threshold: float = 0.0002,
    iteration_callback: Callable[[int, int, int], None] = None,
    pipeline_params: PipelineParams = None,
    network_gui_enabled: bool = False
):
    if pipeline_params is None:
        pipeline_params = PipelineParams()
    
    bg_color = [1, 1, 1] if white_background else [0, 0, 0]
    bg = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    viewpoint_stack = None

    for iteration in tqdm(range(1, iterations + 1)):
        if network_gui_enabled:
            update_network_viewer(scene, gaussian_model, bg, iteration, iterations, pipeline_params)

        torch.cuda.empty_cache()
        gaussian_model.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            gaussian_model.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = scene.get_train_cameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))  # Pick a random camera

        render_pkg = render(viewpoint_cam, gaussian_model, pipeline_params, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg.values()

        ground_truth = viewpoint_cam.original_image.cuda()
        l1 = l1_loss(image, ground_truth)
        loss = (1.0 - lambda_dssim) * l1 + lambda_dssim * (1.0 - ssim(image, ground_truth))
        loss.backward()

        with torch.no_grad():
            # Densification
            if iteration < densify_until_iter:
                gaussian_model.max_radii2D[visibility_filter] = torch.max(gaussian_model.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussian_model.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > densify_from_iter and iteration % densification_interval == 0:
                    size_threshold = 20 if iteration > opacity_reset_interval else None
                    gaussian_model.densify_and_prune(densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opacity_reset_interval == 0 or (white_background and iteration == densify_from_iter):
                    gaussian_model.reset_opacity()

            # Optimizer step
            if iteration < iterations:
                gaussian_model.optimizer.step()
                gaussian_model.optimizer.zero_grad(set_to_none=True)

        if iteration_callback:
            iteration_callback(iteration, gaussian_model._xyz.shape[0], torch.cuda.memory_allocated() / 1024 / 1024)

        torch.cuda.empty_cache()

    return scene, gaussian_model

def update_network_viewer(scene, gaussian_model, bg, iteration, total_iterations, pipeline_params):
    if network_gui.conn == None:
        network_gui.try_connect()
    while network_gui.conn != None:
        try:
            net_image_bytes = None
            custom_cam, do_training, pipeline_params.convert_SHs_python, pipeline_params.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
            if custom_cam != None:
                net_image = render(custom_cam, gaussian_model, pipeline_params, bg, scaling_modifer)["render"]
                net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
            network_gui.send(net_image_bytes, scene.source_path)
            if do_training and ((iteration < int(total_iterations)) or not keep_alive):
                break
        except Exception as e:
            network_gui.conn = None
