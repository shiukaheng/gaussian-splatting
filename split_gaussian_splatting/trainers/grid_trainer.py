from random import randint
import torch
from gaussian_renderer import network_gui, render
from scene.cameras import Camera
from scene.gaussian_model import GaussianModel
from split_gaussian_splatting.scene import Scene
from split_gaussian_splatting.trainers.base_trainer import BaseTrainer
from split_gaussian_splatting.trainers.simple_trainer import SimpleTrainer
from split_gaussian_splatting.training_task import SimpleTrainerParams
from utils.loss_utils import l1_loss, ssim
from tqdm import tqdm
from typing import Callable, List

# Most basic trainer that wraps the original implementation to implement the base signature.
class GridTrainer(BaseTrainer):

    def iteration_callback(self, iteration, num_gaussians, memory):
        self.last_recorded_iteration = iteration
        self.num_gaussians_per_model[self.active_model] = num_gaussians
        total_num_gaussians = sum(self.num_gaussians_per_model)
        if self._iteration_callback:
            self._iteration_callback(int((iteration + self.iteration_offset) / self.num_models), total_num_gaussians, memory)
    
    def __init__(self, iteration_callback: Callable[[int, int, int], None] = None):

        super().__init__(self.iteration_callback)
        self._iteration_callback = iteration_callback
        self.simple_trainer = SimpleTrainer(self.iteration_callback)

        # Below is just all hacks to wrap simple trainer to properly report iterations and number of gaussians.
        self.iteration_offset = 0
        self.last_recorded_iteration = 0
        self.num_models = 1
        self.num_gaussians_per_model = []
        self.active_model = 0

    # For progress reporting only
    def record_offset(self):

        self.iteration_offset += self.last_recorded_iteration

    def train(self, task: SimpleTrainerParams, scene: Scene = None, gaussian_model: GaussianModel = None):

        print("Loading scene...")
        if not scene:
            scene = task.load_scene()

        print("Creating gaussian model...")
        if not gaussian_model:
            gaussian_model = scene.create_gaussians()

        # Pre-train gaussians without densification
        print("Pre-training gaussians...")
        # TODO: raise NotImplementedError("Pre-training not implemented.")

        print("Splitting gaussian model...")
        split_gaussians = gaussian_model.split_to_grid(100000)
        gaussian_model.archive_to_cpu()

        print(f"Split into {len(split_gaussians)} gaussians.")
        self.num_models = len(split_gaussians)
        trained_split_gaussians = []

        print("Training split gaussians...")
        self.num_gaussians_per_model = [len(gaussians[0]) for gaussians in split_gaussians]

        gaussian_visibility = {} # dict: gaussian_model -> camera -> number of visible gaussians

        all_train_cameras = scene.getTrainCameras()

        # Precompute visibility of each gaussian per camera. Bad complexity, but we can optimize later.
        print("Precomputing visibility...")
        with torch.no_grad():
            for i_gaussian, (gaussians, (model_min, model_max)) in enumerate(split_gaussians):
                gaussian_visibility[i_gaussian] = {}
                torch.cuda.empty_cache()
                gaussians.unarchive_to_cuda(task)
                for i_camera, camera in enumerate(all_train_cameras):
                    # TODO: Early exit if grid cell is not visible
                    render_result = render(camera, gaussians, task)
                    # Count visible gaussians
                    gaussian_visibility[i_gaussian][i_camera] = torch.sum(render_result["visibility_filter"]).item()
                gaussians.archive_to_cpu()
                self.record_offset()
                torch.cuda.empty_cache()

        min_points = 50

        print("Training gaussians...")
        for i, (gaussians, (model_min, model_max)) in enumerate(split_gaussians):

            torch.cuda.empty_cache()
            gaussians.unarchive_to_cuda(task)
            self.active_model = i
            gaussians.training_setup(task)
            # Filter cameras by visibility
            cameras = [camera for i_camera, camera in enumerate(all_train_cameras) if gaussian_visibility[i][i_camera] >= min_points]
            print(f'Filtered cameras from {len(all_train_cameras)} to {len(cameras)}')
            if len(cameras) == 0:
                print("No cameras visible, skipping...")
                continue
            trained_gaussians = self.train_loop(task, scene, cameras, gaussians)
            trained_gaussians.cull_outside_box(model_min, model_max) # Cull gaussians outside of grid cell
            # Print memory usage
            trained_gaussians.archive_to_cpu()
            trained_split_gaussians.append(trained_gaussians)
            self.record_offset()
            torch.cuda.empty_cache()

        print("Combining gaussians...")

        combined = GaussianModel(task.sh_degree)
        combined.archive_to_cpu()
        combined.append_multiple(trained_split_gaussians)

        print("Done.")

        return scene, combined
    
    def train_loop(self, task: SimpleTrainerParams, scene: Scene, camera_selection: List[Camera], gaussian_model: GaussianModel):

        bg = self.create_bg(task)

        viewpoint_stack = None

        for iteration in range(1, task.iterations + 1):

            self.update_network_viewer(task, gaussian_model, bg, iteration)

            torch.cuda.empty_cache()

            gaussian_model.update_learning_rate(iteration)

            if iteration % 1000 == 0:
                gaussian_model.oneupSHdegree()

            if not viewpoint_stack:
                viewpoint_stack = camera_selection.copy() # Get training cameras
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

        return gaussian_model
    
    def update_network_viewer(self, task, gaussian_model, bg, iteration):
        if network_gui.conn == None: # If no GUI connection, we just train
            # print("Attempting to connect to GUI...")
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                # print("Sending image...")
                net_image_bytes = None
                custom_cam, do_training, task.convert_SHs_python, task.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    # print("Rendering image...")
                    net_image = render(custom_cam, gaussian_model, task, bg, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, task.source_path)
                if do_training and ((True) or not keep_alive):
                    # print("Breaking...")
                    break
            except Exception as e:
                # print("GUI connection failed.")
                network_gui.conn = None

    def create_bg(self, task):
        bg_color = [1, 1, 1] if task.white_background else [0, 0, 0]
        bg = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        return bg
