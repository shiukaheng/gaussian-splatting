from dataclasses import dataclass
import torch
from gaussian_renderer import render
from sgs2.gaussian import GaussianModel
from sgs2.helpers import PipelineParams
from sgs2.scene import Scene
from sgs2.trainers.simple_trainer import SimpleTrainer
from typing import Callable

@dataclass
class GridTrainer:

    # SimpleTrainerParams; in the future when migrating to 3.11, we can inherit from SimpleTrainerParams
    white_background: bool = False
    iterations: int = 30_000
    lambda_dssim: float = 0.2
    densification_interval: int = 100
    opacity_reset_interval: int = 3000
    densify_from_iter: int = 500
    densify_until_iter: int = 15_000
    densify_grad_threshold: float = 0.0002
    random_background: bool = False
    iteration_callback: Callable[[int, int, int], None] = None
    pipeline_params: PipelineParams = None
    network_gui: bool = False

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
        new_dict["iteration_callback"] = self._iteration_callback
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

        # Pre-train gaussians without densification
        # print("Pre-training gaussians...")
        # TODO: raise NotImplementedError("Pre-training not implemented.")

        print("Splitting gaussian model...")
        split_gaussians = gaussian_model.split_to_grid(100000)
        gaussian_model.to_cpu()

        print(f"Split into {len(split_gaussians)} gaussians.")
        self.num_models = len(split_gaussians)
        trained_split_gaussians = []

        print("Training split gaussians...")
        self.num_gaussians_per_model = [len(gaussians[0]) for gaussians in split_gaussians]

        gaussian_visibility = {} # dict: gaussian_model -> camera -> number of visible gaussians

        all_train_cameras = scene.get_train_cameras()

        # Precompute visibility of each gaussian per camera. Bad complexity, but we can optimize later.
        print("Precomputing visibility...")
        with torch.no_grad():
            for i_gaussian, (gaussians, (model_min, model_max)) in enumerate(split_gaussians):
                gaussian_visibility[i_gaussian] = {}
                torch.cuda.empty_cache()
                gaussians.to_gpu()
                for i_camera, camera in enumerate(all_train_cameras):
                    # TODO: Early exit if grid cell is not visible
                    render_result = render(camera, gaussians, self.pipeline_params)
                    # Count visible gaussians
                    gaussian_visibility[i_gaussian][i_camera] = torch.sum(render_result["visibility_filter"]).item()
                gaussians.to_cpu()
                self.record_offset()
                torch.cuda.empty_cache()

        min_points = 50

        print("Training gaussians...")
        for i, (gaussians, (model_min, model_max)) in enumerate(split_gaussians):

            torch.cuda.empty_cache()
            gaussians.to_gpu()
            self.active_model = i
            # Filter cameras by visibility
            cameras = [camera for i_camera, camera in enumerate(all_train_cameras) if gaussian_visibility[i][i_camera] >= min_points]
            print(f'Filtered cameras from {len(all_train_cameras)} to {len(cameras)}')
            if len(cameras) == 0:
                print("No cameras visible, skipping...")
                continue
            _, trained_gaussians = self.simple_trainer.train(scene, gaussians, cameras)
            trained_gaussians.cull_outside_box(model_min, model_max) # Cull gaussians outside of grid cell
            # Print memory usage
            trained_gaussians.to_cpu()
            trained_split_gaussians.append(trained_gaussians)
            self.record_offset()
            torch.cuda.empty_cache()

        print("Combining gaussians...")

        combined = GaussianModel(gaussian_model.sh_degree)
        combined.to_cpu() # Since the sub-models are on the CPU, we need to move the combined model to the CPU as well
        combined.append_multiple(trained_split_gaussians)
        combined.to_gpu() # For inference, we need to move the combined model back to the GPU

        print("Done.")

        return scene, combined

    def _create_bg(self):
        bg_color = [1, 1, 1] if self.white_background else [0, 0, 0]
        bg = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        return bg
