from random import randint
import torch
from gaussian_renderer import render
from scene.gaussian_model import GaussianModel
from split_gaussian_splatting.scene import Scene
from split_gaussian_splatting.trainers.base_trainer import BaseTrainer
from split_gaussian_splatting.trainers.simple_trainer import SimpleTrainer
from split_gaussian_splatting.training_task import Task
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

    def train(self, task: Task, scene: Scene = None, gaussian_model: GaussianModel = None):

        print("Loading scene...")

        if not scene:
            scene = task.load_scene()

        print("Creating gaussian model...")

        if not gaussian_model:
            gaussian_model = scene.create_gaussians()

        print("Splitting gaussian model...")

        split_gaussians: List[GaussianModel] = gaussian_model.split_to_grid(100)
        gaussian_model.archive_to_cpu()

        print(f"Split into {len(split_gaussians)} gaussians.")
        self.num_models = len(split_gaussians)

        trained_split_gaussians = []

        print("Training split gaussians...")

        self.num_gaussians_per_model = [len(gaussians) for gaussians in split_gaussians]

        for i, gaussians in enumerate(split_gaussians):
            torch.cuda.empty_cache()
            gaussians.unarchive_to_cuda(task)
            self.active_model = i
            gaussians.training_setup(task)
            _, trained_gaussians = self.simple_trainer.train(task, scene, gaussians)
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
