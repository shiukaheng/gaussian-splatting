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
from typing import Callable

# Most basic trainer that wraps the original implementation to implement the base signature.
class GridTrainer(BaseTrainer):
    
    def __init__(self, iteration_callback: Callable[[int, int, int], None] = None):
        super().__init__(iteration_callback)
        self._iteration_callback = iteration_callback
        self.simple_trainer = SimpleTrainer(self.iteration_callback)
        self.iteration_offset = 0
        self.last_recorded_iteration = 0
        self.num_models = 1
        
    def iteration_callback(self, iteration, num_gaussians, memory):
        self.last_recorded_iteration = iteration
        if self._iteration_callback:
            self._iteration_callback(int((iteration + self.iteration_offset) / self.num_models), num_gaussians, memory)

    def record_offset(self):
        self.iteration_offset = self.last_recorded_iteration

    def train(self, task: Task, scene: Scene = None, gaussian_model: GaussianModel = None):

        print("Loading scene...")

        if not scene:
            scene = task.load_scene()

        print("Creating gaussian model...")

        if not gaussian_model:
            gaussian_model = scene.create_gaussians()

        print("Splitting gaussian model...")

        split_gaussians = gaussian_model.split_to_grid(100)

        print(f"Split into {len(split_gaussians)} gaussians.")
        self.num_models = len(split_gaussians)

        trained_split_gaussians = []

        print("Training split gaussians...")

        for gaussians in tqdm(split_gaussians):

            gaussians.training_setup(task)
            _, trained_gaussians = self.simple_trainer.train(task, scene, gaussians)
            trained_split_gaussians.append(trained_gaussians)

            self.record_offset()

        print("Combining gaussians...")

        combined = GaussianModel(task.sh_degree)
        combined.append_multiple(trained_split_gaussians)

        print("Done.")

        return scene, combined
