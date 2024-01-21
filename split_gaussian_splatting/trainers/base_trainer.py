from abc import ABC, abstractmethod
from typing import Callable
from scene.gaussian_model import GaussianModel
from split_gaussian_splatting.training_task import Task
from split_gaussian_splatting.scene import Scene

class BaseTrainer(ABC):
    def __init__(self, iteration_callback: Callable[[int, int, int], None] = None):
        """
        Initialize the BaseTrainer object.

        Args:
            iteration_callback (Callable[[int, int, int], None], optional): 
                A callback function that will be called after each iteration.
                The callback function should take three arguments: iteration (int),
                number of gaussians (int), and VRAM usage in MB (int). 
                Defaults to None.
        """
        self.iteration_callback = iteration_callback

    @abstractmethod
    def train(self, task: Task, scene: Scene = None, gaussian_model: GaussianModel = None) -> [Scene, GaussianModel]:
        """
        Train the model. This method should be implemented in subclasses.

        Parameters:
        task (Task): The task for training. Contains all configuration required to train the model.
        scene (Scene): The scene to be used for training. Contains camera parameters and images. If None, load from task parameters.
        gaussian_model (GaussianModel): The initial gaussian model. If None, gaussian_model will be created from the source.
        """
