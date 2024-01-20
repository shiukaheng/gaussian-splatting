from argparse import ArgumentParser, Namespace
import os
from typing import Any
import uuid

from attr import asdict, dataclass
from arguments import GroupParams, ModelParams, OptimizationParams, PipelineParams
from split_gaussian_splatting.scene import Scene
from scene.gaussian_model import GaussianModel

@dataclass
class Task:
    # Model parameters
    sh_degree: int = 3
    source_path: str = ""
    model_path: str = ""
    images: str = "images"
    resolution: int = -1
    white_background: bool = False
    data_device: str = "cuda"
    eval: bool = False

    # Optimization parameters
    iterations: int = 30_000
    position_lr_init: float = 0.00016
    position_lr_final: float = 0.0000016
    position_lr_delay_mult: float = 0.01
    position_lr_max_steps: int = 30_000
    feature_lr: float = 0.0025
    opacity_lr: float = 0.05
    scaling_lr: float = 0.005
    rotation_lr: float = 0.001
    percent_dense: float = 0.01
    lambda_dssim: float = 0.2
    densification_interval: int = 100
    opacity_reset_interval: int = 3000
    densify_from_iter: int = 500
    densify_until_iter: int = 15_000
    densify_grad_threshold: float = 0.0002
    random_background: bool = False

    # Pipeline parameters
    convert_SHs_python: bool = False
    compute_cov3D_python: bool = False
    debug: bool = False

    def load_scene(self, on_load_progress=None):
        scene = Scene(self, on_load_progress=on_load_progress)
        return scene

    def export_training_namespace(self):
        # Selectively create a dictionary of only model parameters
        model_params = {
            'sh_degree': self.sh_degree,
            'source_path': self.source_path,
            'model_path': self.model_path,
            'images': self.images,
            'resolution': self.resolution,
            'white_background': self.white_background,
            'data_device': self.data_device,
            'eval': self.eval
        }
        
        # Convert the dictionary to a Namespace object
        namespace = Namespace(**model_params)
        
        return namespace
    
    def create_output_folder(self) -> str:
        if not self.model_path:
            if os.getenv('OAR_JOB_ID'):
                unique_str=os.getenv('OAR_JOB_ID')
            else:
                unique_str = str(uuid.uuid4())
            self.model_path = os.path.join("./output/", unique_str[0:10])
            
        # Set up output folder
        print("Output folder: {}".format(self.model_path))
        os.makedirs(self.model_path, exist_ok = True)
        with open(os.path.join(self.model_path, "cfg_args"), 'w') as cfg_log_f:
            cfg_log_f.write(str(self.export_training_namespace()))
        return self.model_path
    
# @dataclass
# class TaskModel:
#     task: Task
#     model: GaussianModel