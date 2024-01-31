import torch
from gaussian_renderer import render

# Profile CUDA memory usage
print(f'Initial without evaluators: {torch.cuda.memory_allocated(device=None)}')

from split_gaussian_splatting.training_task import SimpleTrainerParams

# Profile CUDA memory usage
print(f'Initial: {torch.cuda.memory_allocated(device=None)}')

task = SimpleTrainerParams(source_path="./datasets/train", iterations=200, data_device='cpu', densify_from_iter=0, densification_interval=10)

scene = task.load_scene()

gaussian_model = scene.create_gaussians()

gaussian_model.archive_to_cpu()
gaussian_model.unarchive_to_cuda(task)

render(scene.getTrainCameras()[0], gaussian_model, task)