import torch

# Profile CUDA memory usage
print(f'Initial without evaluators: {torch.cuda.memory_allocated(device=None)}')

from split_gaussian_splatting.training_task import Task

# Profile CUDA memory usage
print(f'Initial: {torch.cuda.memory_allocated(device=None)}')

task = Task(source_path="./datasets/train", iterations=200, data_device='cpu', densify_from_iter=0, densification_interval=10)

scene = task.load_scene()

# Profile CUDA memory usage
print(f'Scene loaded: {torch.cuda.memory_allocated(device=None)}')

gaussian_model = scene.create_gaussians()

# Profile CUDA memory usage
print(f'Gaussian model created: {torch.cuda.memory_allocated(device=None)}')

# Move to CPU
gaussian_model.to_cpu()

# Profile CUDA memory usage
print(f'Gaussian model moved to CPU: {torch.cuda.memory_allocated(device=None)}')

# Move to CUDA
gaussian_model.to_cuda()

# Profile CUDA memory usage
print(f'Gaussian model moved to CUDA: {torch.cuda.memory_allocated(device=None)}')
# Move to CPU
gaussian_model.to_cpu()

# Profile CUDA memory usage
print(f'Gaussian model moved to CPU: {torch.cuda.memory_allocated(device=None)}')

# Move to CUDA
gaussian_model.to_cuda()

# Profile CUDA memory usage
print(f'Gaussian model moved to CUDA: {torch.cuda.memory_allocated(device=None)}')