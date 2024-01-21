import torch
from torch import nn

# Profile CUDA memory usage
print(f'Allocated: {torch.cuda.memory_allocated(device=None)}') # 0

# Create a tensor of size 1000 on CUDA
a = nn.Parameter(torch.empty(1000).cuda())

# Profile CUDA memory usage
print(f'Allocated: {torch.cuda.memory_allocated(device=None)}') # 4096

# Move to CPU
b = a.detach().cpu()
del a
torch.cuda.empty_cache()

# Profile CUDA memory usage
print(f'Allocated: {torch.cuda.memory_allocated(device=None)}') # 4096