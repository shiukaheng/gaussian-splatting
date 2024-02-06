def benchmark(trainers, testsets):
    pass

# Features

# We perform training on all trainer-testset pairs

# We need to measure
# - Time taken for training
# - Memory usage graph
# - Reconstruction metrics: PSNR, SSIM, LPIPS

# We need trainers and testsets be hashable, and we can subsequently hash these combinations and cache the results

# We need to use a consistent folder structure for storing the results