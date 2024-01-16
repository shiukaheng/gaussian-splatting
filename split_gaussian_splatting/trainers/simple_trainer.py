from scene.gaussian_model import GaussianModel

# Most basic trainer that wraps the original implementation to implement the base signature.
class SimpleTrainer:
    
    def train(self, gaussians: GaussianModel, cameras):
