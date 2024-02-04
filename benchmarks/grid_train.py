# Hypothesis: If we have too many views, it will cause gradients be less consistent, and thus the network will be harder to train.

from gaussian_renderer import network_gui
from sgs2.gaussian import GaussianModel
from sgs2.scene import Scene
from sgs2.trainers.grid_trainer import GridTrainer
from sgs2.trainers.simple_trainer import SimpleTrainer
from sgs2.evaluation import evaluate_scene

from sgs2.tests.alleyds import subset

def train():
    network_gui.init("127.0.0.1", 6009)
    scene = Scene("/home/heng/Documents/GitHub/gaussian-splatting/datasets/alleyds")
    gaussian_model = GaussianModel.from_scene(scene)

    # Create trainer
    trainer = GridTrainer(network_gui=True)

    # Train models
    trainer.train(scene, gaussian_model)

# def evaluate():
#     # Load ply files
#     sub_gaussian_model = GaussianModel.from_ply("./alleyds_subset.ply")
#     full_gaussian_model = GaussianModel.from_ply("./alleyds_full.ply")

# def save():
#     full_scene = Scene("/home/heng/Documents/GitHub/gaussian-splatting/datasets/alleyds")
#     full_gaussian_model = GaussianModel.from_scene(full_scene)
#     full_gaussian_model.export_for_sibr(full_scene, "./output/alleyds_full_sibr/")