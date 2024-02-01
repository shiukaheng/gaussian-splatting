# Hypothesis: If we have too many views, it will cause gradients be less consistent, and thus the network will be harder to train.

from gaussian_renderer import network_gui
from sgs2.gaussian import GaussianModel
from sgs2.scene import Scene
from sgs2.trainers.grid_trainer import GridTrainer
from sgs2.trainers.simple_trainer import SimpleTrainer
from sgs2.evaluation import evaluate_scene

from sgs2.tests.alleyds import subset

def train():
    per_image_iters = 50

    network_gui.init("127.0.0.1", 6009)

    # Create scene and subscene to train on
    sub_scene = Scene("/home/heng/Documents/GitHub/gaussian-splatting/datasets/alleyds", camera_name_whitelist=subset)
    anti_sub_scene = Scene("/home/heng/Documents/GitHub/gaussian-splatting/datasets/alleyds", camera_name_blacklist=subset)
    full_scene = Scene("/home/heng/Documents/GitHub/gaussian-splatting/datasets/alleyds")

    # Create corresponding Gaussian models
    sub_gaussian_model = GaussianModel.from_scene(sub_scene)
    full_gaussian_model = GaussianModel.from_scene(full_scene)

    # Create trainer
    simple_trainer = SimpleTrainer(network_gui=True)

    # Train models
    simple_trainer.iterations = len(sub_scene) * per_image_iters
    sub_scene, sub_gaussian_model = simple_trainer.train(sub_scene, sub_gaussian_model)
    sub_gaussian_model.save_ply("./alleyds_subset.ply")

    simple_trainer.iterations = len(full_scene) * per_image_iters
    full_scene, sub_gaussian_model = simple_trainer.train(full_scene, full_gaussian_model)
    full_gaussian_model.save_ply("./alleyds_full.ply")

def evaluate():
    # Load ply files
    sub_gaussian_model = GaussianModel.from_ply("./alleyds_subset.ply")
    full_gaussian_model = GaussianModel.from_ply("./alleyds_full.ply")