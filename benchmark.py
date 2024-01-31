from sgs2.gaussian import GaussianModel
from sgs2.scene import Scene
from sgs2.trainers.simple_trainer import SimpleTrainer

scene = Scene("./datasets/train")

gaussian_model = GaussianModel.from_scene(scene)

simple_trainer = SimpleTrainer()

scene, gaussian_model = simple_trainer.train(scene, gaussian_model)