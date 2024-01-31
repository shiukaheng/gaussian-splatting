from sgs2.gaussian import GaussianModel
from sgs2.scene import Scene
from sgs2.trainers.simple_trainer import SimpleTrainer
from sgs2.evaluation import evaluate_scene

scene = Scene("./datasets/train")

gaussian_model = GaussianModel.from_scene(scene)

simple_trainer = SimpleTrainer(iterations=100)

scene, gaussian_model = simple_trainer.train(scene, gaussian_model)

results = evaluate_scene(scene, gaussian_model)

print(results)