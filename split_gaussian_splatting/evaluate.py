from trainers.simple_trainer import *
from trainers.random_trainer import *

# Define source dataset path
# For the ease of development, our ground truth dataset could actually bit fit into memory
# However, to fully demonstrate the power of our method we will need to integrate with a out-of-core viewer

sources = []
performance_comparison = []

# Reconstruct using vanilla and split method
print(f'Testing on {len(sources)} scenes...')

for source in sources:

    s = SimpleTrainer()
    print(f'Training on {source}...')
    s.train(source)
    print(f'Evaluating on {source}...')
    baseline_performance = s.evaluate()

    # r = RandomTrainer()
    # print(f'Training on {source}...')
    # r.train(source)
    # print(f'Evaluating on {source}...')
    # our_performance = r.evaluate()

    performance_comparison.append({
        "baseline": baseline_performance,
        # "ours": our_performance
    })