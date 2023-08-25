import unip
from unip.core.pruner import BasePruner
from unip.utils.evaluation import *
from models.model_template import ExampleModel

import einops


def test_inference():
    # Your inference code here
    example_input = torch.randn(1, 3, 4, 4)
    model = ExampleModel()
    out = model(example_input)


def test_pruning():
    # Your pruning code here
    example_input = torch.randn(1, 3, 4, 4, requires_grad=True)
    model = ExampleModel()
    pruner = BasePruner(model, example_input, "RandomRatio")
    pruner.algorithm.run(0.6)
    pruner.prune()
    cal_flops(model, example_input, "cpu")


if __name__ == "__main__":
    test_inference()
    test_pruning()
