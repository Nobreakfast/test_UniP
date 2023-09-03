# test_UniP
This repository aimed to provide the test models and test the pytorch pruning framework [UniP](https://github.com/Nobreakfast/UniP.git). 

## Introduction
This repository is based on the [UniP](https://github.com/Nobreakfast/Unip), and help users of UniP to test the pruning framework. If you could not run the UniP with your own models, you should pull a request to this repository. Let's fix your problems and make UniP better.

## Requirements
- `torch`
- `torchvision`
- [`UniP`](https://github.com/Nobreakfast/UniP)

## Getting Started
### 1. Install UniP
``` bash
git clone https://github.com/Nobreakfast/test_UniP.git
cd test_UniP
pip install -e .
```

### 2. Folder structure:
```
test_UniP
├── test_unip
│   ├── models
│   │   ├── template_model.py
│   │   ├── example_model.py
│   │   └── ...
├── tests
│   ├── test_template_model.py
│   ├── test_example_model.py
│   └── ...
├── requirements
│   ├── template_requirements.txt
│   ├── example_requirements.txt
│   └── ...
├── requirements.txt
└── README.md
```

### 3. Steps to add your model and test script:
1. `Fork` this repository to your github account.

2. Clone your `forked repository` to your local machine.
``` bash
cd /path/to/your/disired/directory
git clone https://github.com/your_github_account/test_UniP.git
```

3. :fire: ***`Important`*** :fire: Put your model in the `models` directory, you could create any nick named `folder` or `file`. You could refer to the template file [`test_unip/models/template_model.py`](./test_unip/models/template_model.py)

4. :fire: ***`Important`*** :fire: Add a simple `test script` in the `test` directory, you could create any nick named `file` (`tests/test_your_file.py`). You could refer to the template file [`tests/test_template.py`](./tests/test_template.py). The test script should provide two functions: `test_inference()` and UniP `test_pruning()`:
``` python
import unip
from unip.core.pruner import BasePruner
from unip.utils.evaluation import cal_flops
from test_unip.models.example_model import ExampleModel

def test_inference():
    # Your inference code here
    example_input = torch.randn(1, 3, 224, 224)
    model = ExampleModel()
    out = model(example_input)
def test_pruning():
    # Your pruning code here
    example_input = torch.randn(1, 3, 224, 224)
    model = ExampleModel()
    pruner = BasePruner(model, example_input, "RandomRatio")
    pruner.algorithm.run(0.8)
    pruner.prune()
    cal_flops(model, example_input, "cpu")
```
Make sure the `test_inference()` could run successfully. If your model need some external library, you should add the `requirements/example_requirements.txt` file to provide the requirements. You could refer to the template file [`requirements/template_requirements.txt`](./requirements/template_requirements.txt).

5. Commit your changes and push to your forked repository.
``` bash
git add .
git commit -m "add new model"
git push
```

6. Open your forked respository on github, and pull a request to this repository.

7. Wait for the test result. If the `test_inference()` function could run successfully, we will accept your request. If not, we will leave a comment on your pull request and you should fix the problem and pull a request again.

8. Open the [`UniP Issue Page`](https://github.com/Nobreakfast/UniP/issues) and `create a new issue`. Describe your problem and provide the link of your `PR ID and Link`. We will fix your problem as soon as possible.

9. Once your problem is fixed, we will close your issue and your `tests/test_your_file.py` will be changed to `tests/your_file_PRID.py`. If you still facing problems, you could rechange the file name, pull a request and reopen the issues. 

## Changelogs
- 2023-09-03: Optimize the test process.