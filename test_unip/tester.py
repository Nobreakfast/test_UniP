import importlib
import test_unip.models as models


def get_model(model_file, model_name):
    model_file = "test_unip.models." + model_file
    model_file = importlib.import_module(model_file)
    model = getattr(model_file, model_name)
    return model
