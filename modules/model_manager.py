from typing import *

from huggingface_hub import HfApi, ModelFilter

from modules.logger import logger

from . import config
from .model import StableDiffusionModel

ModelMode = Literal["diffusers", "tensorrt"]

runner = None
mode: ModelMode = config.get("mode")
sd_models: List[StableDiffusionModel] = []
sd_model: Optional[StableDiffusionModel] = None

raw_model_list = config.get("models") or []
if len(raw_model_list) < 1:
    raw_model_list = config.DEFAULT_CONFIG["models"]
for model_data in raw_model_list:
    sd_models.append(StableDiffusionModel(**model_data))


def set_mode(m: ModelMode):
    global mode
    mode = m
    runner.teardown()
    set_model(sd_model.model_id)


def get_model(model_id: str):
    model = [x for x in sd_models if x.model_id == model_id]
    if len(model) < 1:
        return None
    return model[0]


def add_model(model_id: str):
    global sd_models
    sd_models.append(StableDiffusionModel(model_id=model_id))
    config.set("models", [x.dict() for x in sd_models])


def set_model(model_id: str):
    global runner
    global sd_model
    sd_model = [x for x in sd_models if x.model_id == model_id]
    if len(sd_model) != 1:
        raise ValueError("Model not found or multiple models with same ID.")
    else:
        sd_model = sd_model[0]

    if runner is not None:
        runner.teardown()
        del runner

    logger.info(f"Loading {sd_model.model_id}...")
    if mode == "diffusers":
        from modules.runners.diffusers import DiffusersDiffusionRunner

        runner = DiffusersDiffusionRunner(sd_model)
    elif mode == "tensorrt":
        from .runners.tensorrt import TensorRTDiffusionRunner

        runner = TensorRTDiffusionRunner(sd_model)
    logger.info(f"Loaded {sd_model.model_id}...")

    config.set("model", sd_model.model_id)


def set_default_model():
    global sd_model
    prev = config.get("model")
    sd_model = [x for x in sd_models if x.model_id == prev]
    if len(sd_model) != 1:
        sd_model = sd_models[0]
    else:
        sd_model = sd_model[0]

    set_model(sd_model.model_id)


def search_model(model_id: str):
    api = HfApi()
    models = api.list_models(
        filter=ModelFilter(library="diffusers", model_name=model_id)
    )
    return models
