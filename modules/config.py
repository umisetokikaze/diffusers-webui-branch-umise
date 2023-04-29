import argparse
import os

import toml

DEFAULT_CONFIG = {
    "version": "0",
    "images/txt2img/save_dir": "outputs/txt2img",
    "images/txt2img/save_name": "{index}-{seed}-{prompt}.png",
    "images/img2img/save_dir": "outputs/img2img",
    "images/img2img/save_name": "{index}-{seed}-{prompt}.png",
    "model_dir": "models",
    "models": [{"model_id": "runwayml/stable-diffusion-v1-5"}],
    "model": "runwayml/stable-diffusion-v1-5",
    "mode": "diffusers",
}

ROOT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


parser = argparse.ArgumentParser()

parser.add_argument("--config-file", type=str, default="config.toml")

# Network options
parser.add_argument("--host", type=str, default="")
parser.add_argument("--port", type=int, default=7860)
parser.add_argument("--share", action="store_true")

# Model options
parser.add_argument("--model-dir", type=str, default="models")
parser.add_argument("--hf-token", type=str)

# Optimization options
parser.add_argument("--xformers", action="store_true")

cmd_opts, _ = parser.parse_known_args(
    os.environ["COMMANDLINE_ARGS"].split(" ")
    if "COMMANDLINE_ARGS" in os.environ
    else ""
)
cmd_opts_dict = vars(cmd_opts)

opts = {}


def get_config():
    if not os.path.exists(cmd_opts.config_file):
        with open(cmd_opts.config_file, "w") as f:
            f.write(toml.dumps(DEFAULT_CONFIG))

    with open(cmd_opts.config_file, mode="r") as f:
        txt = f.read()

    try:
        config = toml.loads(txt)
    except:
        config = DEFAULT_CONFIG
    return config


def save_config():
    with open(cmd_opts.config_file, mode="w") as f:
        f.write(toml.dumps(opts))


def set(key: str, value: str):
    opts[key] = value
    save_config()


def get(key: str):
    if key in cmd_opts_dict and cmd_opts_dict[key] is not None:
        return cmd_opts_dict[key]
    config = get_config()
    return (
        config[key]
        if key in config
        else (DEFAULT_CONFIG[key] if key in DEFAULT_CONFIG else None)
    )


def init():
    global opts
    if not os.path.exists(cmd_opts.config_file):
        opts = DEFAULT_CONFIG
        save_config()
    else:
        opts = get_config()
