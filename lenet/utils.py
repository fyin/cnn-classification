import importlib.resources as pkg_resources
from pathlib import Path

import torch
import yaml

from lenet import config


def get_device():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_built() or torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)
    if (device == "cuda"):
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
    elif (device == 'mps'):
        print(f"Device name: <mps>")
    else:
        print(
            "NOTE: If you have a GPU/MPS, consider using it for training. Go to https://pytorch.org/get-started/locally/ for instructions.")
        print(
            "      On a Windows machine, Ex, run: conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia")
        print(
            "      On a Mac machine, Ex, run: conda install pytorch::pytorch torchvision torchaudio -c pytorch")

    device = torch.device(device)
    return device

def load_yaml_config(filename):
    with pkg_resources.files(config).joinpath(filename).open("r") as f:
        return yaml.safe_load(f)

def get_dataset_download_dir():
    config_yaml = load_yaml_config("common_config.yaml")
    return config_yaml["data"]["download_dir"]

def get_project_root_dir():
    return Path(__file__).resolve().parent.parent