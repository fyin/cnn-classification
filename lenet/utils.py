import os
from pathlib import Path

import torch


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

def get_tb_writer_path(config):
    return Path(get_project_root_dir()).joinpath(config["model_folder"]).joinpath(config["tensorboard_log_dir"])

def get_model_checkpoint_dir(config):
    # return Path(config["model_folder"]).joinpath(config["model_checkpoint_dir"])
    return Path(get_project_root_dir()).joinpath(config["model_folder"]).joinpath(config["model_checkpoint_dir"])

def get_model_basename(config):
   return get_model_checkpoint_dir(config).joinpath(config["model_basename"])

def get_latest_model_checkpoint(config):
    model_checkpoint_filenames = f"{config['model_basename']}*"
    weights_files = list(get_model_checkpoint_dir(config).glob(model_checkpoint_filenames))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return weights_files[-1]

def get_model_checkpoint(config, epoch):
    model_checkpoint_filename = f"{config['model_basename']}{epoch}.pt"
    return get_model_checkpoint_dir(config).joinpath(model_checkpoint_filename)

def get_project_root_dir():
    current_file = Path(__file__).resolve()
    # Get the project root directory
    project_dir = current_file.parent.parent
    print("Project directory:", project_dir)
    return project_dir

def get_dataset_download_dir(config):
    return Path(get_project_root_dir()).joinpath(config["data_download_dir"])