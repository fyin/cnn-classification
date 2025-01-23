from pathlib import Path

def get_config():
    return {
        "batch_size": 32,
        "num_epochs": 10,
        "num_dataloader_workers": 2,
        "learning_rate": 1e-3,
        "model_folder": "runs",
        "model_checkpoint_dir": "model_checkpoints",
        "model_basename": "cnn_lenet5_",
        "preload": "latest",
        "tensorboard_log_dir": "tensorboard_logs"
    }

def get_tb_writer_path(config):
    return Path(config["model_folder"]).joinpath(config["tensorboard_log_dir"])

def get_model_checkpoint_dir(config):
    return Path(config["model_folder"]).joinpath(config["model_checkpoint_dir"])

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

if __name__ == "__main__":
    config = get_config()
    weights_files = get_latest_model_checkpoint(config)
    print(f'weights_files={weights_files}')
    print(f'get_tb_writer_path(config)={get_tb_writer_path(config)}')
    print(f'get_model_checkpoint_dir(config)={get_model_checkpoint_dir(config)}')
    print(f'get_model_basename(config)={get_model_basename(config)}')
    print(f'get_model_checkpoint(config, 1)={get_model_checkpoint(config, 1)}')