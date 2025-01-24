def get_config():
    return {
        "batch_size": 32,
        "num_epochs": 10,
        "num_dataloader_workers": 2,
        "learning_rate": 1e-3,
        "model_folder": "runs",
        "model_checkpoint_dir": "lenet_checkpoints",
        "model_basename": "cnn_lenet_",
        "preload": "latest",
        "tensorboard_log_dir": "lenet_tensorboard_logs",
        "data_download_dir": "datasets"
    }