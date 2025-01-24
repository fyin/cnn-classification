from lenet.utils import get_project_root_dir, get_tb_writer_path, get_dataset_download_dir, get_model_checkpoint_dir, \
    get_model_checkpoint


def test_get_project_root_dir():
    assert str(get_project_root_dir()).endswith("cnn-classification")

def test_get_tb_writer_path():
    config = {"model_folder": "runs", "tensorboard_log_dir": "lenet_tensorboard_logs"}
    tb_writer_path = get_tb_writer_path(config)
    assert str(tb_writer_path).endswith("runs/lenet_tensorboard_logs")

def test_get_dataset_download_dir():
    config = {"data_download_dir": "datasets"}
    dataset_download_dir = get_dataset_download_dir(config)
    assert str(dataset_download_dir).endswith("cnn-classification/datasets")

def test_get_model_checkpoint_dir():
    config = {"model_folder": "runs", "model_checkpoint_dir": "lenet_checkpoints"}
    model_checkpoint_dir = get_model_checkpoint_dir(config)
    assert str(model_checkpoint_dir).endswith("cnn-classification/runs/lenet_checkpoints")

def test_get_model_checkpoint():
    epoch = 1
    config = {"model_folder": "runs", "model_checkpoint_dir": "lenet_checkpoints", "model_basename": "cnn_lenet_"}
    model_checkpoint_path = get_model_checkpoint(config, epoch)
    assert str(model_checkpoint_path).endswith("cnn-classification/runs/lenet_checkpoints/cnn_lenet_1.pt")
