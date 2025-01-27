from pathlib import Path

from lenet.utils import get_project_root_dir, load_yaml_config, get_dataset_download_dir


def test_load_yaml_config():
    config = load_yaml_config("common_config.yaml")
    assert isinstance(config, dict)
    assert config["model"]["name"] is not None

def test_get_project_root_dir():
    assert isinstance(get_project_root_dir(), Path)
    assert get_project_root_dir().name.endswith("cnn-classification")

def test_get_dataset_download_dir():
    assert get_dataset_download_dir() is not None