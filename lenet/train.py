import multiprocessing
import os
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from ray import train
from lenet.dataset import get_dataloader
from lenet.model import LeNet
from tqdm import tqdm

from lenet.utils import get_device, get_dataset_download_dir


def train_model(config):
    try:
        device = get_device()
        model = LeNet().to(device)
        model.apply(init_weights)
        loss_fun = nn.CrossEntropyLoss()

        print("config", config)

        storage_path = Path(config['project_root']).joinpath("test-results")
        print("storage_path", storage_path)

        # Initialize optimizer
        if config['optimizer'] == "Adam":
            optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.9)
        elif config['optimizer'] == "SGD":
             optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
        else:
            raise ValueError(f"Optimizer {config['optimizer']} is not supported")

        initial_epoch = 0
        checkpoint = train.get_checkpoint()
        if checkpoint:
            with checkpoint.as_directory() as checkpoint_dir:
                checkpoint_dict = torch.load(os.path.join(checkpoint_dir, "checkpoint.pt"))
                initial_epoch = checkpoint_dict["epoch"] + 1
                model.load_state_dict(checkpoint_dict["model_state"])
                optimizer.load_state_dict(checkpoint_dict["optimizer_state_dict"])
        else:
            print("No model to preload, starting from scratch")

        download_path = Path(config['project_root']).joinpath(get_dataset_download_dir())

        for epoch in tqdm(range(initial_epoch, config['epochs'])):

            train_loss = 0.0
            train_dataloader = get_dataloader(batch_size=config['batch_size'], download_path=download_path, is_train=True)
            for i, data in enumerate(train_dataloader, 0):
                # get the inputs
                inputs, labels = data
                #  .to(device) is not in-place operation. It returns a new tensor,
                #  does not modify the original tensor and moves it to the specified device.
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients to prevent accumulation
                optimizer.zero_grad()

                outputs = model(inputs)
                loss = loss_fun(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            accuracy, eval_loss = evaluate_model(model, device, loss_fun, get_dataloader(config['batch_size'], download_path=download_path, is_train=False))
            train_loss /= len(train_dataloader)

            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()
            },
                os.path.join(storage_path, f"checkpoint_{epoch}.pt"),
            )
            metrics = {"train_loss": train_loss, "eval_loss": eval_loss, "accuracy": accuracy}
            train.report(metrics)
    finally:
        torch.cuda.empty_cache()  # Clean up GPU memory if applicable
        torch.mps.empty_cache()
        multiprocessing.active_children()  # Ensures child processes are cleaned up

def evaluate_model(model, device, loss_func, test_loader):
    correct_predictions = 0
    total_eval_samples = 0
    eval_loss = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = loss_func(outputs, labels)
            eval_loss += loss.item()
            _, predicted_indexes = torch.max(outputs, dim=1)
            total_eval_samples += labels.size(0)
            correct_predictions += (predicted_indexes == labels).sum()

    accuracy = 100 * correct_predictions.to("cpu").item() / total_eval_samples
    print(f'\nAccuracy:  {accuracy}%')
    return accuracy, eval_loss/len(test_loader)

def init_weights(m) -> None:
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
