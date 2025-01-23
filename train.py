from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from config import get_latest_model_checkpoint, get_tb_writer_path, get_model_basename, get_config, get_model_checkpoint_dir
from dataset import get_dataloader
from model import LeNet
from tqdm import tqdm


def train_model(config):
    device = get_device()
    model = LeNet().to(device)
    model.apply(init_weights)
    loss_fun = nn.CrossEntropyLoss()

    # Initialize optimizer
    # optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    # Initialize Tensorboard writer
    writer = SummaryWriter(get_tb_writer_path(config))
    global_step = 0
    initial_epoch = 0
    Path(get_model_checkpoint_dir(config)).mkdir(parents=True, exist_ok=True)
    latest_model_checkpoint = get_latest_model_checkpoint(config)
    if latest_model_checkpoint:
        print(f"Preloading model from {latest_model_checkpoint}")
        state = torch.load(latest_model_checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print("No model to preload, starting from scratch")

    for epoch in tqdm(range(initial_epoch, config['num_epochs'])):

        train_loss = 0.0
        train_dataloader = get_dataloader(config, is_train=True)
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
            global_step += 1

        # Log model weights and gradients
        for name, param in model.named_parameters():
            writer.add_histogram(f'Weights/{name}', param, global_step)
            writer.add_histogram(f'Gradients/{name}', param.grad, global_step)

        accuracy, eval_loss = evaluate_model(model, device, loss_fun, get_dataloader(config, is_train=False))
        train_loss /= len(train_dataloader)
        # Log accuracy and losses
        writer.add_scalar('Eval Accuracy', accuracy, global_step)
        writer.add_scalars('Loss', {'Train': train_loss, 'Eval': eval_loss}, global_step)
        writer.flush()

        # Save model checkpoint for later reference and retraining
        model_basename = get_model_basename(config)
        model_checkpoint_filename = f"{model_basename}{epoch}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss,
            'global_step': global_step
        }, model_checkpoint_filename)

    writer.close()

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

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

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
            "NOTE: If you have a GPU, consider using it for training. Go to https://pytorch.org/get-started/locally/ for instructions.")
        print(
            "      On a Windows machine, Ex, run: conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia")
        print(
            "      On a Mac machine, Ex, run: conda install pytorch::pytorch torchvision torchaudio -c pytorch")

    device = torch.device(device)
    return device

if __name__ == '__main__':
    config = get_config()
    train_model(config)
