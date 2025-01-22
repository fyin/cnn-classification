import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from config import get_config

def get_dataloader(config, is_train):
    # (0.5,), one is for Mean normalization values, the other is for Standard deviation values for each channel.
    # Normalize pixel to [-1, 1]
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))])

    # https://pytorch.org/vision/main/generated/torchvision.datasets.MNIST.html
    # https: // www.kaggle.com / datasets / hojjatk / mnist - dataset
    dataset = torchvision.datasets.MNIST(root='./data', train=is_train,
                                           download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config['batch_size'],
                                          shuffle=True, num_workers=config['num_dataloader_workers'])
    return dataloader

def image_class_list():
    return ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

def imshow(img):
    # unnormalize the normalized pixel (with 0.5 for mean and standard deviation)
    img = img / 2 + 0.5
    np_img = img.numpy()
    # image with a tensor of shape [Channel, Height, Width], but matplotlib expects the data with a format of [Height, Width, Channel])
    # So it needs to transpose.
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    # Call plt.show() explicitly to display the image on the screen when running script
    plt.show()


if __name__ == '__main__':
    config = get_config()
    train_loader = get_dataloader(config, is_train=True)
    classes = image_class_list()
    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    # show images
    imshow(torchvision.utils.make_grid(images))
    print(' '.join('%5s' % classes[labels[j]] for j in range(config['batch_size'])))