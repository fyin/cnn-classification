import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 1 input image channel, 6 output channels (filers), 5x5 square convolution
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # Input: 28 * 28 pixels -> 6*24*24 -> 6*12*12 -> 16*8*8 -> 16*4*4
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        # 10 output classes
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # flatten all dimensions to a single dimension for the fully connected layer
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # Since CrossEntropyLoss (which already contains softmax) is used for loss,
        # no need to apply softmax on output during training.
        # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        # Add softmax to the output for probabilities during inference if needed
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

