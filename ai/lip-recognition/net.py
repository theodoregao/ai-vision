import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleConv3(nn.Module):
    def __init__(self, nclasses):
        super(SimpleConv3, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=2)
        self.bn1 = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=2)
        self.bn2 = nn.BatchNorm2d(24)
        self.conv3 = nn.Conv2d(in_channels=24, out_channels=48, kernel_size=3, stride=2)
        self.bn3 = nn.BatchNorm2d(48)
        self.fc1 = nn.Linear(in_features=48 * 5 * 5, out_features=1200)
        self.fc2 = nn.Linear(in_features=1200, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=nclasses)

    def forward(self, x):
        y = F.relu(input=self.bn1(self.conv1(x)))
        y = F.relu(input=self.bn2(self.conv2(y)))
        y = F.relu(input=self.bn3(self.conv3(y)))
        y = y.view(-1, 48 * 5 * 5)
        y = F.relu(input=self.fc1(y))
        y = F.relu(input=self.fc2(y))
        y = self.fc3(y)
        return y


if __name__ == '__main__':
    x = torch.randn(1, 3, 48, 48)
    model = SimpleConv3(4)
    y = model(x)
    print(model)
