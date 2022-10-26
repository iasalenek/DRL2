import torch
from torch import nn
from torch.nn import functional as F

def _get_conv_block(c_in: int, 
                    c_out: int, 
                    batch_norm: bool = False, 
                    dropout: float = 0.1):

    layers = [
        nn.Conv2d(c_in, c_out, 3, 1, 1),
        nn.BatchNorm2d(c_out),
        nn.ReLU(),
        nn.Dropout2d(dropout),
        nn.Conv2d(c_out, c_out, 3, 1, 1),
        nn.BatchNorm2d(c_out),
        nn.ReLU(),
        nn.Dropout2d(dropout)
    ]

    if not batch_norm:
        layers = [x for x in layers if not isinstance(x, nn.BatchNorm2d)]

    return nn.Sequential(*layers)


def _get_last_conv_block(c_in: int, 
                         c_out: int, 
                         batch_norm: bool = False, 
                         dropout: float = 0.1):

    layers = [
        nn.Conv2d(c_in, c_out, 3, 1, 0),
        nn.BatchNorm2d(c_out),
        nn.ReLU(),
        nn.Dropout2d(dropout),
        nn.Conv2d(c_out, c_out, 3, 1, 0),
        nn.ReLU(),
        nn.Dropout2d(dropout)
    ]

    if not batch_norm:
        layers = [x for x in layers if not isinstance(x, nn.BatchNorm2d)]

    return nn.Sequential(*layers)


class Simple_conv(nn.Module):
    def __init__(self, 
                 in_channels: int,
                 batch_norm: bool = False, 
                 dropout: float = 0.1):

        super().__init__()

        self.conv1 = _get_conv_block(in_channels, 32, batch_norm, dropout)
        self.conv2 = _get_conv_block(32, 64, batch_norm, dropout)
        self.conv3 = _get_conv_block(64, 128, batch_norm, dropout)
        self.conv4 = _get_last_conv_block(128, 256, batch_norm, dropout)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 5)
        )

    def forward(self, input):

        x = self.pool(self.conv1(input))
        x = self.pool(self.conv2(x))
        x = self.pool(self.conv3(x))
        x = self.conv4(x)
        x = self.fc(x)

        return x


class Slava_net(nn.Module):
    def __init__(self, in_channels: int, batch_norm=False, dropout=0.1):
        super().__init__()

        self.conv1 = _get_conv_block(
            in_channels, 16 - in_channels, batch_norm, dropout)
        self.conv2 = _get_conv_block(
            16, 32 - in_channels, batch_norm, dropout)
        self.conv3 = _get_conv_block(
            32, 64 - in_channels, batch_norm, dropout)
        self.conv4 = _get_last_conv_block(
            64, 128, batch_norm, dropout)

        self.maxpool = nn.MaxPool2d(2)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 5)
        )

    def forward(self, input):

        x = self.maxpool(self.conv1(input))
        x = torch.cat((x, input[:, :, 10:30, 10:30]), dim=1)
        x = self.maxpool(self.conv2(x))
        x = torch.cat((x, input[:, :, 15:25, 15:25]), dim=1)
        x = self.maxpool(self.conv3(x))
        x = torch.cat((x, input[:, :, 18:23, 18:23]), dim=1)
        x = self.conv4(x)
        x = self.fc(x)

        return x
