import torch
from torch import nn
from torch.nn import functional as F

class Slava_net(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()

        self.conv1 = self._get_conv_block(in_channels, 16 - in_channels)
        self.conv2 = self._get_conv_block(16, 32 - in_channels)
        self.conv3 = self._get_conv_block(32, 64 - in_channels)
        self.conv4 = self._get_last_conv_block(64, 128)

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

    @staticmethod
    def _get_conv_block(c_in, c_out):

        block = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(c_out, c_out, 3, 1, 1),
            nn.
            ReLU(),
        )

        return block
    
    @staticmethod
    def _get_last_conv_block(c_in, c_out):

        block = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, 1, 0),
            nn.ReLU(),
            nn.Conv2d(c_out, c_out, 3, 1, 0),
            nn.ReLU(),
        )

        return block

        
