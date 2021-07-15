import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import torch.nn.functional as F
from doubleconv import DoubleConv


class UNetds(nn.Module):
    def __init__(self, in_channels, out_channels, features):
        super(UNetds, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.resol = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        #Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        #Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose3d(feature*2, feature, kernel_size=(2, 2, 2), stride=(2, 2, 2))
            )
            self.ups.append(DoubleConv(feature*2, feature))

        for feature in reversed(features):
            self.resol.append(
                nn.Conv3d(feature, out_channels, kernel_size=(1, 1, 1))
            )

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

    def forward(self, x):
        skip_connections = []
        res = []
        logits = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            concat_skip = torch.cat([skip_connection, x], 1)
            x = self.ups[idx+1](concat_skip)
            res.append(x)

        for ind in range(len(res)):
            x = res[ind]
            y = self.resol[ind](x)
            logits.append(y)

        return logits
