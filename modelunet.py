import torch
import torch.nn as nn
from doubleconv import DoubleConv


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, features):
        super(UNet, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
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

        self.outs = nn.Conv3d(features[0], out_channels, kernel_size=(1, 1, 1))
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

    def forward(self, x):
        skip_connections = []
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

        return self.outs(x)
