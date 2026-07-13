'''
Unet module from SPL-BEV

Project:
    EAST-SPL
    https://github.com/AbolfazlChM95/EAST-SPL/

Ref:
    https://github.com/IvarPersson/SPL-BEV 
'''

# Helper function
import torch
from torch import nn

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[8, 16, 32]):
        super(UNet, self).__init__()
        self.encoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.decoder = nn.ModuleList()
        self.upsample = nn.ModuleList()

        # Encoder
        for feature in features:
            self.encoder.append(self.double_conv(in_channels, feature))
            in_channels = feature

        # Decoder
        for feature in reversed(features):
            self.upsample.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.decoder.append(self.double_conv(feature * 2, feature))
        # Bottleneck
        self.bottleneck = self.double_conv(features[-1], features[-1] * 2)

        # Final layer
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        # Encoder path
        for enc in self.encoder:
            x = enc(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # Decoder path
        for idx, (up, dec) in enumerate(zip(self.upsample, self.decoder)):
            x = up(x)
            if x.shape != skip_connections[idx].shape:
                x = nn.functional.interpolate(x, size=skip_connections[idx].shape[2:])
            x = torch.cat((skip_connections[idx], x), dim=1)
            x = dec(x)
        return self.final_conv(x)

    @staticmethod
    def double_conv(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )