import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.5):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        if dropout:
            self.double_conv.append(nn.Dropout2d(p=dropout))

    def forward(self, x):
        return self.double_conv(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.max_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):

        return self.max_conv(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)  # After concatenation

    def forward(self, x, skip_features):
        x = self.up(x)
        if x.shape != skip_features.shape:
            x = torch.nn.functional.interpolate(
                x, size=skip_features.shape[-2:], mode="bilinear", align_corners=True
            )
        x = torch.cat([x, skip_features], dim=1)  # Concatenate skip features
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, features=[64, 128, 256, 512]):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features

        self.encoder_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()

        self.init_conv = DoubleConv(in_channels, features[0])
        # Create encoder path

        for i in range(1, len(features)):
            self.encoder_blocks.append(EncoderBlock(features[i - 1], features[i]))

        # Center part (bottleneck)
        self.center = DoubleConv(features[-1], features[-1] * 2)

        # Create decoder path
        for feature in reversed(features):
            self.decoder_blocks.append(DecoderBlock(feature * 2, feature))

        # Final segmentation layer
        self.segmentation_head = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        x = self.init_conv(x)
        skip_connections.append(x)
        # Encoder path
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)
            skip_connections.append(x)

        # Bottleneck
        x = self.center(x)

        # Decoder path
        skip_connections = skip_connections[::-1]
        for i, decoder_block in enumerate(self.decoder_blocks):
            skip = skip_connections[i]
            x = decoder_block(x, skip)

        # Final segmentation map
        return self.segmentation_head(x)
