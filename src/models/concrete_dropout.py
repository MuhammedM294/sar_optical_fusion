import torch
import torch.nn as nn
import torch.nn.functional as F


class ConcreteDropout2d(nn.Module):
    """
    Concrete Dropout for 2D inputs (Conv2d feature maps).
    Learns dropout probability during training.
    """

    def __init__(self, weight_regularizer=1e-6, dropout_regularizer=1e-5):
        super().__init__()
        # Trainable logit of dropout probability
        self.logit_p = nn.Parameter(torch.tensor(-2.0))
        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        self.regularization = 0.0

    def forward(self, x, layer):
        """
        Apply concrete dropout before a given Conv2d layer.
        Args:
            x: input tensor
            layer: nn.Conv2d layer
        """
        eps = 1e-7
        temp = 0.1

        # Learnable dropout probability
        p = torch.sigmoid(self.logit_p)

        # Sample mask from Concrete distribution
        u = torch.rand_like(x)
        drop_prob = (
            torch.log(p + eps)
            - torch.log(1 - p + eps)
            + torch.log(u + eps)
            - torch.log(1 - u + eps)
        )
        drop_prob = torch.sigmoid(drop_prob / temp)
        retain_prob = 1 - drop_prob

        # Apply dropout mask (reparameterization trick)
        x = x * retain_prob
        x = x / (1 - p + eps)  # rescale

        # Apply conv layer
        out = layer(x)

        # Regularization loss
        weight_reg = self.weight_regularizer * torch.sum(layer.weight**2) / (1 - p)
        # dropout_reg = p * torch.log(p + eps) + (1 - p) * torch.log(1 - p + eps)
        # Scale by number of features per sample
        # dropout_reg *= self.dropout_regularizer * (x.numel() / x.shape[0])

        dropout_reg = -(p * torch.log(p + eps) + (1 - p) * torch.log(1 - p + eps))
        dropout_reg *= self.dropout_regularizer * (x.numel() / x.shape[0])

        self.regularization = weight_reg + dropout_reg

        return out


class DoubleConv(nn.Module):
    """(Conv2D -> BN -> ReLU) * 2 with optional Concrete Dropout."""

    def __init__(self, in_channels, out_channels, use_concrete_dropout=True):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.use_concrete_dropout = use_concrete_dropout
        if use_concrete_dropout:
            self.cd1 = ConcreteDropout2d()
            self.cd2 = ConcreteDropout2d()
        else:
            self.dropout = nn.Dropout2d(0.5)

        self.regularization = 0.0

    def forward(self, x):
        if self.use_concrete_dropout:
            x = F.relu(self.bn1(self.cd1(x, self.conv1)))
            x = F.relu(self.bn2(self.cd2(x, self.conv2)))
            self.regularization = self.cd1.regularization + self.cd2.regularization
        else:
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = self.dropout(x)
            self.regularization = 0.0
        return x


class EncoderBlock(nn.Module):
    """Downsampling block: MaxPool + DoubleConv."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.double_conv = DoubleConv(
            in_channels, out_channels, use_concrete_dropout=True
        )

    def forward(self, x):
        x = self.maxpool(x)
        x = self.double_conv(x)
        return x


class DecoderBlock(nn.Module):
    """Upsampling block: ConvTranspose2d + DoubleConv."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.double_conv = DoubleConv(
            in_channels, out_channels, use_concrete_dropout=True
        )

    def forward(self, x, skip_features):
        x = self.up(x)
        if x.shape != skip_features.shape:
            x = F.interpolate(
                x, size=skip_features.shape[-2:], mode="bilinear", align_corners=True
            )
        x = torch.cat([x, skip_features], dim=1)
        x = self.double_conv(x)
        return x


class UNet(nn.Module):
    """U-Net with Concrete Dropout in convolutional blocks."""

    def __init__(self, in_channels, out_channels, features=[64, 128, 256, 512]):
        super().__init__()
        self.encoder_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()

        # Initial conv
        self.init_conv = DoubleConv(in_channels, features[0], use_concrete_dropout=True)

        # Encoder
        for i in range(1, len(features)):
            self.encoder_blocks.append(EncoderBlock(features[i - 1], features[i]))

        # Bottleneck
        self.center = DoubleConv(
            features[-1], features[-1] * 2, use_concrete_dropout=True
        )

        # Decoder
        for feature in reversed(features):
            self.decoder_blocks.append(DecoderBlock(feature * 2, feature))

        # Output segmentation head
        self.segmentation_head = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        regularization_loss = 0.0

        # Encoder
        x = self.init_conv(x)
        skip_connections.append(x)
        regularization_loss += self.init_conv.regularization
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)
            skip_connections.append(x)
            # collect reg loss
            regularization_loss += encoder_block.double_conv.regularization

        # Bottleneck
        x = self.center(x)
        regularization_loss += self.center.regularization

        # Decoder
        skip_connections = skip_connections[::-1]
        for i, decoder_block in enumerate(self.decoder_blocks):
            skip = skip_connections[i]
            x = decoder_block(x, skip)
            regularization_loss += decoder_block.double_conv.regularization

        # Output
        out = self.segmentation_head(x)
        return out, regularization_loss


# -------------------------
# 🔍 Quick test
# -------------------------
if __name__ == "__main__":
    model = UNet(in_channels=3, out_channels=2)
    x = torch.randn(2, 3, 128, 128)  # batch of 2
    y, reg_loss = model(x)
    print("Output shape:", y.shape)
    print("Regularization loss:", reg_loss.item())
